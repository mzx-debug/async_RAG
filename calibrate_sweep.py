#!/usr/bin/env python3
"""
Calibration sweep: profile gen_time = f(batch_size) for each (xE, xR) combination.

Uses the LINEAR model (gen_time = gen_base + gen_per_q × batch_size), which
fits vLLM continuous batching with R²=0.999999 across bs=32/64/256.

Usage:
    python calibrate_sweep.py                                    # sweep all actions
    python calibrate_sweep.py --action 0 0 --batch-sizes 1 4 16 32 64 128
    python calibrate_sweep.py --workdir /path/to/async_RAG
    python calibrate_sweep.py --dry-run
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path


BATCH_SIZES = [1, 4, 16, 32, 64, 128]
DEFAULT_ACTIONS = [(0, 0), (0, 1), (1, 0), (1, 1)]


def run_experiment(
    xE: int,
    xR: int,
    batch_size: int,
    sample_queries: int,
    script: Path,
    workdir: Path,
    out_path: Path,
) -> Path:
    if out_path.exists():
        print(f"  [SKIP] {out_path.name} exists")
        return out_path

    cmd = [
        sys.executable, str(script),
        "--xE", str(xE), "--xR", str(xR),
        "--b", str(batch_size),
        "--sample-queries", str(sample_queries),
        "--pipeline-mode", "async_plain",
        "--index-path", str(workdir / "indexes/beir_nfcorpus/faiss.index"),
        "--corpus-path", str(workdir / "data/beir_nfcorpus/corpus.jsonl"),
        "--queries-file", str(workdir / "data/beir_nfcorpus/queries.jsonl"),
        "--generator-model", "Qwen/Qwen2.5-1.5B-Instruct",
        "--output-json", str(out_path),
        "--max-model-len", "8192",
        "--gpu-memory-utilization", "0.6",
        "--gpu-id", "0",
        "--fixed-action",
    ]

    env = os.environ.copy()
    env["HF_ENDPOINT"] = "https://hf-mirror.com"
    activate = "/home/cloudteam/Software/conda/envs/p702/bin/python"
    full_cmd = " ".join(cmd)

    print(f"  Running: xE={xE}, xR={xR}, b={batch_size}...")
    result = subprocess.run(
        [activate] + cmd[1:], env=env, cwd=str(workdir),
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        print(f"  [ERROR] exit={result.returncode}")
        print(f"  stderr: {result.stderr[-500:]}")
    return out_path


def extract_data(json_path: Path):
    """Extract (batch_size, gen_ms, ret_ms) points from a result file.

    gen_ms and ret_ms are TOTAL times for the batch (not per-query).
    """
    try:
        with open(json_path) as f:
            d = json.load(f)
    except Exception as e:
        print(f"  [ERROR reading {json_path}]: {e}")
        return []

    points = []
    for b in d.get("per_batch", []):
        bs = b["batch_size"]
        gen_sec = b["generation_sec"]
        ret_sec = b["retrieval_sec"]
        points.append((bs, gen_sec * 1000, ret_sec * 1000))  # (bs, gen_ms, ret_ms)

    return points


def fit_linear(points):
    """Fit: gen_time_ms = gen_base + gen_per_q × batch_size.

    Returns (gen_base_ms, gen_per_q_ms_per_query, r_squared).
    """
    if len(points) < 2:
        return None, None, None

    n = len(points)
    sx = sum(p[0] for p in points)
    sy = sum(p[1] for p in points)
    sxy = sum(p[0] * p[1] for p in points)
    sxx = sum(p[0] ** 2 for p in points)

    denom = n * sxx - sx * sx
    if abs(denom) < 1e-9:
        return None, None, None

    b_coef = (n * sxy - sx * sy) / denom          # gen_per_q
    a_coef = (sy - b_coef * sx) / n                # gen_base

    y_mean = sy / n
    ss_tot = sum((p[1] - y_mean) ** 2 for p in points)
    ss_res = sum((p[1] - (a_coef + b_coef * p[0])) ** 2 for p in points)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return max(0.0, a_coef), max(0.0, b_coef), r2


def fit_overlap_model(xe0_results):
    """
    Fit queue_penalty and er_overlap_coef from xE=0 sweep data.

    Model (per-query):
        wall_q - gen_q = queue_penalty + er_overlap_coef * ret_q

    y = a + b * x
      y = wall_q - gen_q  (overhead above generation)
      x = ret_q           (retrieval time per query)
      a = queue_penalty
      b = er_overlap_coef

    er_overlap_penalty = b * median(ret_q) — the overlap residual
    expressed as a per-query overhead.

    Returns (queue_penalty, er_overlap_penalty, r_squared).
    """
    # Collect (ret_q, overhead) points from xE=0 data
    points = []
    for (xE, xR), batch_data in xe0_results.items():
        if xE != 0:
            continue
        for bs, gen_ms, ret_ms in batch_data:
            wall_q = (gen_ms + ret_ms) / bs
            gen_q = gen_ms / bs
            ret_q = ret_ms / bs
            overhead = wall_q - gen_q  # = queue + er_overlap * ret_q
            if 0 < overhead < 100 and ret_q > 0:
                points.append((ret_q, overhead))

    if len(points) < 2:
        return None, None, None

    # Linear regression: overhead = a + b * ret_q
    n = len(points)
    sx = sum(x for x, y in points)
    sy = sum(y for x, y in points)
    sxy = sum(x * y for x, y in points)
    sxx = sum(x * x for x, y in points)

    denom = n * sxx - sx * sx
    if abs(denom) < 1e-9:
        return None, None, None

    b_coef = (n * sxy - sx * sy) / denom       # er_overlap_coef
    a_coef = (sy - b_coef * sx) / n             # queue_penalty

    # R^2
    y_mean = sy / n
    ss_tot = sum((y - y_mean) ** 2 for x, y in points)
    ss_res = sum((y - (a_coef + b_coef * x)) ** 2 for x, y in points)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else 0.0

    # er_overlap_penalty = overlap_coef * median(ret_q)
    median_ret_q = sorted(points)[len(points) // 2][0]
    er_overlap_penalty = max(0.0, b_coef) * median_ret_q

    queue_penalty = max(0.0, a_coef)
    er_overlap_penalty = max(0.0, er_overlap_penalty)

    return queue_penalty, er_overlap_penalty, r2


def print_results_table(results):
    """Print per-action calibration results."""
    print()
    print("=" * 80)
    print("Calibration Results")
    print("=" * 80)

    for (xE, xR), points in sorted(results.items()):
        if not points:
            continue

        gen_base, gen_per_q, r2 = fit_linear(points)
        desc = {
            (0, 0): "CPU E+R",
            (0, 1): "GPU ret",
            (1, 0): "GPU emb",
            (1, 1): "GPU E+R",
        }.get((xE, xR), "")

        print(f"\n### ({xE},{xR}) {desc}")
        print(f"  {'bs':>5} | {'gen_ms (total)':>16} | {'gen_ms/q':>10}")
        print(f"  {'-' * 45}")
        for bs, gen_ms in sorted(points):
            print(f"  {bs:>5} | {gen_ms:>16.1f} | {gen_ms / bs:>10.1f}")

        if gen_base is not None:
            print(f"\n  Linear model: gen = {gen_base:.0f} + {gen_per_q:.1f} × bs   (R²={r2:.6f})")
            print(f"  gen_base    = {gen_base:.0f} ms  (prefill + kernel launch overhead)")
            print(f"  gen_per_q   = {gen_per_q:.1f} ms/q  (marginal cost per query)")
            print(f"  predictions:")
            for bs_t in [1, 4, 16, 32, 64, 128, 256]:
                pred = gen_base + gen_per_q * bs_t
                print(f"    b={bs_t:>3}: gen={pred:>7.0f}ms  score={pred / bs_t:>6.1f}ms/q")
        else:
            print(f"\n  Not enough data points to fit model")

    # Summary table
    print()
    print("### Summary: linear model coefficients")
    print(f"  {'Action':>10} | {'gen_base':>10} | {'gen_per_q':>12} | {'R²':>8} | {'b=128 score':>12}")
    print(f"  {'-' * 65}")
    for (xE, xR), points in sorted(results.items()):
        if not points:
            continue
        gen_base, gen_per_q, r2 = fit_linear(points)
        action = f"({xE},{xR})"
        if gen_base is not None:
            pred_128 = gen_base + gen_per_q * 128
            print(f"  {action:>10} | {gen_base:>10.0f} | {gen_per_q:>12.1f} | {r2:>8.5f} | {pred_128/128:>12.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="Calibration sweep: profile gen_time = f(batch_size) per (xE, xR). "
                    "Uses the LINEAR model: gen_time = gen_base + gen_per_q × batch_size."
    )
    parser.add_argument(
        "--workdir", type=str, default=None,
        help="Directory containing async_rag_pipeline.py (default: this script's directory)"
    )
    parser.add_argument(
        "--action", nargs=2, type=int, action="append",
        help="Specific (xE xR) to calibrate. Can be repeated. "
             "Example: --action 0 1 --action 1 0"
    )
    parser.add_argument(
        "--batch-sizes", nargs="+", type=int, default=BATCH_SIZES,
        help=f"Batch sizes to sweep (default: {BATCH_SIZES})"
    )
    parser.add_argument(
        "--sample-queries", type=int, default=256,
        help="Number of queries per run (default: 256)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print planned experiments without running"
    )
    parser.add_argument(
        "--avg-output-tokens", type=float, default=120.0,
        help="Average output tokens per query (default: 120.0)"
    )
    args = parser.parse_args()

    # Resolve workdir
    script_dir = Path(__file__).parent.resolve()
    workdir = Path(args.workdir).resolve() if args.workdir else script_dir
    script = workdir / "async_rag_pipeline.py"
    if not script.exists():
        print(f"ERROR: async_rag_pipeline.py not found at {script}")
        sys.exit(1)

    output_dir = workdir / "output" / "calibration_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)

    actions = [tuple(a) for a in args.action] if args.action else DEFAULT_ACTIONS
    b_sizes = sorted(args.batch_sizes)

    print(f"Workdir:     {workdir}")
    print(f"Script:      {script}")
    print(f"Output dir:  {output_dir}")
    print(f"Actions:     {actions}")
    print(f"Batch sizes: {b_sizes}")
    print(f"Sample queries: {args.sample_queries}")
    print()

    results = {(xE, xR): [] for (xE, xR) in actions}

    if args.dry_run:
        print("Dry run — would run these experiments:")
        for (xE, xR) in actions:
            for b in b_sizes:
                print(f"  ({xE},{xR}) b={b}")
        return

    total_runs = len(actions) * len(b_sizes)
    run = 0
    for (xE, xR) in actions:
        print(f"\n=== Calibrating ({xE}, {xR}) ===")
        for b in b_sizes:
            run += 1
            out_path = output_dir / f"calib_{xE}_{xR}_b{b}.json"
            print(f"[{run}/{total_runs}] xE={xE}, xR={xR}, b={b}")
            run_experiment(xE, xR, b, args.sample_queries, script, workdir, out_path)
            time.sleep(1)

            points = extract_data(out_path)
            if points:
                results[(xE, xR)].extend(points)
                for bs, gen_ms, ret_ms in points:
                    print(f"  → bs={bs}: gen={gen_ms:.0f}ms ret={ret_ms:.0f}ms")

    # Save raw data
    raw_path = output_dir / "raw_results.json"
    serializable = {
        f"({k[0]},{k[1]})": [(b, g, r) for b, g, r in pts]
        for k, pts in results.items()
    }
    with open(raw_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nRaw data saved to: {raw_path}")

    # Fit and print per-action linear model
    print_results_table(results)

    # ── Fit queue + er_overlap from xE=0 sweep data ───────────────────
    print()
    print("=" * 80)
    print("Queue + E+R Overlap (xE=0 sweep fit)")
    print("=" * 80)
    print("Model: wall_q - gen_q = queue_penalty + er_overlap_coef * ret_q")
    print()

    queue_penalty, er_overlap_penalty, r2 = fit_overlap_model(results)
    if queue_penalty is not None:
        print(f"  queue_penalty         = {queue_penalty:.2f} ms/q  (intercept)")
        print(f"  er_overlap_penalty   = {er_overlap_penalty:.2f} ms/q  (overlap coefficient * median ret_q)")
        print(f"  R²                   = {r2:.4f}")
    else:
        print("  Not enough xE=0 data to fit overlap model")
        queue_penalty = 2.5
        er_overlap_penalty = 0.0

    # Save all fitted coefficients (v4 params)
    coeffs = {}
    for (xE, xR), points in results.items():
        gen_base, gen_per_q, gen_r2 = fit_linear(points)
        coeffs[f"({xE},{xR})"] = {
            "gen_base": gen_base,
            "gen_per_query": gen_per_q,
            "gen_r_squared": gen_r2,
            "data_points": [(b, g, r) for b, g, r in points],
        }

    coeffs["__meta__"] = {
        "avg_output_tokens": args.avg_output_tokens,
        "queue_penalty": queue_penalty,
        "er_overlap_penalty": er_overlap_penalty,
        "overlap_r_squared": r2,
    }

    coeffs_path = output_dir / "fitted_coefficients.json"
    with open(coeffs_path, "w") as f:
        json.dump(coeffs, f, indent=2)
    print(f"\nFitted coefficients saved to: {coeffs_path}")


if __name__ == "__main__":
    main()
