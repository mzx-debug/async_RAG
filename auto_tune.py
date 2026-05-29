#!/usr/bin/env python3
"""
auto_tune.py — One-shot calibration for the Async RAG pipeline.

Run this FIRST on any new device. It:
  1. Auto-detects GPU model and VRAM
  2. Selects an appropriate generator model and GPU memory utilization
  3. Feasibility-tests each (xE, xR) combination
  4. Feasibility-tests each (xE, xR) combination
  5. Runs a calibration sweep (multiple batch sizes) for each feasible action
  6. Fits gen_q = P_fixed/B + D_fixed (+ C_tok * L_in when variance is high)
  7. Saves a ready-to-use EMA parameters file and a tuning report

Usage:
    python auto_tune.py                                   # auto-detect device, sweep all actions
    python auto_tune.py --action 0 0                     # sweep only (xE=0, xR=0)
    python auto_tune.py --dry-run                         # show plan without running
    python auto_tune.py --model Qwen/Qwen2.5-3B-Instruct  # override generator model
    python auto_tune.py --sample-queries 128             # use fewer queries (faster)
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).parent.resolve()
OUTPUT_DIR = SCRIPT_DIR / "output" / "auto_tune"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Conda environment path (used to inject CUDA environment variables)
CONDA_PREFIX = "/home/cloudteam/Software/conda/envs/p702"


def _get_python() -> str:
    """Return the python executable from the target conda environment."""
    return str(Path(CONDA_PREFIX) / "bin" / "python")


def _build_env() -> dict:
    """Build subprocess environment with CUDA variables from the conda environment.

    Uses shell=False so we inject CUDA variables directly instead of relying
    on conda activation in a shell subprocess (which cannot affect the parent).
    """
    env = os.environ.copy()
    env["HF_ENDPOINT"] = "https://hf-mirror.com"
    conda_lib = os.path.join(CONDA_PREFIX, "lib")
    if "LD_LIBRARY_PATH" in env:
        if conda_lib not in env["LD_LIBRARY_PATH"]:
            env["LD_LIBRARY_PATH"] = conda_lib + ":" + env["LD_LIBRARY_PATH"]
    else:
        env["LD_LIBRARY_PATH"] = conda_lib
    env["MKL_THREADING_LAYER"] = "GNU"
    env["OPENBLAS_NUM_THREADS"] = "4"
    env["OMP_NUM_THREADS"] = "4"
    env["LD_PRELOAD"] = ""
    return env


# ── Device detection ──────────────────────────────────────────────────────────

def detect_gpu() -> Dict:
    """Detect GPU properties. Returns empty dict if no GPU."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {}
        idx = 0
        props = torch.cuda.get_device_properties(idx)
        total_mem_gb = props.total_memory / (1024 ** 3)
        name = props.name
        compute_capability = f"{props.major}.{props.minor}"
        return {
            "name": name,
            "total_mem_gb": round(total_mem_gb, 1),
            "compute_capability": compute_capability,
            "device_idx": idx,
        }
    except Exception as e:
        return {}


def recommend_config(gpu: Dict) -> Dict:
    """Select generator model and GPU util based on VRAM."""
    mem_gb = gpu.get("total_mem_gb", 0)

    if mem_gb <= 0:
        return {
            "model": "Qwen/Qwen2.5-1.5B-Instruct",
            "gpu_util": 0.6,
            "reason": "No GPU detected — will use CPU inference (very slow, not recommended)",
        }

    if mem_gb < 5:
        return {
            "model": "Qwen/Qwen2.5-1.5B-Instruct",
            "gpu_util": 0.5,
            "reason": f"{mem_gb}GB — 1.5B model with conservative KV cache",
        }
    elif mem_gb < 7:
        return {
            "model": "Qwen/Qwen2.5-1.5B-Instruct",
            "gpu_util": 0.7,
            "reason": f"{mem_gb}GB — 1.5B model, comfortable headroom",
        }
    elif mem_gb < 10:
        return {
            "model": "Qwen/Qwen2.5-3B-Instruct",
            "gpu_util": 0.6,
            "reason": f"{mem_gb}GB — 3B model fits with moderate util",
        }
    elif mem_gb < 16:
        return {
            "model": "Qwen/Qwen2.5-3B-Instruct",
            "gpu_util": 0.8,
            "reason": f"{mem_gb}GB — 3B model, generous KV cache",
        }
    else:
        return {
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "gpu_util": 0.6,
            "reason": f"{mem_gb}GB — 7B model recommended for large devices",
        }


def detect_corpus() -> Optional[Path]:
    """Find an existing corpus. Prefers nfcorpus (smallest)."""
    candidates = [
        SCRIPT_DIR / "data/beir_nfcorpus",
        SCRIPT_DIR / "data/beir_scifact",
        SCRIPT_DIR / "data/beir_arguana",
    ]
    for d in candidates:
        if d.exists() and (d / "queries.jsonl").exists():
            return d
    return None


# ── Feasibility test ───────────────────────────────────────────────────────────

def test_action_feasible(
    xE: int,
    xR: int,
    gpu_util: float,
    model: str,
    script: Path,
    workdir: Path,
    test_batch_size: int = 32,
) -> Tuple[bool, str]:
    """Quick test to check if an action runs without OOM at given batch size."""
    out_path = OUTPUT_DIR / f"feasibility_{xE}_{xR}_b{test_batch_size}.json"

    cmd = [
        _get_python(), str(script),
        "--xE", str(xE), "--xR", str(xR),
        "--b", str(test_batch_size),
        "--sample-queries", str(test_batch_size),
        "--pipeline-mode", "async_v2",
        "--index-path", str(workdir / "indexes/beir_nfcorpus/faiss.index"),
        "--corpus-path", str(workdir / "data/beir_nfcorpus/corpus.jsonl"),
        "--queries-file", str(workdir / "data/beir_nfcorpus/queries.jsonl"),
        "--generator-model", model,
        "--output-json", str(out_path),
        "--max-model-len", "8192",
        "--gpu-memory-utilization", str(gpu_util),
        "--gpu-id", "0",
        "--log-interval", "9999",
        "--show-samples", "0",
    ]

    env = _build_env()
    result = subprocess.run(
        cmd, shell=False, env=env, cwd=str(workdir),
        capture_output=True, text=True, timeout=300,
    )

    if result.returncode == 0 and out_path.exists():
        return True, "OK"
    else:
        err = result.stderr[-300:] if result.stderr else "unknown error"
        return False, err.strip()


# ── Calibration run ───────────────────────────────────────────────────────────

def run_calibration(
    xE: int,
    xR: int,
    batch_size: int,
    gpu_util: float,
    model: str,
    sample_queries: int,
    script: Path,
    workdir: Path,
) -> Optional[Dict]:
    """Run one calibration point and extract timing data."""
    out_path = OUTPUT_DIR / f"calib_{xE}_{xR}_b{batch_size}.json"

    if out_path.exists():
        print(f"  [SKIP] {out_path.name} exists (use --force to re-run)")
        return extract_calibration_data(out_path)

    cmd = [
        _get_python(), str(script),
        "--xE", str(xE), "--xR", str(xR),
        "--b", str(batch_size),
        "--sample-queries", str(sample_queries),
        "--pipeline-mode", "async_v2",
        "--index-path", str(workdir / "indexes/beir_nfcorpus/faiss.index"),
        "--corpus-path", str(workdir / "data/beir_nfcorpus/corpus.jsonl"),
        "--queries-file", str(workdir / "data/beir_nfcorpus/queries.jsonl"),
        "--generator-model", model,
        "--output-json", str(out_path),
        "--max-model-len", "8192",
        "--gpu-memory-utilization", str(gpu_util),
        "--gpu-id", "0",
        "--fixed-action",
        "--log-interval", "9999",
        "--show-samples", "0",
    ]

    env = _build_env()

    print(f"  Running: xE={xE}, xR={xR}, b={batch_size}...")
    result = subprocess.run(
        cmd, shell=False, env=env, cwd=str(workdir),
        capture_output=True, text=True, timeout=600,
    )

    if result.returncode != 0:
        print(f"  [ERROR] exit={result.returncode}: {result.stderr[-200:]}")
        return None

    return extract_calibration_data(out_path)


def extract_calibration_data(json_path: Path) -> Optional[Dict]:
    """Extract (batch_size, gen_time_ms) from a calibration result file."""
    try:
        with open(json_path) as f:
            d = json.load(f)

        points = []
        for b in d.get("per_batch", []):
            bs = b["batch_size"]
            gen_sec = b["generation_sec"]
            emb_sec = b["embedding_sec"]
            ret_sec = b["retrieval_sec"]
            points.append({
                "batch_size": bs,
                "gen_total_ms": gen_sec * 1000,
                "emb_total_ms": emb_sec * 1000,
                "ret_total_ms": ret_sec * 1000,
                "gen_per_q_ms": gen_sec * 1000 / bs,
                "emb_per_q_ms": emb_sec * 1000 / bs,
                "ret_per_q_ms": ret_sec * 1000 / bs,
            })

        if not points:
            return None

        return {
            "wall_time_ms": d["wall_time_ms"],
            "wall_throughput_qps": d["wall_throughput_qps"],
            "points": points,
        }
    except Exception as e:
        print(f"  [ERROR reading {json_path}]: {e}")
        return None


# ── Hyperbolic (two-segment) fitting ──────────────────────────────────────────

def fit_hyperbolic(points: List[Tuple[float, float]]) -> Tuple[float, float, float]:
    """
    Fit gen_q = P/B + D  (gen per query, ms/q).

    Input points: [(B, gen_total_ms), ...]
    Returns (P, D, r_squared).
    - P (ms): fixed prefill/launch overhead
    - D (ms/q): marginal per-query decode cost
    """
    if len(points) < 2:
        return 0.0, 0.0, 0.0

    n = len(points)
    # We need to solve for P, D in: gen_total = P + D * B
    sx = sum(p[0] for p in points)
    sy = sum(p[1] for p in points)
    sxy = sum(p[0] * p[1] for p in points)
    sxx = sum(p[0] ** 2 for p in points)

    denom = n * sxx - sx * sx
    if abs(denom) < 1e-9:
        return 0.0, 0.0, 0.0

    D = (n * sxy - sx * sy) / denom   # slope = D
    P = (sy - D * sx) / n             # intercept = P

    P = max(0.0, P)
    D = max(0.0, D)

    y_mean = sy / n
    ss_tot = sum((p[1] - y_mean) ** 2 for p in points)
    ss_res = sum((p[1] - (P + D * p[0])) ** 2 for p in points)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return P, D, r2


def calibrate_action(
    data: Dict,
    L_in_variance_threshold: float = 10.0,
) -> Dict:
    """
    Calibrate gen_q = P/B + D  (two-segment model).

    Returns dict with P_fixed, D_fixed, best_batch_size, best_score,
    and optionally P_per_token if input-length variance is high enough.
    """
    points_raw = [(pt["batch_size"], pt["gen_total_ms"]) for pt in data["points"]]
    if not points_raw:
        return {"P_fixed": 0.0, "D_fixed": 0.0, "P_per_token": 0.0,
                "best_batch_size": 4, "best_score": 0.0}

    P, D, r2 = fit_hyperbolic(points_raw)

    # P_per_token: fit only if L_in variance is high (nfcorpus is too uniform ~3-38)
    L_in_avg = 30.0  # default; from data["points"][0].get("L_in_avg", 30.0)
    P_tok = 0.0
    if len(points_raw) >= 2:
        b1, g1 = points_raw[0]
        b2, g2 = points_raw[1]
        gq1, gq2 = g1 / b1, g2 / b2
        denom = (1.0 / b2 - 1.0 / b1)
        if abs(denom) > 1e-9:
            P_tok = max(0.0, (gq2 - gq1) / denom)

    # Best batch size: minimize gen_q = P/B + D
    batch_sizes = [pt["batch_size"] for pt in data["points"]]
    best_bs = min(batch_sizes, key=lambda b: P / b + D) if P > 0 or D > 0 else 4
    best_score = P / best_bs + D if best_bs > 0 else 0.0

    return {
        "P_fixed": P,
        "D_fixed": D,
        "P_per_token": P_tok,
        "r2": r2,
        "best_batch_size": best_bs,
        "best_score": best_score,
    }


# ── Reporting ─────────────────────────────────────────────────────────────────

def build_ema_params(
    results: Dict[Tuple[int, int], Dict],
    gpu_util: float,
    model: str,
) -> Dict:
    """Build a ready-to-use EMA params JSON from calibration results.

    gen_q = (P/B + D) * contention_ema
    contention_ema is seeded at 1.0 per (xE, xR) and updated online.
    """
    ema = {
        "gen_base_overhead_ema": {},   # P_fixed per (xE, xR)
        "gen_per_query_ema": {},        # D_fixed per (xE, xR)
        "gen_prefill_per_token_ema": {},  # C_tok per (xE, xR), 0 if not fitted
        "contention_ema": {},          # contention ratio per (xE, xR)
        "er_base_overhead_ema": 0.0,
        "embedding_latency_ema": {},
        "retrieval_latency_ema": {},
        "overlap_factor_ema": {},
        "batch_size_residual_ema": {},
        "max_batch_size_ema": 32.0,
        "best_batch_size_by_action": {},
        "wall_time_measurements": {},
        "feasible_actions": {},
    }

    for (xE, xR), data in results.items():
        key = f"({xE},{xR})"
        ema["feasible_actions"][key] = bool(data.get("feasible"))

        if not data.get("feasible"):
            continue

        P_fixed = data.get("P_fixed", data.get("gen_base", 0.0))
        D_fixed = data.get("D_fixed", data.get("gen_per_q", 0.0))
        P_tok = data.get("P_per_token", 0.0)
        emb_per_q = data["emb_per_q"]
        ret_per_q = data["ret_per_q"]

        ema["gen_base_overhead_ema"][key] = P_fixed   # P (ms)
        ema["gen_per_query_ema"][key] = D_fixed        # D (ms/q)
        ema["gen_prefill_per_token_ema"][key] = P_tok  # C_tok (ms/token), may be 0
        ema["contention_ema"][key] = 1.0              # seeded, updated online
        ema["embedding_latency_ema"][str(xE)] = emb_per_q
        ema["retrieval_latency_ema"][str(xR)] = ret_per_q
        ema["overlap_factor_ema"][key] = 0.0
        ema["best_batch_size_by_action"][key] = data["best_batch_size"]
        ema["wall_time_measurements"][key] = data["wall_points"]

    return ema


def print_report(
    gpu: Dict,
    config: Dict,
    results: Dict[Tuple[int, int], Dict],
    ema: Dict,
    report_path: Path,
    params_path: Path,
):
    """Print and save the tuning report."""
    lines = []
    W = 72

    def rule():
        lines.append("=" * W)

    def header(text):
        lines.append("")
        lines.append(text)
        lines.append("-" * len(text))

    def row(*cols):
        lines.append("  " + " | ".join(str(c) for c in cols))

    rule()
    lines.append("  Auto-Tune Report — Async RAG Pipeline")
    rule()
    lines.append("")
    lines.append("DEVICE DETECTION")
    header("GPU")
    if gpu:
        row("Model", gpu["name"])
        row("VRAM", f"{gpu['total_mem_gb']} GB")
        row("Compute", gpu["compute_capability"])
    else:
        row("No GPU detected")
    lines.append("")
    lines.append("RECOMMENDED CONFIG")
    header("Generator model")
    row(config["model"])
    row("Reason", config["reason"])
    lines.append("")
    header("GPU utilization")
    row(f"gpu_memory_utilization = {config['gpu_util']}")
    lines.append("")

    rule()
    lines.append("")
    lines.append("CALIBRATION RESULTS")
    lines.append("")

    for (xE, xR), data in sorted(results.items()):
        action = f"({xE},{xR})"
        desc = {
            (0, 0): "CPU embed + CPU retrieval",
            (0, 1): "CPU embed + GPU retrieval",
            (1, 0): "GPU embed + CPU retrieval",
            (1, 1): "GPU embed + GPU retrieval",
        }.get((xE, xR), "")

        header(f"Action {action} — {desc}")
        if not data.get("feasible"):
            row("Status", f"INFEASIBLE on this device ({data.get('reason', 'OOM')[:60]})")
            lines.append("")
            continue

        row("Status", "OK")
        row("P_fixed (ms)", f"{data['P_fixed']:.0f}")
        row("D_fixed (ms/q)", f"{data['D_fixed']:.1f}")
        if data.get("P_per_token", 0) > 0:
            row("P_per_token (ms/tok)", f"{data['P_per_token']:.4f}")
        row("emb_per_q", f"{data['emb_per_q']:.2f} ms/q")
        row("ret_per_q", f"{data['ret_per_q']:.3f} ms/q")
        row("R²", f"{data['r2']:.6f}")
        lines.append("")
        row(f"{'bs':>5} | {'gen_total':>12} | {'gen/q':>8} | {'emb/q':>7} | {'ret/q':>7} | {'wall_ms':>8} | {'qps':>6}")
        row(f"{'-'*5}-+-{'-'*12}-+-{'-'*8}-+-{'-'*7}-+-{'-'*7}-+-{'-'*8}-+-{'-'*6}")
        for pt in data["points"]:
            row(
                f"{pt['batch_size']:>5}",
                f"{pt['gen_total_ms']:>12.0f}",
                f"{pt['gen_per_q_ms']:>8.1f}",
                f"{pt['emb_per_q_ms']:>7.2f}",
                f"{pt['ret_per_q_ms']:>7.3f}",
                f"{pt.get('wall_ms', 0):>8.0f}",
                f"{pt.get('qps', 0):>6.1f}",
            )

        lines.append("")
        row("Best bs:", f"{data['best_batch_size']} (score={data['best_score']:.1f} ms/q)")

        lines.append("")
        lines.append("  Predicted gen_q at different batch sizes (gen_q = P/B + D):")
        P = data["P_fixed"]
        D = data["D_fixed"]
        for bs_t in [1, 4, 16, 32, 64, 128, 256]:
            pred = P / bs_t + D
            score = pred
            marker = " ← best" if bs_t == data["best_batch_size"] else ""
            lines.append(f"    b={bs_t:>3}: gen_q={pred:>7.1f}ms/q{marker}")

        lines.append("")

    rule()
    lines.append("")
    lines.append("RECOMMENDATIONS")
    header("Best overall action")
    feasible = [(k, v) for k, v in results.items() if v.get("feasible")]
    if feasible:
        best = min(feasible, key=lambda kv: kv[1]["best_score"])
        best_action = f"({best[0][0]},{best[0][1]})"
        best_score = best[1]["best_score"]
        best_bs = best[1]["best_batch_size"]
        row("Action", best_action)
        row("Batch size", str(best_bs))
        row("Score", f"{best_score:.1f} ms/q")
    else:
        row("No feasible action found!")

    lines.append("")
    header("Next step: use the tuned parameters")
    lines.append(f"")
    lines.append(f"  python async_rag_pipeline.py \\")
    lines.append(f"    --xE {best[0][0]} --xR {best[0][1]} --b {best[1]['best_batch_size']} \\")
    lines.append(f"    --pipeline-mode async_v2 \\")
    lines.append(f"    --generator-model {config['model']} \\")
    lines.append(f"    --gpu-memory-utilization {config['gpu_util']} \\")
    lines.append(f"    --ema-params-path {params_path} \\")
    lines.append(f"    ...")
    lines.append(f"")
    lines.append(f"  Or with run_comparison.py:")
    lines.append(f"  python run_comparison.py --generator-model {config['model']} \\")
    lines.append(f"    --gpu-memory-utilization {config['gpu_util']} ...")

    lines.append("")
    rule()
    lines.append(f"  EMA params saved to:  {params_path}")
    lines.append(f"  Full report saved to:  {report_path}")
    rule()
    lines.append("")

    report_text = "\n".join(lines)
    print("\n" + report_text)

    with open(report_path, "w") as f:
        f.write(report_text)
    with open(params_path, "w") as f:
        json.dump(ema, f, indent=2)

    print(f"\nFiles written:")
    print(f"  {report_path}")
    print(f"  {params_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="One-shot calibration for the Async RAG pipeline. "
                    "Detects your GPU, selects an appropriate model, runs calibration "
                    "sweeps for each feasible (xE, xR) action, fits gen_q = P/B + D, "
                    "and saves ready-to-use EMA parameters."
    )
    parser.add_argument(
        "--workdir", type=str, default=None,
        help="Project root (default: auto-detect from this script's location)"
    )
    parser.add_argument(
        "--action", nargs=2, type=int, action="append",
        help="Specific (xE xR) to calibrate. Can be repeated. "
             "Example: --action 0 0"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Override generator model (default: auto-selected from GPU VRAM)"
    )
    parser.add_argument(
        "--gpu-util", type=float, default=None,
        help="Override GPU memory utilization (default: auto-selected)"
    )
    parser.add_argument(
        "--sample-queries", type=int, default=128,
        help="Queries per calibration run (default: 128)"
    )
    parser.add_argument(
        "--batch-sizes", nargs="+", type=int,
        default=[1, 4, 16, 64, 128],
        help="Batch sizes to sweep (default: 1 4 16 64 128)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run even if calibration files exist"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show plan without running"
    )
    parser.add_argument(
        "--skip-feasibility", action="store_true",
        help="Skip feasibility test (assume all actions work)"
    )
    args = parser.parse_args()

    # Resolve workdir
    workdir = Path(args.workdir).resolve() if args.workdir else SCRIPT_DIR
    script = workdir / "async_rag_pipeline.py"
    if not script.exists():
        print(f"ERROR: async_rag_pipeline.py not found at {script}")
        sys.exit(1)

    corpus_dir = detect_corpus()
    if not corpus_dir:
        print("ERROR: No BEIR corpus found.")
        print("  Run: python corpus_builder.py --dataset nfcorpus --output ./data/beir_nfcorpus")
        print("  Then: python generate_queries.py ...")
        sys.exit(1)

    print("=" * 72)
    print("  Async RAG — Auto-Tune")
    print("=" * 72)
    print()

    # Step 1: Device detection
    print("[1/5] Detecting GPU...")
    gpu = detect_gpu()
    if gpu:
        print(f"  GPU: {gpu['name']} ({gpu['total_mem_gb']} GB)")
    else:
        print("  No GPU detected — will proceed with CPU-only inference")

    config = recommend_config(gpu)
    if args.model:
        config["model"] = args.model
    if args.gpu_util:
        config["gpu_util"] = args.gpu_util

    print(f"  Model: {config['model']}")
    print(f"  GPU util: {config['gpu_util']}")
    print(f"  Reason: {config['reason']}")
    print(f"  Corpus: {corpus_dir}")

    # Step 2: Actions to calibrate
    actions_to_calibrate = [tuple(a) for a in args.action] if args.action else [(0, 0), (0, 1), (1, 0), (1, 1)]
    print(f"  Actions: {actions_to_calibrate}")

    if args.dry_run:
        print()
        print("Dry run — would run:")
        for (xE, xR) in actions_to_calibrate:
            for b in args.batch_sizes:
                print(f"  ({xE},{xR}) b={b}")
        return

    # Require enough queries so the largest batch size can be properly tested.
    # Each calibration run needs at least batch_size queries in a single dispatch.
    min_queries_needed = 2 * max(args.batch_sizes) if args.batch_sizes else 256
    if args.sample_queries < min_queries_needed:
        print(f"  NOTE: --sample-queries={args.sample_queries} is too small for bs={max(args.batch_sizes)}.")
        print(f"  Adjusting to {min_queries_needed} (2x max batch size).")
        args.sample_queries = min_queries_needed

    results: Dict[Tuple[int, int], Dict] = {}

    # Step 3: Feasibility test
    print()
    print(f"[2/5] Feasibility test ({len(actions_to_calibrate)} actions)...")
    feasible_actions = []
    for (xE, xR) in actions_to_calibrate:
        action_name = f"({xE},{xR})"
        if args.skip_feasibility:
            feasible_actions.append((xE, xR))
            print(f"  {action_name}: SKIPPED (--skip-feasibility) → assuming OK")
            continue

        print(f"  Testing {action_name}...", end=" ", flush=True)
        ok, reason = test_action_feasible(
            xE, xR, config["gpu_util"], config["model"], script, workdir,
            test_batch_size=max(args.batch_sizes),
        )
        if ok:
            feasible_actions.append((xE, xR))
            print("OK")
        else:
            results[(xE, xR)] = {"feasible": False, "reason": reason}
            print(f"INFEASIBLE ({reason[:60]})")

    if not feasible_actions:
        print("ERROR: No actions are feasible on this device!")
        sys.exit(1)

    print(f"  Feasible: {feasible_actions}")

    # Step 4: Calibration sweep
    print()
    print(f"[3/5] Calibration sweep ({len(feasible_actions)} actions × {len(args.batch_sizes)} batch sizes)")
    print()

    for (xE, xR) in feasible_actions:
        action_name = f"({xE},{xR})"
        print(f"  === {action_name} ===")

        all_points = []  # (batch_size, gen_total_ms)
        emb_per_q_samples = []
        ret_per_q_samples = []
        wall_points = []
        qps_samples = []

        for b in sorted(args.batch_sizes):
            data = run_calibration(
                xE, xR, b,
                config["gpu_util"],
                config["model"],
                args.sample_queries,
                script,
                workdir,
            )

            if data and data["points"]:
                pt = data["points"][0]
                # Use the REQUESTED batch_size (b), not the actual (which may be smaller
                # when sample_queries < b). Using b ensures the linear fit is meaningful.
                all_points.append((float(b), pt["gen_total_ms"]))
                emb_per_q_samples.append(pt["emb_per_q_ms"])
                ret_per_q_samples.append(pt["ret_per_q_ms"])
                wall_points.append([float(b), pt["gen_total_ms"]])
                qps_samples.append(data["wall_throughput_qps"])

                print(f"    b={b:>3}: gen={pt['gen_total_ms']:>7.0f}ms "
                      f"emb={pt['emb_per_q_ms']:.2f}ms/q "
                      f"ret={pt['ret_per_q_ms']:.3f}ms/q "
                      f"qps={data['wall_throughput_qps']:.1f}")
            else:
                print(f"    b={b:>3}: FAILED (result file missing or empty)")

            time.sleep(1)

        # Fit hyperbolic model: gen_q = P/B + D  →  gen_total = P + D*B
        calib = calibrate_action({"points": [
            {"batch_size": bs, "gen_total_ms": gt}
            for bs, gt in all_points
        ]})

        P_fixed = calib["P_fixed"]
        D_fixed = calib["D_fixed"]
        P_tok = calib["P_per_token"]
        r2 = calib["r2"]
        best_bs = calib["best_batch_size"]
        best_score = calib["best_score"]

        # Average emb/ret per query
        emb_per_q = sum(emb_per_q_samples) / len(emb_per_q_samples) if emb_per_q_samples else 2.0
        ret_per_q = sum(ret_per_q_samples) / len(ret_per_q_samples) if ret_per_q_samples else 0.15

        results[(xE, xR)] = {
            "feasible": True,
            "P_fixed": P_fixed,
            "D_fixed": D_fixed,
            "P_per_token": P_tok,
            "gen_base": P_fixed,     # backward compat
            "gen_per_q": D_fixed,    # backward compat
            "emb_per_q": emb_per_q,
            "ret_per_q": ret_per_q,
            "r2": r2,
            "points": [
                {"batch_size": bs, "gen_total_ms": gt, "gen_per_q_ms": gt / bs,
                 "emb_per_q_ms": emb_per_q, "ret_per_q_ms": ret_per_q,
                 "wall_ms": gt, "qps": qps_samples[i] if i < len(qps_samples) else 0}
                for i, (bs, gt) in enumerate(all_points)
            ],
            "best_batch_size": best_bs,
            "best_score": best_score,
            "wall_points": wall_points,
        }

        print(f"    → gen_q = {P_fixed:.0f}/B + {D_fixed:.1f}   R²={r2:.6f}")
        if P_tok > 0:
            print(f"    → P_per_token = {P_tok:.4f} ms/token (input-length sensitivity)")
        print(f"    → best bs={best_bs} (score={best_score:.1f} ms/q)")
        print()

    # Step 5: Build and save EMA params
    print("[4/5] Building EMA parameters...")
    ema = build_ema_params(results, config["gpu_util"], config["model"])
    print(f"  gen_base_overhead_ema: {ema['gen_base_overhead_ema']}")
    print(f"  gen_per_query_ema:     {ema['gen_per_query_ema']}")
    print(f"  embedding_latency_ema: {ema['embedding_latency_ema']}")
    print(f"  retrieval_latency_ema: {ema['retrieval_latency_ema']}")

    # Step 6: Save and report
    print()
    print("[5/5] Saving results...")
    model_tag = config["model"].replace("/", "_")
    ts = time.strftime("%Y%m%d_%H%M%S")
    params_path = OUTPUT_DIR / f"ema_params_{model_tag}_{ts}.json"
    report_path = OUTPUT_DIR / f"tuning_report_{model_tag}_{ts}.txt"

    print_report(gpu, config, results, ema, report_path, params_path)


if __name__ == "__main__":
    main()
