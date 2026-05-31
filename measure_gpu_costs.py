#!/usr/bin/env python3
"""
Targeted measurement of GPU embedding (xE=1) and GPU retrieval (xR=1) costs.

For each action and batch size, run async_plain (fixed action) and measure:
  - (1,0): GPU Emb + GPU Gen — gives e1 = emb_ms / (L * B)
  - (1,1): GPU Emb + GPU Ret + GPU Gen — gives r1, alpha1

Usage:
    python3 measure_gpu_costs.py          # all 4 actions, 3 batch sizes
    python3 measure_gpu_costs.py --dry-run
"""

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).parent.resolve()
PYTHON = "/home/cloudteam/Software/conda/envs/p702/bin/python"
OUTPUT_DIR = ROOT / "output" / "gpu_cost_measurement"
TIMEOUT_PER_RUN = 600

# Actions to measure
ACTIONS = [
    (0, 0),  # CPU Emb + CPU Ret + GPU Gen (baseline, already known)
    (1, 0),  # GPU Emb + GPU Gen (measures e1)
    (0, 1),  # CPU Emb + GPU Ret + GPU Gen (measures r1, alpha1 — partial data from sweep)
    (1, 1),  # GPU Emb + GPU Ret + GPU Gen (measures combined GPU cost)
]
BATCH_SIZES = [1, 4, 16]
QUERIES = 100
SCENARIO = "S1"  # nfcorpus + flat + short + gpu=0.8

CORPUS_CONFIG = {
    "beir_nfcorpus": {
        "corpus_path": str(ROOT / "data" / "beir_nfcorpus" / "corpus.jsonl"),
        "queries_path": str(ROOT / "data" / "beir_nfcorpus" / "queries.jsonl"),
        "index_path": str(ROOT / "indexes" / "beir_nfcorpus" / "faiss_flat.index"),
    },
}


def load_queries_short(path: str, n: int) -> List[Dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("token_length", 0) <= 8:
                records.append(rec)
    random.shuffle(records)
    return records[:n]


def build_query_file(n: int) -> str:
    cfg = CORPUS_CONFIG["beir_nfcorpus"]
    queries = load_queries_short(cfg["queries_path"], n)
    out = OUTPUT_DIR / "queries.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    return str(out)


def run_fixed_action(xE: int, xR: int, B: int, queries_path: str) -> Optional[Dict]:
    cfg = CORPUS_CONFIG["beir_nfcorpus"]
    gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    out_json = OUTPUT_DIR / f"cost_xE{xE}_xR{xR}_b{B}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        PYTHON, str(ROOT / "async_rag_pipeline.py"),
        "--pipeline-mode", "async_plain",
        "--xE", str(xE), "--xR", str(xR),
        "--b", str(B),
        "--fixed-action",
        "--index-path", cfg["index_path"],
        "--corpus-path", cfg["corpus_path"],
        "--queries-file", queries_path,
        "--generator-model", "Qwen/Qwen2.5-1.5B-Instruct",
        "--output-json", str(out_json),
        "--max-model-len", "8192",
        "--gpu-memory-utilization", "0.8",
        "--gpu-id", str(gpu_id),
        "--sample-queries", str(QUERIES),
    ]

    env = os.environ.copy()
    env["HF_ENDPOINT"] = "https://hf-mirror.com"
    env["PYTHONUNBUFFERED"] = "1"

    label = f"xE={xE}, xR={xR}, B={B}"
    print(f"  {label}...", end=" ", flush=True)
    t0 = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=TIMEOUT_PER_RUN, env=env,
        )
        elapsed = time.time() - t0
        if result.returncode == 0 and out_json.exists():
            with open(out_json) as f:
                data = json.load(f)
            print(f"OK ({elapsed:.0f}s)")
            return data
        else:
            print(f"FAIL (exit={result.returncode}, {elapsed:.0f}s)")
            for ln in result.stderr.strip().split("\n")[-3:]:
                if ln.strip():
                    print(f"    {ln}")
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT")
    except Exception as e:
        print(f"ERROR: {e}")
    return None


def extract_costs(data: Dict) -> Dict:
    """Extract per-batch cost components."""
    batches = data.get("per_batch", [])
    fb = data.get("feedback_trace", [])

    fb_by_idx = {f["batch_index"]: f for f in fb}

    rows = []
    for b in batches:
        idx = b.get("batch_index", 0)
        fb_entry = fb_by_idx.get(idx, {})
        rows.append({
            "B": b.get("batch_size", 0),
            "emb_ms": fb_entry.get("embedding_ms_per_query", 0),
            "ret_ms": fb_entry.get("retrieval_ms_per_query", 0),
            "gen_ms": fb_entry.get("generation_ms_per_query", 0),
            "wall_ms": (
                fb_entry.get("embedding_ms_per_query", 0) +
                fb_entry.get("retrieval_ms_per_query", 0) +
                fb_entry.get("generation_ms_per_query", 0)
            ),
            "tokens": b.get("generated_tokens", 0),
        })
    return rows


def fit_gpu_emb_rate(rows: List[Dict]) -> Optional[float]:
    """Fit e1 = emb_ms / (L * B) from (1,0) data."""
    pts = [(r["B"], r["emb_ms"]) for r in rows if r["emb_ms"] > 0]
    if len(pts) < 2:
        return None
    # emb_ms = e1 * L * B; emb_ms/B = e1 * L
    # L ~ 5 tokens (short queries)
    L = 5.0
    rates = [emb_ms / (L * B) for B, emb_ms in pts if B > 0]
    return statistics.mean(rates) if rates else None


def fit_gpu_ret_params(rows: List[Dict]) -> Optional[Tuple[float, float]]:
    """Fit r1, alpha1 from (1,1) or (0,1) data: ret_ms = r1 * B^(alpha1-1)."""
    pts = [(r["B"], r["ret_ms"]) for r in rows if r["ret_ms"] > 0]
    if len(pts) < 2:
        return None
    # log(ret_ms) = log(r1) + (alpha1-1) * log(B)
    import math
    log_B = [math.log(B) for B, _ in pts if B > 0]
    log_r = [math.log(r) for _, r in pts if r > 0]
    if len(log_B) < 2:
        return None
    n = len(log_B)
    sx = sum(log_B)
    sy = sum(log_r)
    sxy = sum(x * y for x, y in zip(log_B, log_r))
    sxx = sum(x * x for x in log_B)
    d = n * sxx - sx * sx
    if abs(d) < 1e-9:
        return None
    alpha = 1 + (n * sxy - sx * sy) / d
    r = math.exp((sy - (alpha - 1) * sx) / n)
    alpha = max(0.1, min(1.0, alpha))
    r = max(0.01, r)
    return r, alpha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"GPU COST MEASUREMENT")
    print(f"{'='*70}")
    print(f"  Actions:  {ACTIONS}")
    print(f"  B sizes:  {BATCH_SIZES}")
    print(f"  Queries:   {QUERIES}")
    print(f"  Scenario:  {SCENARIO}")
    print(f"  Output:    {OUTPUT_DIR}")
    print(f"{'='*70}\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    queries_path = build_query_file(QUERIES)

    if args.dry_run:
        for xE, xR in ACTIONS:
            for B in BATCH_SIZES:
                print(f"  Would run: xE={xE}, xR={xR}, B={B}")
        return

    results = {}

    for xE, xR in ACTIONS:
        print(f"\n{'─'*70}")
        print(f"  Action ({xE},{xR}):")
        print(f"{'─'*70}")
        results[(xE, xR)] = {}

        for B in BATCH_SIZES:
            data = run_fixed_action(xE, xR, B, queries_path)
            if data:
                rows = extract_costs(data)
                results[(xE, xR)][B] = rows
            else:
                results[(xE, xR)][B] = None

    # ── Print results ──────────────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print(f"MEASURED COSTS PER ACTION AND BATCH SIZE")
    print(f"{'='*70}")
    print(f"  {'Action':>8} | {'B':>4} | {'emb_ms/q':>10} | {'ret_ms/q':>10} | {'gen_ms/q':>10} | {'wall_ms/q':>10}")
    print(f"  {'-'*65}")

    for xE, xR in ACTIONS:
        for B in BATCH_SIZES:
            rows = results.get((xE, xR), {}).get(B)
            if rows:
                # average across batches
                emb = statistics.mean(r["emb_ms"] for r in rows)
                ret = statistics.mean(r["ret_ms"] for r in rows)
                gen = statistics.mean(r["gen_ms"] for r in rows)
                wall = statistics.mean(r["wall_ms"] for r in rows)
                print(f"  ({xE},{xR})     | {B:>4} | {emb:>10.4f} | {ret:>10.4f} | {gen:>10.2f} | {wall:>10.2f}")

    # ── Fit GPU embedding rate from (1,0) ────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"FITTED GPU COSTS")
    print(f"{'='*70}")

    # e1 from (1,0)
    all_10_rows = []
    for B in BATCH_SIZES:
        rows = results.get((1, 0), {}).get(B)
        if rows:
            all_10_rows.extend(rows)
    if all_10_rows:
        e1 = fit_gpu_emb_rate(all_10_rows)
        if e1:
            print(f"  e1 (GPU emb rate)       = {e1:.4f} ms/token  (from (1,0) data)")
            print(f"  [Previous default: 0.016 ms/token]")

    # r1, alpha1 from (1,1) or (0,1)
    for key, label in [((1, 1), "(1,1)"), ((0, 1), "(0,1)")]:
        all_rows = []
        for B in BATCH_SIZES:
            rows = results.get(key, {}).get(B)
            if rows:
                all_rows.extend(rows)
        if all_rows:
            fit = fit_gpu_ret_params(all_rows)
            if fit:
                r1, a1 = fit
                print(f"  r1 ({label} ret) = {r1:.4f}  alpha1 = {a1:.4f}")
                print(f"  [Previous default: r1=0.50, alpha1=0.30]")

    # ── Compare model predictions vs actual for each action ──────────────────
    print(f"\n{'='*70}")
    print(f"MODEL PREDICTIONS vs ACTUAL (using fitted GPU costs)")
    print(f"{'='*70}")

    # Use fitted e1 if available, otherwise default
    e1_fitted = e1 if all_10_rows else 0.016
    r1_fitted, a1_fitted = None, None
    for key, label in [((1, 1), "(1,1)"), ((0, 1), "(0,1)")]:
        all_rows = []
        for B in BATCH_SIZES:
            rows = results.get(key, {}).get(B)
            if rows:
                all_rows.extend(rows)
        if all_rows:
            fit = fit_gpu_ret_params(all_rows)
            if fit:
                r1_fitted, a1_fitted = fit
                break

    # Model params
    gen_per_token = 0.2135
    gen_base = 1109.0
    queue = 0.23
    avg_out = 120.0
    L = 5.0
    e0, e1_model = 0.084, e1_fitted
    r0, a0 = 0.68, 0.55
    r1_model, a1_model = r1_fitted if r1_fitted else 0.50, a1_fitted if a1_fitted else 0.30
    K01 = 0.55

    def predict(xE, xR, B):
        gen_q = gen_per_token * avg_out
        gen_base_q = gen_base / B
        ret_cpu = r0 * (B ** (a0 - 1))
        ret_gpu = r1_model * (B ** (a1_model - 1))
        emb_cpu = e0 * L
        emb_gpu = e1_model * L
        xfer = K01 * L if (xE == 0 and xR == 1) else 0.0
        cpu_q = emb_cpu + ret_cpu + xfer
        gpu_q = gen_q + gen_base_q + (emb_gpu if xE == 1 else 0.0) + (ret_gpu if xR == 1 else 0.0)
        return max(cpu_q, gpu_q) + queue

    print(f"  Using: e1={e1_model:.4f}, r1={r1_model:.4f}(α={a1_model:.2f})")
    print()
    print(f"  {'Action':>8} | {'B':>4} | {'actual_wall':>12} | {'pred_wall':>11} | {'err%':>8}")
    print(f"  {'-'*55}")

    for xE, xR in ACTIONS:
        for B in BATCH_SIZES:
            rows = results.get((xE, xR), {}).get(B)
            if rows:
                actual = statistics.mean(r["wall_ms"] for r in rows)
                pred = predict(xE, xR, B)
                err = (pred - actual) / actual * 100
                print(f"  ({xE},{xR})     | {B:>4} | {actual:>12.2f} | {pred:>11.2f} | {err:>+8.2f}%")

    # ── Save ────────────────────────────────────────────────────────────────
    out = OUTPUT_DIR / "measurement_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {out}")


if __name__ == "__main__":
    main()
