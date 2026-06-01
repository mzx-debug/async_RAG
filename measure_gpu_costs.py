#!/usr/bin/env python3
"""
Targeted measurement of all 4 action costs.

Phase 1: (0,0) and (0,1) — gpu=0.8, max_model_len=8192
Phase 2: (1,0) and (1,1) — gpu=0.5, max_model_len=8192 (reduce for GPU Emb safety)

Usage:
    python3 measure_gpu_costs.py --phase 1    # (0,0), (0,1)
    python3 measure_gpu_costs.py --phase 2    # (1,0), (1,1)
    python3 measure_gpu_costs.py              # both phases
"""

import argparse
import json
import math
import os
import random
import statistics
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).parent.resolve()
PYTHON = "/home/cloudteam/Software/conda/envs/p702/bin/python"
OUTPUT_DIR = ROOT / "output" / "gpu_cost_measurement"
TIMEOUT_PER_RUN = 600

QUERIES = 100

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
    random.seed(42)
    random.shuffle(records)
    return records[:n]


def build_query_file() -> str:
    cfg = CORPUS_CONFIG["beir_nfcorpus"]
    queries = load_queries_short(cfg["queries_path"], QUERIES * 2)
    out = OUTPUT_DIR / "queries.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    return str(out)


def run_fixed_action(
    xE: int, xR: int, B: int, queries_path: str,
    gpu_util: float, max_model_len: int,
) -> Optional[Dict]:
    cfg = CORPUS_CONFIG["beir_nfcorpus"]
    gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    out_json = OUTPUT_DIR / f"cost_xE{xE}_xR{xR}_b{B}_gpu{int(gpu_util*10)}.json"

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
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_util),
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
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_PER_RUN, env=env)
        elapsed = time.time() - t0
        if result.returncode == 0 and out_json.exists():
            with open(out_json) as f:
                data = json.load(f)
            print(f"OK ({elapsed:.0f}s)")
            return data
        else:
            print(f"FAIL ({elapsed:.0f}s)")
            for ln in result.stderr.strip().split("\n"):
                if any(k in ln.lower() for k in ["error", "fail", "cuda", "memory", "oom"]):
                    print(f"    {ln[:120]}")
            return None
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT")
        return None


def extract_costs(data: Dict) -> List[Dict]:
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
            "wall_ms": fb_entry.get("embedding_ms_per_query", 0)
                      + fb_entry.get("retrieval_ms_per_query", 0)
                      + fb_entry.get("generation_ms_per_query", 0),
        })
    return rows


def fit_emb_rate(rows: List[Dict], L: float = 5.0) -> Optional[float]:
    pts = [(r["B"], r["emb_ms"]) for r in rows if r["emb_ms"] > 0]
    if len(pts) < 2:
        return None
    rates = [emb_ms / (L * B) for B, emb_ms in pts if B > 0]
    return statistics.mean(rates) if rates else None


def fit_ret_params(rows: List[Dict]) -> Optional[Tuple[float, float]]:
    pts = [(r["B"], r["ret_ms"]) for r in rows if r["ret_ms"] > 0]
    pts = [(B, r) for B, r in pts if B > 0 and r > 0]
    if len(pts) < 2:
        return None
    log_B = [math.log(B) for B, _ in pts]
    log_r = [math.log(r) for _, r in pts]
    n = len(log_B)
    sx, sy = sum(log_B), sum(log_r)
    sxy = sum(x * y for x, y in zip(log_B, log_r))
    sxx = sum(x * x for x in log_B)
    d = n * sxx - sx * sx
    if abs(d) < 1e-9:
        return None
    alpha = 1 + (n * sxy - sx * sy) / d
    r = math.exp((sy - (alpha - 1) * sx) / n)
    return max(0.01, r), max(0.1, min(1.0, alpha))


def predict(xE, xR, B, params: Dict) -> float:
    p = params
    gen_q = p["gen_per_token"] * p["avg_out"]
    gen_base_q = p["gen_base"] / B
    ret_cpu = p["r0"] * (B ** (p["a0"] - 1))
    ret_gpu = p["r1"] * (B ** (p["a1"] - 1))
    emb_cpu = (p["e0"] * p["L"]) if xE == 0 else 0.0
    emb_gpu = (p["e1"] * p["L"]) if xE == 1 else 0.0
    xfer = p["K01"] * p["L"] if (xE == 0 and xR == 1) else (
        p["K10"] * p["L"] if (xE == 1 and xR == 0) else 0.0)
    cpu_q = emb_cpu + ret_cpu + xfer
    gpu_q = gen_q + gen_base_q + emb_gpu + ret_gpu
    return max(cpu_q, gpu_q) + p["queue"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, choices=[1, 2], default=None,
                        help="Run only phase 1 (0,0),(0,1) or phase 2 (1,0),(1,1)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    queries_path = build_query_file()

    # Phase configs
    phases = []
    if args.phase is None or args.phase == 1:
        phases.append({
            "name": "Phase 1: CPU Emb actions (0,0) and (0,1)",
            "actions": [(0, 0), (0, 1)],
            "gpu_util": 0.8,
            "max_model_len": 8192,
        })
    if args.phase is None or args.phase == 2:
        phases.append({
            "name": "Phase 2: GPU Emb actions (1,0) and (1,1)",
            "actions": [(1, 0), (1, 1)],
            "gpu_util": 0.5,  # lower to give headroom for GPU Emb
            "max_model_len": 8192,
        })

    BATCH_SIZES = [1, 4, 8]

    all_results = {}

    for phase in phases:
        print(f"\n{'='*70}")
        print(f"  {phase['name']}")
        print(f"  gpu_util={phase['gpu_util']}, max_model_len={phase['max_model_len']}")
        print(f"{'='*70}")

        for xE, xR in phase["actions"]:
            print(f"\n  Action ({xE},{xR}):")
            all_results[(xE, xR)] = {}
            for B in BATCH_SIZES:
                data = run_fixed_action(
                    xE, xR, B, queries_path,
                    gpu_util=phase["gpu_util"],
                    max_model_len=phase["max_model_len"],
                )
                rows = extract_costs(data) if data else None
                all_results[(xE, xR)][B] = rows

    # ── Print measured costs ─────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print(f"  MEASURED COSTS")
    print(f"{'='*70}")
    print(f"  {'Action':>8} | {'B':>4} | {'emb_ms':>10} | {'ret_ms':>10} | {'gen_ms':>10} | {'wall_ms':>10}")
    print(f"  {'-'*62}")

    for xE, xR in sorted(all_results.keys()):
        for B in BATCH_SIZES:
            rows = all_results[(xE, xR)].get(B)
            if rows:
                emb = statistics.mean(r["emb_ms"] for r in rows)
                ret = statistics.mean(r["ret_ms"] for r in rows)
                gen = statistics.mean(r["gen_ms"] for r in rows)
                wall = statistics.mean(r["wall_ms"] for r in rows)
                print(f"  ({xE},{xR})     | {B:>4} | {emb:>10.4f} | {ret:>10.4f} | {gen:>10.2f} | {wall:>10.2f}")

    # ── Fit parameters ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  FITTED PARAMETERS")
    print(f"{'='*70}")

    fitted = {}

    # e1 from (1,0)
    rows_10 = [r for B in BATCH_SIZES for r in (all_results.get((1,0), {}).get(B) or [])]
    e1 = fit_emb_rate(rows_10) if rows_10 else None
    if e1:
        print(f"  e1 (GPU emb rate): {e1:.4f} ms/token  [default: 0.016]")
        fitted["e1"] = e1
    else:
        print(f"  e1: NOT MEASURED (need (1,0) data)")
        fitted["e1"] = 0.016

    # r0, a0 from (0,0)
    rows_00 = [r for B in BATCH_SIZES for r in (all_results.get((0,0), {}).get(B) or [])]
    r0_fit = fit_ret_params(rows_00) if rows_00 else None
    if r0_fit:
        r0, a0 = r0_fit
        print(f"  r0 (CPU ret):      {r0:.4f}  alpha0={a0:.4f}  [default: 0.68/0.55]")
        fitted["r0"], fitted["a0"] = r0, a0
    else:
        fitted["r0"], fitted["a0"] = 0.68, 0.55

    # r1, a1 from (0,1)
    rows_01 = [r for B in BATCH_SIZES for r in (all_results.get((0,1), {}).get(B) or [])]
    r1_fit = fit_ret_params(rows_01) if rows_01 else None
    if r1_fit:
        r1, a1 = r1_fit
        print(f"  r1 (GPU ret):      {r1:.4f}  alpha1={a1:.4f}  [default: 0.50/0.30]")
        fitted["r1"], fitted["a1"] = r1, a1
    else:
        fitted["r1"], fitted["a1"] = 0.50, 0.30

    # ── Model validation ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  MODEL PREDICTION vs ACTUAL")
    print(f"{'='*70}")

    params = {
        "e0": 0.084, "e1": fitted.get("e1", 0.016),
        "r0": fitted.get("r0", 0.68), "a0": fitted.get("a0", 0.55),
        "r1": fitted.get("r1", 0.50), "a1": fitted.get("a1", 0.30),
        "gen_per_token": 0.2135, "gen_base": 1109.0,
        "avg_out": 120.0, "L": 5.0, "queue": 0.23,
        "K01": 0.55, "K10": 0.16,
    }
    print(f"  Params: e1={params['e1']:.4f}, r0={params['r0']:.4f}(α0={params['a0']:.2f}), "
          f"r1={params['r1']:.4f}(α1={params['a1']:.2f})")
    print()
    print(f"  {'Action':>8} | {'B':>4} | {'actual':>10} | {'pred':>10} | {'err%':>8}")
    print(f"  {'-'*50}")

    has_error = False
    for xE, xR in sorted(all_results.keys()):
        for B in BATCH_SIZES:
            rows = all_results[(xE, xR)].get(B)
            if not rows:
                continue
            actual = statistics.mean(r["wall_ms"] for r in rows)
            pred = predict(xE, xR, B, params)
            err = (pred - actual) / actual * 100 if actual > 0 else 0
            flag = " ← FIX ME" if abs(err) > 10 else ""
            print(f"  ({xE},{xR})     | {B:>4} | {actual:>10.2f} | {pred:>10.2f} | {err:>+7.1f}%{flag}")
            if abs(err) > 10:
                has_error = True

    if not has_error:
        print(f"\n  All predictions within 10% — model is well calibrated.")

    # ── Save ────────────────────────────────────────────────────────────────
    out = OUTPUT_DIR / "measurement_results.json"
    serializable = {}
    for k, v in all_results.items():
        serializable[f"{k[0]}_{k[1]}"] = {str(bk): bv for bk, bv in v.items()}
    with open(out, "w") as f:
        json.dump({"results": serializable, "fitted": fitted, "params": params}, f, indent=2)
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()
