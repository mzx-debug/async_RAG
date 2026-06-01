#!/usr/bin/env python3
"""Analyze existing measurement data from Phase 1."""

import json
import math
import statistics
from pathlib import Path

ROOT = Path("/home/cloudteam/rag_mzx")
OUT = ROOT / "output" / "gpu_cost_measurement"

BATCH_SIZES = [1, 4, 8]

def load_costs(xE, xR, gpu_suffix):
    rows = []
    for B in BATCH_SIZES:
        path = OUT / f"cost_xE{xE}_xR{xR}_b{B}_{gpu_suffix}.json"
        if not path.exists():
            path = OUT / f"cost_xE{xE}_xR{xR}_b{B}.json"
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        fb = data.get("feedback_trace", [])
        fb_by_idx = {e["batch_index"]: e for e in fb}
        for b in data.get("per_batch", []):
            idx = b.get("batch_index", 0)
            e = fb_by_idx.get(idx, {})
            wall = (e.get("embedding_ms_per_query", 0)
                    + e.get("retrieval_ms_per_query", 0)
                    + e.get("generation_ms_per_query", 0))
            rows.append({
                "B": B,
                "emb_ms": e.get("embedding_ms_per_query", 0),
                "ret_ms": e.get("retrieval_ms_per_query", 0),
                "gen_ms": e.get("generation_ms_per_query", 0),
                "wall_ms": wall,
            })
    return rows


def fit_emb_rate(rows, L=5.0):
    pts = [(r["B"], r["emb_ms"]) for r in rows if r["emb_ms"] > 0 and r["B"] > 0]
    if len(pts) < 2:
        return None
    rates = [emb / (L * B) for B, emb in pts]
    return statistics.mean(rates)


def fit_ret_params(rows):
    pts = [(r["B"], r["ret_ms"]) for r in rows if r["ret_ms"] > 0 and r["B"] > 0]
    if len(pts) < 2:
        return None
    log_B = [math.log(B) for B, _ in pts]
    log_r = [math.log(r) for _, r in pts]
    n = len(log_B)
    sx, sy = sum(log_B), sum(log_r)
    sxy = sum(x*y for x, y in zip(log_B, log_r))
    sxx = sum(x*x for x in log_B)
    d = n * sxx - sx*sx
    if abs(d) < 1e-9:
        return None
    alpha = 1 + (n*sxy - sx*sy) / d
    r = math.exp((sy - (alpha-1)*sx) / n)
    return max(0.01, r), max(0.1, min(1.0, alpha))


def predict(xE, xR, B, p):
    gen_q = p["gen_per_token"] * p["avg_out"]
    gen_base_q = p["gen_base"] / B
    ret_cpu = p["r0"] * (B ** (p["a0"] - 1))
    ret_gpu = p["r1"] * (B ** (p["a1"] - 1))
    emb_cpu = (p["e0"] * p["L"]) if xE == 0 else 0.0
    emb_gpu = (p["e1"] * p["L"]) if xE == 1 else 0.0
    xfer = (p["K01"] * p["L"]) if (xE == 0 and xR == 1) else (
            (p["K10"] * p["L"]) if (xE == 1 and xR == 0) else 0.0)
    cpu_q = emb_cpu + ret_cpu + xfer
    gpu_q = gen_q + gen_base_q + emb_gpu + ret_gpu
    return max(cpu_q, gpu_q) + p["queue"]


def main():
    # Load Phase 1 data (gpu=0.8, suffix gpu8)
    rows_00 = load_costs(0, 0, "gpu8")
    rows_01 = load_costs(0, 1, "gpu8")

    print("=" * 70)
    print("  PHASE 1 RESULTS (gpu=0.8)")
    print("=" * 70)

    print(f"\n  (0,0) data: {len(rows_00)} rows")
    print(f"  (0,1) data: {len(rows_01)} rows")

    # Per-action per-B summary
    print(f"\n  {'Action':>6} | {'B':>4} | {'emb_ms':>10} | {'ret_ms':>10} | {'gen_ms':>10} | {'wall_ms':>10}")
    print(f"  {'-'*60}")

    for action, rows in [("0,0", rows_00), ("0,1", rows_01)]:
        if not rows:
            continue
        for B in BATCH_SIZES:
            rB = [r for r in rows if r["B"] == B]
            if not rB:
                continue
            emb = statistics.mean(r["emb_ms"] for r in rB)
            ret = statistics.mean(r["ret_ms"] for r in rB)
            gen = statistics.mean(r["gen_ms"] for r in rB)
            wall = statistics.mean(r["wall_ms"] for r in rB)
            print(f"  ({action})    | {B:>4} | {emb:>10.4f} | {ret:>10.4f} | {gen:>10.2f} | {wall:>10.2f}")

    # Fit r0, a0 from (0,0) — CPU retrieval
    r0_fit = fit_ret_params(rows_00)
    r1_fit = fit_ret_params(rows_01)  # GPU retrieval
    e0_fit = fit_emb_rate(rows_00)    # CPU emb rate (should be ~0.084)

    print(f"\n  {'='*70}")
    print(f"  FITTED PARAMETERS")
    print(f"  {'='*70}")

    if r0_fit:
        r0, a0 = r0_fit
        print(f"  r0 (CPU ret):  {r0:.4f}  alpha0={a0:.4f}  [default: 0.68 / 0.55]")
    else:
        r0, a0 = 0.68, 0.55
        print(f"  r0: FAILED TO FIT")

    if r1_fit:
        r1, a1 = r1_fit
        print(f"  r1 (GPU ret):  {r1:.4f}  alpha1={a1:.4f}  [default: 0.50 / 0.30]")
    else:
        r1, a1 = 0.50, 0.30
        print(f"  r1: FAILED TO FIT")

    if e0_fit:
        print(f"  e0 (CPU emb):  {e0_fit:.4f} ms/token  [default: 0.084]")
    else:
        e0_fit = 0.084

    # Model validation
    print(f"\n  {'='*70}")
    print(f"  MODEL PREDICTION vs ACTUAL (using fitted r0,a0,r1,a1)")
    print(f"  {'='*70}")

    params = {
        "e0": e0_fit, "e1": 0.016,   # e1 not measured yet
        "r0": r0, "a0": a0,
        "r1": r1, "a1": a1,
        "gen_per_token": 0.2135, "gen_base": 1109.0,
        "avg_out": 120.0, "L": 5.0, "queue": 0.23,
        "K01": 0.55, "K10": 0.16,
    }

    print(f"  Params: r0={r0:.4f}(α0={a0:.2f}), r1={r1:.4f}(α1={a1:.2f}), e0={e0_fit:.4f}")
    print(f"\n  {'Action':>6} | {'B':>4} | {'actual':>10} | {'pred':>10} | {'err%':>8}")
    print(f"  {'-'*44}")

    for action, rows in [("0,0", rows_00), ("0,1", rows_01)]:
        if not rows:
            continue
        xE, xR = map(int, action.split(","))
        for B in BATCH_SIZES:
            rB = [r for r in rows if r["B"] == B]
            if not rB:
                continue
            actual = statistics.mean(r["wall_ms"] for r in rB)
            pred = predict(xE, xR, B, params)
            err = (pred - actual) / actual * 100 if actual > 0 else 0
            flag = " ← LARGE ERR" if abs(err) > 20 else ""
            print(f"  ({action})    | {B:>4} | {actual:>10.2f} | {pred:>10.2f} | {err:>+7.1f}%{flag}")

    # Cross-validate: (0,1) retrieval with (0,0) model and vice versa
    print(f"\n  {'='*70}")
    print(f"  KEY INSIGHT: Retrieval cost comparison (0,0) vs (0,1)")
    print(f"  {'='*70}")

    for B in BATCH_SIZES:
        r00 = [r for r in rows_00 if r["B"] == B]
        r01 = [r for r in rows_01 if r["B"] == B]
        if not r00 or not r01:
            continue
        ret00 = statistics.mean(r["ret_ms"] for r in r00)
        ret01 = statistics.mean(r["ret_ms"] for r in r01)
        emb00 = statistics.mean(r["emb_ms"] for r in r00)
        emb01 = statistics.mean(r["emb_ms"] for r in r01)
        gen01 = statistics.mean(r["gen_ms"] for r in r01)
        gen00 = statistics.mean(r["gen_ms"] for r in r00)
        print(f"  B={B}: ret(CPU)={ret00:.4f}  ret(GPU)={ret01:.4f}  ratio={ret00/ret01:.2f}x  "
              f"gen(CPU)={gen00:.1f}  gen(GPU)={gen01:.1f}")


if __name__ == "__main__":
    main()
