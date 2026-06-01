#!/usr/bin/env python3
"""Joint analysis of Phase 1 (gpu=0.8) and Phase 2 (gpu=0.5)."""

import json
import math
import statistics
from pathlib import Path

ROOT = Path("/home/cloudteam/rag_mzx")
OUT = ROOT / "output" / "gpu_cost_measurement"
BATCH_SIZES = [1, 4, 8]


def load_all():
    """Load all 4 actions from both phases."""
    data = {}
    # Phase 1: gpu=0.8 → suffix "gpu8"
    for xE in [0]:
        for xR in [0, 1]:
            for B in BATCH_SIZES:
                path = OUT / f"cost_xE{xE}_xR{xR}_b{B}_gpu8.json"
                if not path.exists():
                    path = OUT / f"cost_xE{xE}_xR{xR}_b{B}.json"
                if not path.exists():
                    continue
                with open(path) as f:
                    d = json.load(f)
                fb = {e["batch_index"]: e for e in d.get("feedback_trace", [])}
                for b in d.get("per_batch", []):
                    idx = b.get("batch_index", 0)
                    e = fb.get(idx, {})
                    wall = (e.get("embedding_ms_per_query", 0)
                            + e.get("retrieval_ms_per_query", 0)
                            + e.get("generation_ms_per_query", 0))
                    key = (xE, xR, B)
                    data.setdefault(key, []).append({
                        "B": B,
                        "emb_ms": e.get("embedding_ms_per_query", 0),
                        "ret_ms": e.get("retrieval_ms_per_query", 0),
                        "gen_ms": e.get("generation_ms_per_query", 0),
                        "wall_ms": wall,
                        "gpu": "0.8",
                    })
    # Phase 2: gpu=0.5 → suffix "gpu5"
    for xE in [1]:
        for xR in [0, 1]:
            for B in BATCH_SIZES:
                path = OUT / f"cost_xE{xE}_xR{xR}_b{B}_gpu5.json"
                if not path.exists():
                    continue
                with open(path) as f:
                    d = json.load(f)
                fb = {e["batch_index"]: e for e in d.get("feedback_trace", [])}
                for b in d.get("per_batch", []):
                    idx = b.get("batch_index", 0)
                    e = fb.get(idx, {})
                    wall = (e.get("embedding_ms_per_query", 0)
                            + e.get("retrieval_ms_per_query", 0)
                            + e.get("generation_ms_per_query", 0))
                    key = (xE, xR, B)
                    data.setdefault(key, []).append({
                        "B": B,
                        "emb_ms": e.get("embedding_ms_per_query", 0),
                        "ret_ms": e.get("retrieval_ms_per_query", 0),
                        "gen_ms": e.get("generation_ms_per_query", 0),
                        "wall_ms": wall,
                        "gpu": "0.5",
                    })
    return data


def avg(rows, field):
    vals = [r[field] for r in rows if r[field] > 0]
    return statistics.mean(vals) if vals else 0.0


def fit_emb_rate(rows):
    pts = [(r["B"], r["emb_ms"]) for r in rows if r["emb_ms"] > 0 and r["B"] > 0]
    if len(pts) < 2:
        return None
    rates = [emb / (5.0 * B) for B, emb in pts]
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
    d = n*sxx - sx*sx
    if abs(d) < 1e-9:
        return None
    alpha = 1 + (n*sxy - sx*sy) / d
    r = math.exp((sy - (alpha-1)*sx) / n)
    return max(0.01, r), max(0.05, min(1.0, alpha))


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
    data = load_all()
    total = sum(len(v) for v in data.values())
    print(f"Loaded {total} data points from {len(data)} (action, B) keys")

    # ── All data ──────────────────────────────────────────────────────────
    print("\n" + "="*75)
    print("  ALL MEASURED DATA")
    print("="*75)
    print(f"  {'Action':>8} | {'B':>4} | {'gpu':>5} | {'emb_ms':>10} | {'ret_ms':>10} | {'gen_ms':>10} | {'wall_ms':>10}")
    print(f"  {'-'*75}")
    for xE in [0, 1]:
        for xR in [0, 1]:
            for B in BATCH_SIZES:
                rows = data.get((xE, xR, B), [])
                if not rows:
                    continue
                print(f"  ({xE},{xR})     | {B:>4} | {rows[0]['gpu']:>5} | "
                      f"{avg(rows,'emb_ms'):>10.4f} | {avg(rows,'ret_ms'):>10.4f} | "
                      f"{avg(rows,'gen_ms'):>10.2f} | {avg(rows,'wall_ms'):>10.2f}")

    # ── Fit ───────────────────────────────────────────────────────────────
    rows_00 = data.get((0, 0, BATCH_SIZES[0]), []) + data.get((0, 0, BATCH_SIZES[1]), []) + data.get((0, 0, BATCH_SIZES[2]), [])
    rows_01 = data.get((0, 1, BATCH_SIZES[0]), []) + data.get((0, 1, BATCH_SIZES[1]), []) + data.get((0, 1, BATCH_SIZES[2]), [])
    rows_10 = data.get((1, 0, BATCH_SIZES[0]), []) + data.get((1, 0, BATCH_SIZES[1]), []) + data.get((1, 0, BATCH_SIZES[2]), [])
    rows_11 = data.get((1, 1, BATCH_SIZES[0]), []) + data.get((1, 1, BATCH_SIZES[1]), []) + data.get((1, 1, BATCH_SIZES[2]), [])

    e0_fit = fit_emb_rate(rows_00)
    e1_fit = fit_emb_rate(rows_10)
    r0_fit = fit_ret_params(rows_00)
    r1_fit = fit_ret_params(rows_01)

    print("\n" + "="*75)
    print("  FITTED PARAMETERS")
    print("="*75)
    print(f"  e0 (CPU emb):   {e0_fit:.4f} ms/token  [OLD: 0.084]")
    print(f"  e1 (GPU emb):   {e1_fit:.4f} ms/token  [OLD: 0.016]")
    if r0_fit:
        print(f"  r0 (CPU ret):  {r0_fit[0]:.4f}  alpha0={r0_fit[1]:.4f}  [OLD: 0.68/0.55]")
    if r1_fit:
        print(f"  r1 (GPU ret):  {r1_fit[0]:.4f}  alpha1={r1_fit[1]:.4f}  [OLD: 0.50/0.30]")

    r0, a0 = r0_fit if r0_fit else (0.68, 0.55)
    r1, a1 = r1_fit if r1_fit else (0.50, 0.30)
    e0 = e0_fit or 0.084
    e1 = e1_fit or 0.016

    # ── Model validation ──────────────────────────────────────────────────
    print("\n" + "="*75)
    print("  MODEL VALIDATION: OLD vs NEW params")
    print("="*75)

    OLD = {"e0": 0.084, "e1": 0.016, "r0": 0.68, "a0": 0.55, "r1": 0.50, "a1": 0.30,
           "gen_per_token": 0.2135, "gen_base": 1109.0, "avg_out": 120.0, "L": 5.0,
           "queue": 0.23, "K01": 0.55, "K10": 0.16}
    NEW = dict(OLD, e0=e0, e1=e1, r0=r0, a0=a0, r1=r1, a1=a1)

    print(f"  {'Action':>8} | {'B':>4} | {'actual':>10} | {'old_pred':>10} | {'old_err%':>8} | {'new_pred':>10} | {'new_err%':>8}")
    print(f"  {'-'*85}")

    old_max_err = 0.0
    new_max_err = 0.0
    for xE in [0, 1]:
        for xR in [0, 1]:
            for B in BATCH_SIZES:
                rows = data.get((xE, xR, B), [])
                if not rows:
                    continue
                actual = avg(rows, "wall_ms")
                old_p = predict(xE, xR, B, OLD)
                new_p = predict(xE, xR, B, NEW)
                old_err = (old_p - actual) / actual * 100 if actual > 0 else 0
                new_err = (new_p - actual) / actual * 100 if actual > 0 else 0
                old_max_err = max(old_max_err, abs(old_err))
                new_max_err = max(new_max_err, abs(new_err))
                flag = " ← FIX" if abs(new_err) > 20 else ""
                print(f"  ({xE},{xR})     | {B:>4} | {actual:>10.2f} | {old_p:>10.2f} | {old_err:>+7.1f}% | {new_p:>10.2f} | {new_err:>+7.1f}%{flag}")

    print(f"\n  OLD model max error: {old_max_err:.1f}%")
    print(f"  NEW model max error: {new_max_err:.1f}%")

    # ── Key insights ─────────────────────────────────────────────────────
    print("\n" + "="*75)
    print("  KEY INSIGHTS")
    print("="*75)

    print("\n  CPU vs GPU Retrieval (B=1,4,8):")
    for B in BATCH_SIZES:
        r00 = data.get((0, 0, B), [])
        r01 = data.get((0, 1, B), [])
        if not r00 or not r01:
            continue
        ret_cpu = avg(r00, "ret_ms")
        ret_gpu = avg(r01, "ret_ms")
        winner = "GPU" if ret_gpu < ret_cpu else "CPU"
        print(f"    B={B}: CPU={ret_cpu:.4f}ms  GPU={ret_gpu:.4f}ms  winner={winner} ({max(ret_cpu,ret_gpu)/min(ret_cpu,ret_gpu):.2f}x)")

    print("\n  CPU vs GPU Embedding:")
    for B in BATCH_SIZES:
        r00 = data.get((0, 0, B), [])
        r10 = data.get((1, 0, B), [])
        if not r00 or not r10:
            continue
        emb_cpu = avg(r00, "emb_ms")
        emb_gpu = avg(r10, "emb_ms")
        winner = "GPU" if emb_gpu < emb_cpu else "CPU"
        print(f"    B={B}: CPU={emb_cpu:.4f}ms  GPU={emb_gpu:.4f}ms  winner={winner} ({max(emb_cpu,emb_gpu)/min(emb_cpu,emb_gpu):.2f}x)")

    print("\n  (0,0) vs (0,1) wall time:")
    for B in BATCH_SIZES:
        r00 = data.get((0, 0, B), [])
        r01 = data.get((0, 1, B), [])
        if not r00 or not r01:
            continue
        w00 = avg(r00, "wall_ms")
        w01 = avg(r01, "wall_ms")
        winner = "(0,1)" if w01 < w00 else "(0,0)"
        print(f"    B={B}: (0,0)={w00:.2f}ms  (0,1)={w01:.2f}ms  winner={winner} ({abs(w00-w01):.2f}ms)")

    print("\n" + "="*75)
    print("  RECOMMENDED CODE DEFAULTS")
    print("="*75)
    print(f"  e0:  0.084  →  {e0:.4f}")
    print(f"  e1:  0.016  →  {e1:.4f}")
    print(f"  r0:  0.68   →  {r0:.4f}  (alpha0: 0.55 → {a0:.4f})")
    print(f"  r1:  0.50   →  {r1:.4f}  (alpha1: 0.30 → {a1:.4f})")

    result = {
        "e0": round(e0, 6),
        "e1": round(e1, 6),
        "r0": round(r0, 6),
        "a0": round(a0, 4),
        "r1": round(r1, 6),
        "a1": round(a1, 4),
        "gen_per_token": 0.2135,
        "gen_base": 1109.0,
        "avg_out": 120.0,
        "L": 5.0,
        "queue": 0.23,
        "K01": 0.55,
        "K10": 0.16,
        "old_max_err": round(old_max_err, 1),
        "new_max_err": round(new_max_err, 1),
    }
    out = OUT / "fitted_params.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()
