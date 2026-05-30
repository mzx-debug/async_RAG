#!/usr/bin/env python3
"""
Compute initial EMA calibration parameters from per-batch measurement files.

Uses per-batch MEDIAN (robust) estimation rather than log-log regression,
since the power-law fit is noisy with few data points and CPU retrieval
has a significant fixed-overhead component.

Usage:
    python compute_calib_params.py \
        --files output/test_plain_b1.json output/test_plain_b4.json ...
        --emb-tokens-per-query 5.0 \
        --output output/calibrated_params.json
"""
import argparse, json, math, statistics
from pathlib import Path
from collections import defaultdict


def extract(files):
    """Extract measurements from pipeline JSON output files."""
    ret_raw = {0: [], 1: []}   # {xR: [(B, ret_ms_total), ...]}
    emb_raw = {0: [], 1: []}   # {xE: [(B, emb_ms_total), ...]}
    wall_raw = {}              # {(xE,xR): [(B, ret_ms, gen_ms), ...]}
    for fp in files:
        with open(fp) as f:
            d = json.load(f)
        cfg = d.get("config", {})
        xE = cfg.get("xE", 0); xR = cfg.get("xR", 0)
        for b in d.get("per_batch", []):
            B = b["batch_size"]
            ret_raw[xR].append((B, b["retrieval_sec"]*1000))
            emb_raw[xE].append((B, b["embedding_sec"]*1000))
            key = (xE, xR)
            if key not in wall_raw:
                wall_raw[key] = []
            wall_raw[key].append((B, b["retrieval_sec"]*1000, b["generation_sec"]*1000))
    return ret_raw, emb_raw, wall_raw


def per_query_by_B(raw_points):
    """Group by B, return {B: [ret_ms/q, ...]}."""
    by_B = defaultdict(list)
    for B, ret_total in raw_points:
        by_B[B].append(ret_total / B)
    return {B: v for B, v in sorted(by_B.items())}


def fit_ret_model(raw_points, xR_label):
    """
    Fit retrieval model: ret_ms/q = r * B^(alpha-1) + base_overhead/B

    We use MEDIAN of per-query values per B to be robust against outliers,
    then do log-log regression on the median values.
    """
    by_B = per_query_by_B(raw_points)
    B_vals = sorted(by_B.keys())
    medians = {B: statistics.median(v) for B, v in by_B.items()}

    print(f"  {xR_label} per-B medians: {medians}")

    if len(medians) < 2:
        return 1.0, 0.5, 0.0

    # Log-log regression on medians
    log_B = [math.log(B) for B in B_vals]
    log_t = [math.log(t) for B, t in medians.items()]
    n = len(log_B)
    s_lb = sum(log_B); s_lt = sum(log_t)
    s_lblt = sum(lb*lt for lb,lt in zip(log_B,log_t))
    s_lb2 = sum(lb*lb for lb in log_B)
    denom = n*s_lb2 - s_lb*s_lb
    if abs(denom) < 1e-9:
        return 1.0, 0.5, 0.0
    alpha = (n*s_lblt - s_lb*s_lt) / denom
    log_r = (s_lt - alpha*s_lb) / n
    r = math.exp(log_r)

    # R²
    y_mean = s_lt / n
    ss_tot = sum((lt-y_mean)**2 for lt in log_t)
    ss_res = sum((lt-(log_r+alpha*lb))**2 for lb,lt in zip(log_B,log_t))
    r2 = 1.0 - ss_res/ss_tot if ss_tot > 1e-9 else 0.0

    # Clamp to physical range [0.3, 1.0]
    alpha = max(0.3, min(1.0, alpha))
    return r, alpha, r2


def fit_emb_model(raw_points, xE_label, L):
    """Compute embedding rate: emb_ms/q = e * L."""
    if not raw_points:
        return 0.05
    # Per-batch emb_ms/q
    empq = [emb_ms / B for B, emb_ms in raw_points]
    avg = statistics.median(empq)
    e = avg / L
    print(f"  {xE_label}: median_empq={avg:.3f}ms  →  e={e:.5f} ms/token (L~{L})")
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--emb-tokens-per-query", type=float, default=5.0)
    ap.add_argument("--output", default="output/calibrated_params.json")
    args = ap.parse_args()
    files = [Path(fp) for fp in args.files]

    ret_raw, emb_raw, wall_raw = extract(files)

    # ── Retrieval ────────────────────────────────────────────────────────
    print("\n=== Retrieval: ret_ms/q = r * B^(alpha-1) ===")
    r_params = {}
    for xR in [0, 1]:
        r, alpha, r2 = fit_ret_model(ret_raw[xR], f"xR={xR}")
        r_params[xR] = dict(r=r, alpha=alpha, r2=r2)
        print(f"  xR={xR}: r={r:.4f} alpha={alpha:.4f} R^2={r2:.4f}")
        # Show predictions vs actual medians
        by_B = per_query_by_B(ret_raw[xR])
        medians = {B: statistics.median(v) for B, v in by_B.items()}
        for B in sorted(medians):
            pred = r * (B ** (alpha - 1))
            err = (pred - medians[B]) / medians[B] * 100
            print(f"    B={B:3d}: pred={pred:.3f}ms/q actual={medians[B]:.3f}ms/q err={err:+.1f}%")

    # ── Embedding ────────────────────────────────────────────────────────
    print("\n=== Embedding: emb_ms/q = e * L ===")
    e_params = {}
    for xE in [0, 1]:
        e = fit_emb_model(emb_raw[xE], f"xE={xE}", args.emb_tokens_per_query)
        e_params[xE] = e

    # ── Write calibration JSON ───────────────────────────────────────────
    out = {
        "version": 3,
        "emb_rate_ema": {str(k): e_params.get(k, 0.05) for k in [0, 1]},
        "ret_r_ema":    {str(k): r_params[k]["r"] for k in r_params},
        "ret_alpha_ema":{str(k): r_params[k]["alpha"] for k in r_params},
        "gen_P0_ema": 2072.0,
        "gen_p_lin_ema": 0.0,
        "gen_g_ema": 57.1,
        "transfer_K_ema": {"(0,1)": 0.0, "(1,0)": 0.0, "(1,1)": 0.0},
        "contention_ema": {"(0,0)": 1.0, "(0,1)": 1.0, "(1,0)": 1.0, "(1,1)": 1.0},
        "er_base_overhead_ema": {},
        "ret_measurements": {str(k): [list(p) for p in sorted(set(ret_raw[int(k)]))]
                             for k in ["0","1"] if ret_raw[int(k)]},
        "wall_time_measurements": {f"({k[0]},{k[1]})": [list(m) for m in sorted(set(wall_raw[k]))]
                                  for k in wall_raw},
        "max_batch_size_ema": 32.0,
        "best_batch_size_by_action": {},
        "feasible_actions": {"(0,0)": True, "(0,1)": True, "(1,0)": True, "(1,1)": True},
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
