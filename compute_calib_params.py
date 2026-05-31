#!/usr/bin/env python3
"""
Compute initial EMA calibration parameters from per-batch measurement files.

Uses per-batch MEDIAN (robust) estimation rather than log-log regression,
since the power-law fit is noisy with few data points and CPU retrieval
has a significant fixed-overhead component.

Fits the v4 cost model:
    wall_q = gen_per_token * avg_output_tokens
           + queue_penalty
           + gpu_contention    (xE=1 only, default 0 for portability)
           + er_overlap_penalty  (xE=0 only, learned from data)
           + xfer_q            (xE != xR only)

Usage:
    python compute_calib_params.py \
        --files output/calib_*.json \
        --emb-tokens-per-query 5.0 \
        --avg-output-tokens 120.0 \
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
    Fit retrieval model: ret_ms/q = r * B^(alpha-1)

    Uses MEDIAN of per-query values per B to be robust against outliers,
    then does log-log regression on the median values.
    """
    by_B = per_query_by_B(raw_points)
    B_vals = sorted(by_B.keys())
    medians = {B: statistics.median(v) for B, v in by_B.items()}

    print(f"  {xR_label} per-B medians: {medians}")

    if len(medians) < 2:
        return 1.0, 0.5, 0.0

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

    y_mean = s_lt / n
    ss_tot = sum((lt-y_mean)**2 for lt in log_t)
    ss_res = sum((lt-(log_r+alpha*lb))**2 for lb,lt in zip(log_B,log_t))
    r2 = 1.0 - ss_res/ss_tot if ss_tot > 1e-9 else 0.0

    alpha = max(0.3, min(1.0, alpha))
    return r, alpha, r2


def fit_emb_model(raw_points, xE_label, L):
    """Compute embedding rate: emb_ms/q = e * L."""
    if not raw_points:
        return 0.05
    empq = [emb_ms / B for B, emb_ms in raw_points]
    avg = statistics.median(empq)
    e = avg / L
    print(f"  {xE_label}: median_empq={avg:.3f}ms  →  e={e:.5f} ms/token (L~{L})")
    return e


def fit_gen_model(wall_raw, avg_output_tokens):
    """
    Fit v4 generation parameters:
        gen_per_token: median(gen_ms/q) / avg_output_tokens
        queue_penalty + er_overlap_penalty: wall_q - gen_q for xE=0 batches

    queue_penalty and er_overlap_penalty are NOT fully separable offline
    (both contribute to xE=0 residual). This function fits their SUM as
    queue_penalty; er_overlap_penalty defaults to 0 and refines online.

    gpu_contention is NOT fitted here (defaults to 0 for portability,
    converges online from xE=1 runtime data).
    """
    # Collect xE=0 measurements: wall_q - gen_q = queue + er_overlap + xfer
    residual_obs = []
    for (xE, xR), measurements in wall_raw.items():
        for B, ret_ms, gen_ms in measurements:
            if xE == 0:
                wall_q = (ret_ms + gen_ms) / B
                gen_q = gen_ms / B
                residual = wall_q - gen_q
                if 0 < residual < 50:
                    residual_obs.append(residual)

    if residual_obs:
        queue_penalty = statistics.median(residual_obs)
    else:
        queue_penalty = 2.5
    queue_penalty = max(0.0, min(50.0, queue_penalty))

    # er_overlap_penalty defaults to 0 offline (online EMA refines it)
    er_overlap_penalty = 0.0

    # Fit gen_per_token: median gen_ms/q across all batches / avg_output_tokens
    gen_per_q_list = []
    for measurements in wall_raw.values():
        for B, ret_ms, gen_ms in measurements:
            gen_per_q_list.append(gen_ms / B)

    if gen_per_q_list:
        median_gen_per_q = statistics.median(gen_per_q_list)
        gen_per_token = median_gen_per_q / avg_output_tokens
        gen_per_token = max(0.1, min(1.0, gen_per_token))
    else:
        gen_per_token = 0.378

    print(f"  gen_per_token={gen_per_token:.4f}ms/token  queue+er_overlap={queue_penalty:.2f}ms/q")
    return gen_per_token, queue_penalty, er_overlap_penalty


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--emb-tokens-per-query", type=float, default=5.0)
    ap.add_argument("--avg-output-tokens", type=float, default=120.0)
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

    # ── Generation + Queue ───────────────────────────────────────────────
    print(f"\n=== Generation (avg_out={args.avg_output_tokens}) ===")
    gen_per_token, queue_penalty, er_overlap_penalty = fit_gen_model(wall_raw, args.avg_output_tokens)

    # ── Write v4 calibration JSON ───────────────────────────────────────
    out = {
        "version": 4,
        "emb_rate_ema": {str(k): e_params.get(k, 0.05) for k in [0, 1]},
        "ret_r_ema":    {str(k): r_params[k]["r"] for k in r_params},
        "ret_alpha_ema":{str(k): r_params[k]["alpha"] for k in r_params},
        "gen_per_token_ema": gen_per_token,
        "avg_output_tokens_ema": args.avg_output_tokens,
        "queue_penalty_ema": queue_penalty,
        "gpu_contention_ema": 0.0,
        "er_overlap_penalty_ema": er_overlap_penalty,
        "transfer_K_ema": {"(0,1)": 0.55, "(1,0)": 0.16, "(1,1)": 0.0},
        "er_base_overhead_ema": 0.0,
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
