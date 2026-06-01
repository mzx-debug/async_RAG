#!/usr/bin/env python3
"""
Validate the cost model by running async_v2 under varied scenarios,
then comparing predicted vs actual per-query latency per action.

Output structure from async_rag_pipeline:
  - dispatch_trace: chosen action, predicted cost per action, predicted dispatch cost
  - per_batch: actual wall time breakdown per batch
  - feedback_trace: actual ms/q per batch

Usage:
    python3 validate_model.py --scenarios S1 S5 S9
    python3 validate_model.py --all          # all 12
    python3 validate_model.py --reuse        # skip runs that already have results
    python3 validate_model.py --dry-run
"""

import argparse
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

OUTPUT_DIR = ROOT / "output" / "model_validation"
PYTHON = "/home/cloudteam/Software/conda/envs/p702/bin/python"
TIMEOUT_PER_RUN = 600  # seconds


# ── Sweep-fitted model params ──────────────────────────────────────────────────
GEN_PER_TOKEN = 0.2135   # ms/token
GEN_BASE = 1109.0         # ms
QUEUE_PENALTY = 0.23      # ms/q
AVG_OUTPUT = 120.0        # tokens
L_DEFAULT = 5.0            # default query length
# Linear model: emb_per_q = e_base*L + overhead/B,  ret_per_q = r_base + overhead/B
E0, E1 = 0.36, 0.00          # ms/token, CPU and GPU embedding per-token rate
OH_E0, OH_E1 = 2.71, 4.87    # ms, CPU and GPU embedding fixed overhead
R0, OH_R0 = 0.20, 1.36       # CPU retrieval: per-query marginal + overhead/B
R1, OH_R1 = 0.00, 1.50       # GPU retrieval: per-query marginal + overhead/B
K01 = 0.55                    # ms/token, CPU->GPU transfer


def predict_wall_q(xE: int, xR: int, B: int, L_q: float = L_DEFAULT) -> float:
    B = max(1, B)
    L_q = L_q if L_q > 0 else L_DEFAULT
    gen_q = GEN_PER_TOKEN * AVG_OUTPUT
    gen_base_q = GEN_BASE / B
    # New linear models: emb = e*L + overhead/B,  ret = r + overhead/B
    emb_cpu = E0 * L_q + OH_E0 / B if xE == 0 else 0.0
    emb_gpu = E1 * L_q + OH_E1 / B if xE == 1 else 0.0
    ret_cpu = R0 + OH_R0 / B if xR == 0 else 0.0
    ret_gpu = R1 + OH_R1 / B if xR == 1 else 0.0
    xfer = K01 * L_q if (xE == 0 and xR == 1) else 0.0
    cpu_q = emb_cpu + ret_cpu + xfer
    gpu_q = gen_q + gen_base_q + emb_gpu + ret_gpu
    return max(cpu_q, gpu_q) + QUEUE_PENALTY


# ── Scenario definitions ───────────────────────────────────────────────────────

@dataclass
class Scenario:
    id: str
    corpus: str
    index: str
    query_mode: str   # "short" | "long" | "mixed"
    gpu_util: float
    desc: str

SCENARIOS = [
    Scenario("S1",  "beir_nfcorpus", "flat", "short", 0.8, "nfcorpus + flat + short + gpu=0.8"),
    Scenario("S2",  "beir_nfcorpus", "hnsw", "short", 0.8, "nfcorpus + hnsw + short + gpu=0.8"),
    Scenario("S3",  "beir_fiqa",     "flat", "short", 0.8, "fiqa + flat + short + gpu=0.8"),
    Scenario("S4",  "beir_fiqa",     "hnsw", "short", 0.8, "fiqa + hnsw + short + gpu=0.8"),
    Scenario("S5",  "beir_fiqa",     "flat", "long",  0.8, "fiqa + flat + long  + gpu=0.8"),
    Scenario("S6",  "beir_fiqa",     "hnsw", "long",  0.8, "fiqa + hnsw + long  + gpu=0.8"),
    Scenario("S7",  "beir_nfcorpus", "flat", "mixed", 0.8, "nfcorpus + flat + mixed + gpu=0.8"),
    Scenario("S8",  "beir_nfcorpus", "hnsw", "mixed", 0.8, "nfcorpus + hnsw + mixed + gpu=0.8"),
    Scenario("S9",  "beir_nfcorpus", "flat", "short", 0.3, "nfcorpus + flat + short + gpu=0.3"),
    Scenario("S10", "beir_nfcorpus", "hnsw", "short", 0.3, "nfcorpus + hnsw + short + gpu=0.3"),
    Scenario("S11", "beir_fiqa",     "flat", "long",  0.3, "fiqa + flat + long  + gpu=0.3"),
    Scenario("S12", "beir_fiqa",     "hnsw", "long",  0.3, "fiqa + hnsw + long  + gpu=0.3"),
]

CORPUS_CONFIG = {
    "beir_nfcorpus": {
        "corpus_path": str(ROOT / "data" / "beir_nfcorpus" / "corpus.jsonl"),
        "queries_path": str(ROOT / "data" / "beir_nfcorpus" / "queries.jsonl"),
        "index_path": str(ROOT / "indexes" / "beir_nfcorpus" / "faiss_{index}.index"),
    },
    "beir_fiqa": {
        "corpus_path": str(ROOT / "data" / "beir_fiqa" / "corpus.jsonl"),
        "queries_path": str(ROOT / "data" / "beir_fiqa" / "queries_with_length.jsonl"),
        "index_path": str(ROOT / "indexes" / "beir_fiqa" / "faiss_{index}.index"),
    },
}


# ── Data loading ───────────────────────────────────────────────────────────────

def load_queries_filtered(path: str, mode: str) -> List[Dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if mode == "short":
                if rec.get("token_length", 0) <= 8:
                    records.append(rec)
            elif mode == "long":
                if rec.get("token_length", 0) >= 12:
                    records.append(rec)
            else:
                records.append(rec)
    random.shuffle(records)
    return records


def build_query_file(scenario: Scenario, n: int) -> tuple:
    cfg = CORPUS_CONFIG[scenario.corpus]
    queries = load_queries_filtered(cfg["queries_path"], scenario.query_mode)
    if len(queries) < n:
        queries = (queries * math.ceil(n / max(1, len(queries))))[:n]
        random.shuffle(queries)
    else:
        queries = queries[:n]

    out_path = OUTPUT_DIR / f"queries_{scenario.id}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    return str(out_path), queries


def run_pipeline(scenario: Scenario, n_queries: int) -> Optional[Dict]:
    cfg = CORPUS_CONFIG[scenario.corpus]
    index_path = cfg["index_path"].format(index=scenario.index)
    queries_path, _ = build_query_file(scenario, n_queries)
    gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])

    out_json = OUTPUT_DIR / f"result_{scenario.id}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        PYTHON, str(ROOT / "async_rag_pipeline.py"),
        "--pipeline-mode", "async_v2",
        "--xE", "0", "--xR", "0",
        "--b", "32",
        "--index-path", index_path,
        "--corpus-path", cfg["corpus_path"],
        "--queries-file", queries_path,
        "--generator-model", "Qwen/Qwen2.5-1.5B-Instruct",
        "--output-json", str(out_json),
        "--max-model-len", "8192",
        "--gpu-memory-utilization", str(scenario.gpu_util),
        "--gpu-id", str(gpu_id),
        "--sample-queries", str(n_queries),
    ]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    print(f"  Running {scenario.id} ({scenario.desc[:40]})...", end=" ", flush=True)
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
            if result.stderr:
                lines = result.stderr.strip().split("\n")
                for ln in lines[-3:]:
                    print(f"    {ln}")
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT ({TIMEOUT_PER_RUN}s)")
    except Exception as e:
        print(f"ERROR: {e}")
    return None


def load_result(scenario: Scenario) -> Optional[Dict]:
    out_json = OUTPUT_DIR / f"result_{scenario.id}.json"
    if out_json.exists():
        with open(out_json) as f:
            return json.load(f)
    return None


# ── Analysis ───────────────────────────────────────────────────────────────────

def analyze(data: Dict) -> Dict[str, Any]:
    """Compute per-action and per-batch-type prediction errors."""
    dispatch = data.get("dispatch_trace", [])
    feedback = data.get("feedback_trace", [])
    batches = data.get("per_batch", [])

    # Build per-batch lookup: batch_index -> feedback
    fb_by_idx = {f["batch_index"]: f for f in feedback}

    # Per-action stats
    by_action: Dict[tuple, Dict] = {}
    pred_errors = []   # predicted dispatch cost vs actual
    action_correct = []  # was the predicted-min action actually chosen?
    candidate_errors = []  # for all candidate actions

    for disp in dispatch:
        idx = disp["dispatch_index"]
        fb = fb_by_idx.get(idx)
        xE = int(disp["chosen_action"].get("xE", 0))
        xR = int(disp["chosen_action"].get("xR", 0))
        B = int(disp["chosen_batch_size"])
        key = (xE, xR)

        if fb is None:
            continue

        # Actual wall time per query from feedback
        gen_ms_q = fb.get("generation_ms_per_query", 0)
        emb_ms_q = fb.get("embedding_ms_per_query", 0)
        ret_ms_q = fb.get("retrieval_ms_per_query", 0)
        actual_q = gen_ms_q + emb_ms_q + ret_ms_q
        L_q = fb.get("token_length_avg", L_DEFAULT)

        if actual_q <= 0:
            continue

        # Model prediction for the chosen action
        pred_q = predict_wall_q(xE, xR, B, L_q)

        if key not in by_action:
            by_action[key] = {"actual": [], "pred": [], "L": [], "B": []}
        by_action[key]["actual"].append(actual_q)
        by_action[key]["pred"].append(pred_q)
        by_action[key]["L"].append(L_q)
        by_action[key]["B"].append(B)

        # Prediction error for chosen action
        err_pct = (pred_q - actual_q) / actual_q * 100
        pred_errors.append(err_pct)

        # Check action selection: was the predicted-min action actually chosen?
        # Compare predicted cost for chosen action vs the min predicted of all candidates
        chosen_cost_pred = disp.get("predicted_dispatch_cost_ms_per_query", 0)
        cand_actions = disp.get("candidate_actions", [])
        if cand_actions:
            min_pred = min(c.get("cost_ms_per_query", float("inf")) for c in cand_actions)
            # Action selection error: how much worse is chosen vs min?
            if min_pred > 0:
                action_err = (chosen_cost_pred - min_pred) / min_pred * 100
                action_correct.append({"chosen": chosen_cost_pred, "min_pred": min_pred, "err": action_err})

        # Log all candidate predictions
        for c in cand_actions:
            c_xE = int(c.get("xE", 0))
            c_xR = int(c.get("xR", 0))
            c_cost = c.get("cost_ms_per_query", 0)
            c_pred = predict_wall_q(c_xE, c_xR, B, L_q)
            if c_cost > 0:
                candidate_errors.append({
                    "xE": c_xE, "xR": c_xR, "B": B,
                    "sched_cost": c_cost,
                    "model_pred": c_pred,
                    "err": (c_cost - c_pred) / c_cost * 100,
                })

    # Aggregate per-action
    action_rows = []
    for key, d in sorted(by_action.items()):
        xE, xR = key
        acts = d["actual"]
        preds = d["pred"]
        if not acts:
            continue
        errs = [(p - a) / a * 100 for a, p in zip(acts, preds) if a > 0]
        action_rows.append({
            "action": f"({xE},{xR})",
            "n": len(acts),
            "median_actual": statistics.median(acts),
            "median_pred": statistics.median(preds),
            "mean_err": statistics.mean(errs),
            "median_err": statistics.median(errs),
            "max_abs_err": max(errs, key=abs),
            "stdev_err": statistics.stdev(errs) if len(errs) > 1 else 0.0,
            "avg_B": statistics.mean(d["B"]),
            "avg_L": statistics.mean(d["L"]),
        })

    # Overall
    all_actuals = [v for d in by_action.values() for v in d["actual"]]
    all_preds = [v for d in by_action.values() for v in d["pred"]]
    if all_actuals:
        overall_errs = [(p - a) / a * 100 for a, p in zip(all_actuals, all_preds) if a > 0]
        overall = {
            "n": len(all_actuals),
            "median_actual": statistics.median(all_actuals),
            "median_pred": statistics.median(all_preds),
            "mean_err": statistics.mean(overall_errs),
            "median_err": statistics.median(overall_errs),
            "max_abs_err": max(overall_errs, key=abs),
        }
    else:
        overall = {"n": 0}

    # Action selection quality
    if candidate_errors:
        # How accurate is the scheduler's cost model vs our model?
        sched_vs_model = [(c["sched_cost"] - c["model_pred"]) / c["sched_cost"] * 100
                          for c in candidate_errors if c["sched_cost"] > 0]
        action_sel = {
            "n_candidates": len(candidate_errors),
            "sched_vs_model_mean_err": statistics.mean(sched_vs_model) if sched_vs_model else 0,
            "sched_vs_model_median_err": statistics.median(sched_vs_model) if sched_vs_model else 0,
        }
    else:
        action_sel = {"n_candidates": 0}

    return {
        "overall": overall,
        "by_action": action_rows,
        "action_selection": action_sel,
        "n_dispatches": len(dispatch),
        "n_batches": len(batches),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Validate cost model under real async_v2 workloads")
    parser.add_argument("--scenarios", nargs="+", default=["S1", "S5", "S9"])
    parser.add_argument("--all", action="store_true", help="All 12 scenarios")
    parser.add_argument("--queries", type=int, default=300,
                        help="Queries per scenario (default: 300)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reuse", action="store_true",
                        help="Skip runs that already have result JSON")
    args = parser.parse_args()

    scenarios = SCENARIOS if args.all else [s for s in SCENARIOS if s.id in args.scenarios]
    if not scenarios:
        print("No scenarios selected.")
        return

    n_q = args.queries

    print(f"\n{'='*80}")
    print(f"MODEL VALIDATION RUN")
    print(f"{'='*80}")
    print(f"  Scenarios:  {[s.id for s in scenarios]}")
    print(f"  Queries:     {n_q}")
    print(f"  Output:     {OUTPUT_DIR}")
    print(f"  Model params:")
    print(f"    gen_per_token={GEN_PER_TOKEN}, gen_base={GEN_BASE}, queue={QUEUE_PENALTY}")
    print(f"    e0={E0}(oh={OH_E0}), e1={E1}(oh={OH_E1}), r0={R0}(oh={OH_R0}), r1={R1}(oh={OH_R1}), K01={K01}")
    print(f"{'='*80}\n")

    if args.dry_run:
        for s in scenarios:
            print(f"  Would run: {s.id} ({s.desc})")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    for s in scenarios:
        print(f"\n{'─'*80}")
        print(f"  Scenario {s.id}: {s.desc}")
        print(f"{'─'*80}")

        data = load_result(s) if args.reuse else None
        if data is None:
            data = run_pipeline(s, n_q)

        if data is None:
            print(f"  SKIPPED: no result")
            all_results.append({"scenario": s.id, "status": "skipped"})
            continue

        analysis = analyze(data)

        # Print per-action results
        print(f"\n  {'Action':>8} | {'n':>4} | {'actual_ms':>10} | {'pred_ms':>10} | {'err%':>8} | {'avg_B':>6}")
        print(f"  {'-' * 65}")
        for r in analysis["by_action"]:
            print(f"  {r['action']:>8} | {r['n']:>4} | {r['median_actual']:>10.1f} | "
                  f"{r['median_pred']:>10.1f} | {r['median_err']:+8.2f}% | {r['avg_B']:>6.0f}")

        ov = analysis["overall"]
        if ov["n"] > 0:
            print(f"  {'OVERALL':>8} | {ov['n']:>4} | {ov['median_actual']:>10.1f} | "
                  f"{ov['median_pred']:>10.1f} | {ov['median_err']:+8.2f}%")

        asel = analysis["action_selection"]
        if asel["n_candidates"] > 0:
            print(f"\n  Action selection (scheduler cost vs model): "
                  f"median_err={asel['sched_vs_model_median_err']:+.2f}%, "
                  f"mean_err={asel['sched_vs_model_mean_err']:+.2f}% "
                  f"(n={asel['n_candidates']} candidate evaluations)")

        summary = {
            "scenario": s.id,
            "status": "ok",
            "overall": ov,
            "by_action": analysis["by_action"],
            "action_selection": asel,
            "n_dispatches": analysis["n_dispatches"],
        }
        all_results.append(summary)

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n\n{'='*80}")
    print(f"SUMMARY: Per-Scenario Prediction Error")
    print(f"{'='*80}")
    print(f"  {'Scenario':>8} | {'n':>5} | {'median_actual':>12} | {'median_err%':>12} | {'max_err%':>10}")
    print(f"  {'-' * 60}")

    for r in all_results:
        if r.get("status") != "ok":
            print(f"  {r['scenario']:>8} | {'--':>5} | {'--':>12} | {'SKIPPED':>12}")
            continue
        ov = r["overall"]
        max_err = max((e["max_abs_err"] for e in r["by_action"]), default=0.0)
        print(f"  {r['scenario']:>8} | {ov['n']:>5} | {ov['median_actual']:>12.1f} | "
              f"{ov['median_err']:>+12.2f}% | {max_err:>+10.2f}%")

    # Save
    out = OUTPUT_DIR / "validation_summary.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved to: {out}")

    # Diagnosis
    print(f"\n{'='*80}")
    print("DIAGNOSIS")
    print(f"{'='*80}")
    ok = [r for r in all_results if r.get("status") == "ok"]
    if not ok:
        print("  No successful runs.")
        return

    all_errs = [r["overall"]["median_err"] for r in ok if isinstance(r["overall"].get("median_err"), (int, float))]
    mean_overall = statistics.mean(all_errs) if all_errs else 0.0
    worst = max(ok, key=lambda r: abs(r["overall"].get("median_err", 0)))
    best = min(ok, key=lambda r: abs(r["overall"].get("median_err", 0)))

    print(f"  Runs: {len(ok)}/{len(all_results)} succeeded")
    print(f"  Overall median error: {mean_overall:+.2f}%")
    print(f"  Worst: {worst['scenario']} ({worst['overall'].get('median_err', 0):+.2f}%)")
    print(f"  Best:  {best['scenario']} ({best['overall'].get('median_err', 0):+.2f}%)")
    print(f"  Max absolute error: {max(abs(r['overall'].get('median_err', 0)) for r in ok):.2f}%")

    # Breakdown by action
    print(f"\n  By Action (pooled across scenarios):")
    action_pooled: Dict[str, List] = {}
    for r in ok:
        for a in r.get("by_action", []):
            k = a["action"]
            if k not in action_pooled:
                action_pooled[k] = []
            action_pooled[k].append(a["median_err"])

    print(f"  {'Action':>8} | {'n_scen':>8} | {'mean_err%':>10} | {'max_err%':>10}")
    print(f"  {'-' * 45}")
    for k in sorted(action_pooled.keys()):
        errs = action_pooled[k]
        print(f"  {k:>8} | {len(errs):>8} | {statistics.mean(errs):>+10.2f}% | {max(errs, key=abs):>+10.2f}%")


if __name__ == "__main__":
    main()
