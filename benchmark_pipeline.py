#!/usr/bin/env python3
"""
Compare pipeline modes to demonstrate async_v2's superiority.

Comparisons:
  1. serial     vs async_plain  → measures pipeline overlap benefit
  2. async_plain vs async_v2   → measures adaptive scheduling benefit
  3. async_v2(fixed) vs async_v2(adaptive) → measures adaptive action selection

Configuration: nfcorpus + flat + short + gpu=0.8 + (xE=0,xR=0)
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).parent.resolve()
PYTHON = "/home/cloudteam/Software/conda/envs/p702/bin/python"
OUT_DIR = ROOT / "output" / "pipeline_comparison"
TIMEOUT = 600

CORPUS_CONFIG = {
    "beir_nfcorpus": {
        "corpus_path": str(ROOT / "data" / "beir_nfcorpus" / "corpus.jsonl"),
        "queries_path": str(ROOT / "data" / "beir_nfcorpus" / "queries.jsonl"),
        "index_path": str(ROOT / "indexes" / "beir_nfcorpus" / "faiss_flat.index"),
    },
    "beir_fiqa": {
        "corpus_path": str(ROOT / "data" / "beir_fiqa" / "corpus.jsonl"),
        "queries_path": str(ROOT / "data" / "beir_fiqa" / "queries.jsonl"),
        "index_path": str(ROOT / "indexes" / "beir_fiqa" / "faiss_flat.index"),
    },
}

SCENARIOS = {
    "S1": {"corpus": "beir_nfcorpus", "index": "flat", "query_mode": "short", "gpu_util": 0.8},
    "S5": {"corpus": "beir_fiqa",     "index": "flat", "query_mode": "long",  "gpu_util": 0.8},
}


@dataclass
class RunResult:
    mode: str
    action: str
    wall_ms: float
    throughput_qps: float
    n_batches: int
    avg_batch_size: float
    feedback: List[Dict] = field(default_factory=list)
    error: Optional[str] = None


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
    import random
    random.seed(42)
    random.shuffle(records)
    return records[:n]


def build_query_file(corpus_key: str, n: int) -> str:
    cfg = CORPUS_CONFIG[corpus_key]
    queries = load_queries_short(cfg["queries_path"], n)
    out = OUT_DIR / f"queries_{corpus_key}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    return str(out)


def run_pipeline(
    mode: str,
    xE: int,
    xR: int,
    batch_size: int,
    queries_path: str,
    corpus_key: str,
    gpu_util: float,
    fixed_action: bool = False,
    n_queries: int = 300,
) -> RunResult:
    cfg = CORPUS_CONFIG[corpus_key]
    gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    out_json = OUT_DIR / f"run_{mode}_xE{xE}_xR{xR}_b{batch_size}_gpu{gpu_util}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        PYTHON, str(ROOT / "async_rag_pipeline.py"),
        "--pipeline-mode", mode,
        "--xE", str(xE), "--xR", str(xR),
        "--b", str(batch_size),
        "--index-path", cfg["index_path"],
        "--corpus-path", cfg["corpus_path"],
        "--queries-file", queries_path,
        "--generator-model", "Qwen/Qwen2.5-1.5B-Instruct",
        "--output-json", str(out_json),
        "--max-model-len", "1024",
        "--gpu-memory-utilization", str(gpu_util),
        "--gpu-id", str(gpu_id),
        "--sample-queries", str(n_queries),
    ]
    if fixed_action:
        cmd.append("--fixed-action")

    env = os.environ.copy()
    env["HF_ENDPOINT"] = "https://hf-mirror.com"
    env["PYTHONUNBUFFERED"] = "1"

    label = f"{mode} xE={xE},xR={xR}, B={batch_size}, gpu={gpu_util}"
    if fixed_action:
        label += " fixed"
    print(f"  {label}...", end=" ", flush=True)
    t0 = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT, env=env)
        elapsed = time.time() - t0
        if result.returncode == 0 and out_json.exists():
            with open(out_json) as f:
                data = json.load(f)
            wall_ms = data.get("wall_time_ms", elapsed * 1000)
            qps = data.get("throughput_qps", 0.0)
            per_batch = data.get("per_batch", [])
            avg_B = statistics.mean(b.get("batch_size", 0) for b in per_batch) if per_batch else 0.0
            fb = data.get("feedback_trace", [])
            print(f"OK ({elapsed:.0f}s, {qps:.1f} qps)")
            return RunResult(
                mode=mode,
                action=f"({xE},{xR})",
                wall_ms=wall_ms,
                throughput_qps=qps,
                n_batches=len(per_batch),
                avg_batch_size=avg_B,
                feedback=fb,
            )
        else:
            err = result.stderr.strip().split("\n")[-1] if result.stderr else "unknown"
            print(f"FAIL ({elapsed:.0f}s): {err}")
            return RunResult(mode=mode, action=f"({xE},{xR})", wall_ms=float("inf"),
                             throughput_qps=0, n_batches=0, avg_batch_size=0,
                             error=err)
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT ({time.time()-t0:.0f}s)")
        return RunResult(mode=mode, action=f"({xE},{xR})", wall_ms=float("inf"),
                         throughput_qps=0, n_batches=0, avg_batch_size=0, error="timeout")
    except Exception as e:
        print(f"ERROR: {e}")
        return RunResult(mode=mode, action=f"({xE},{xR})", wall_ms=float("inf"),
                         throughput_qps=0, n_batches=0, avg_batch_size=0, error=str(e))


def extract_cost_model_fits(fb: List[Dict]) -> Dict:
    """Fit gen_per_token and gen_base from feedback trace."""
    if not fb:
        return {}
    gen_ms = [f.get("generation_ms_per_query", 0) for f in fb if f.get("generation_ms_per_query", 0) > 0]
    B_vals = [f.get("batch_size", 1) for f in fb if f.get("generation_ms_per_query", 0) > 0]
    if len(gen_ms) >= 2:
        import math
        log_B = [math.log(b) for b in B_vals]
        log_g = [math.log(g) for g in gen_ms]
        n = len(log_B)
        sx = sum(log_B)
        sy = sum(log_g)
        sxy = sum(x * y for x, y in zip(log_B, log_g))
        sxx = sum(x * x for x in log_B)
        d = n * sxx - sx * sx
        if abs(d) > 1e-9:
            a = (n * sxy - sx * sy) / d
            b_int = (sy - a * sx) / n
            gen_base = math.exp(b_int)
            gen_per_token = a
            return {"gen_base": gen_base, "gen_per_token": gen_per_token}
    return {}


def print_comparison_table(results: List[RunResult], scenario: str):
    print(f"\n{'='*90}")
    print(f"  {scenario} — Pipeline Comparison Results")
    print(f"{'='*90}")
    print(f"  {'Mode':<25} | {'Action':>8} | {'Wall(ms)':>10} | {'QPS':>8} | {'Batches':>7} | {'Avg B':>7}")
    print(f"  {'-'*75}")

    for r in results:
        if r.error:
            print(f"  {r.mode:<25} | {r.action:>8} | {'FAIL':>10} | {'—':>8} | {r.n_batches:>7} | {r.avg_batch_size:>7.1f}")
        else:
            print(f"  {r.mode:<25} | {r.action:>8} | {r.wall_ms:>10.1f} | {r.throughput_qps:>8.1f} | {r.n_batches:>7} | {r.avg_batch_size:>7.1f}")

    # Compute speedups
    serial = next((r for r in results if r.mode == "serial" and not r.error), None)
    async_plain = next((r for r in results if r.mode == "async_plain" and not r.error), None)
    async_v2_fixed = next((r for r in results if r.mode == "async_v2_fixed" and not r.error), None)
    async_v2_adapt = next((r for r in results if r.mode == "async_v2" and not r.error), None)

    print(f"\n{'='*90}")
    print(f"  Speedup Analysis (wall time)")
    print(f"{'='*90}")

    def speedup(baseline: RunResult, current: RunResult, label: str):
        if baseline and current and baseline.wall_ms < float("inf") and current.wall_ms < float("inf"):
            s = baseline.wall_ms / current.wall_ms
            print(f"  {label:<45}: {s:.2f}x  ({baseline.wall_ms:.1f}ms → {current.wall_ms:.1f}ms)")
            return s
        return None

    if serial and async_plain:
        speedup(serial, async_plain, "Pipeline overlap (serial → async_plain)")

    if serial and async_v2_fixed:
        speedup(serial, async_v2_fixed, "Pipeline + dyn-B (serial → async_v2_fixed)")

    if serial and async_v2_adapt:
        speedup(serial, async_v2_adapt, "Full async_v2 (serial → async_v2_adapt)")

    if async_plain and async_v2_fixed:
        speedup(async_plain, async_v2_fixed, "Dyn batch only (async_plain → async_v2_fixed)")

    if async_plain and async_v2_adapt:
        speedup(async_plain, async_v2_adapt, "Adaptive sched (async_plain → async_v2)")

    if async_v2_fixed and async_v2_adapt:
        speedup(async_v2_fixed, async_v2_adapt, "Adaptive action (fixed → adaptive)")

    # Per-batch breakdown from feedback trace
    print(f"\n{'='*90}")
    print(f"  Per-Batch Latency Breakdown (async_v2)")
    print(f"{'='*90}")
    if async_v2_adapt and async_v2_adapt.feedback:
        fb = async_v2_adapt.feedback
        emb = [f.get("embedding_ms_per_query", 0) for f in fb if f.get("embedding_ms_per_query", 0) > 0]
        ret = [f.get("retrieval_ms_per_query", 0) for f in fb if f.get("retrieval_ms_per_query", 0) > 0]
        gen = [f.get("generation_ms_per_query", 0) for f in fb if f.get("generation_ms_per_query", 0) > 0]
        if emb:
            print(f"  Embedding:   avg={statistics.mean(emb):.4f} ms/q  min={min(emb):.4f}  max={max(emb):.4f}")
        if ret:
            print(f"  Retrieval:   avg={statistics.mean(ret):.4f} ms/q  min={min(ret):.4f}  max={max(ret):.4f}")
        if gen:
            print(f"  Generation:  avg={statistics.mean(gen):.2f} ms/q  min={min(gen):.2f}  max={max(gen):.2f}")

        fits = extract_cost_model_fits(fb)
        if fits:
            print(f"  Gen model:   gen_base={fits['gen_base']:.1f}ms  gen_per_token={fits['gen_per_token']:.4f}ms/tok")

        # Show action distribution
        action_counts = {}
        for f in fb:
            a = f.get("action_taken", "(?)")
            action_counts[a] = action_counts.get(a, 0) + 1
        print(f"  Action dist: {dict(sorted(action_counts.items()))}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", default="S1,S5", help="Comma-separated scenario IDs")
    parser.add_argument("--queries", type=int, default=300, help="Number of queries per run")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for fixed-B modes")
    args = parser.parse_args()

    scenario_ids = args.scenarios.split(",")

    all_results = {}
    for sid in scenario_ids:
        if sid not in SCENARIOS:
            print(f"Unknown scenario: {sid}")
            continue
        sc = SCENARIOS[sid]
        corpus_key = sc["corpus"]
        gpu_util = sc["gpu_util"]

        print(f"\n{'#'*90}")
        print(f"  Scenario {sid}: {corpus_key}, gpu={gpu_util}")
        print(f"{'#'*90}")

        # Build query file once per corpus
        queries_path = build_query_file(corpus_key, args.queries * 2)

        results = []

        # Run 1: serial (baseline)
        r = run_pipeline("serial", xE=0, xR=0, batch_size=args.batch_size,
                         queries_path=queries_path, corpus_key=corpus_key, gpu_util=gpu_util,
                         n_queries=args.queries)
        results.append(r)

        # Run 2: async_plain (pipeline overlap only, fixed action)
        r = run_pipeline("async_plain", xE=0, xR=0, batch_size=args.batch_size,
                         queries_path=queries_path, corpus_key=corpus_key, gpu_util=gpu_util,
                         n_queries=args.queries)
        results.append(r)

        # Run 3: async_v2 with fixed action (pipeline overlap + dynamic batch)
        r = run_pipeline("async_v2_fixed", xE=0, xR=0, batch_size=args.batch_size,
                         queries_path=queries_path, corpus_key=corpus_key, gpu_util=gpu_util,
                         fixed_action=True, n_queries=args.queries)
        results.append(r)

        # Run 4: async_v2 adaptive (full adaptive scheduling)
        r = run_pipeline("async_v2", xE=0, xR=0, batch_size=args.batch_size,
                         queries_path=queries_path, corpus_key=corpus_key, gpu_util=gpu_util,
                         n_queries=args.queries)
        results.append(r)

        all_results[sid] = results
        print_comparison_table(results, sid)

    # ── Aggregate summary ────────────────────────────────────────────────────
    print(f"\n\n{'='*90}")
    print(f"  AGGREGATE SUMMARY ACROSS SCENARIOS")
    print(f"{'='*90}")
    print(f"  {'Scenario':>8} | {'Mode':<20} | {'Wall(ms)':>10} | {'QPS':>8} | {'speedup':>8}")
    print(f"  {'-'*65}")

    for sid, results in all_results.items():
        serial_r = next((r for r in results if r.mode == "serial" and not r.error), None)
        for r in results:
            if r.error:
                continue
            if serial_r and serial_r.wall_ms < float("inf"):
                spd = serial_r.wall_ms / r.wall_ms
                print(f"  {sid:>8} | {r.mode:<20} | {r.wall_ms:>10.1f} | {r.throughput_qps:>8.1f} | {spd:>8.2f}x")
            else:
                print(f"  {sid:>8} | {r.mode:<20} | {r.wall_ms:>10.1f} | {r.throughput_qps:>8.1f} | {'—':>8}")

    # ── Save results ───────────────────────────────────────────────────────
    out = OUT_DIR / "benchmark_results.json"
    serializable = []
    for sid, results in all_results.items():
        for r in results:
            serializable.append({
                "scenario": sid,
                "mode": r.mode,
                "action": r.action,
                "wall_ms": r.wall_ms if r.wall_ms < float("inf") else None,
                "throughput_qps": r.throughput_qps,
                "n_batches": r.n_batches,
                "avg_batch_size": r.avg_batch_size,
                "error": r.error,
                "feedback_summary": {
                    "n": len(r.feedback),
                    "actions": list(set(f.get("action_taken", "?") for f in r.feedback)) if r.feedback else [],
                },
            })
    with open(out, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  Saved to: {out}")


if __name__ == "__main__":
    main()
