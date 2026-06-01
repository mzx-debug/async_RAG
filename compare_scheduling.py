#!/usr/bin/env python3
"""
Compare async_v2 scheduling vs async_plain (serial B=32) at different gpu_util values.
Measures actual latency and throughput to verify the scheduling benefit.
"""

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
PYTHON = "/home/cloudteam/Software/conda/envs/p702/bin/python"
OUT_DIR = ROOT / "output" / "gpu_util_comparison"
TIMEOUT = 300
QUERIES = 300

CORPUS_CONFIG = {
    "beir_nfcorpus": {
        "corpus_path": str(ROOT / "data" / "beir_nfcorpus" / "corpus.jsonl"),
        "queries_path": str(ROOT / "data" / "beir_nfcorpus" / "queries.jsonl"),
        "index_path": str(ROOT / "indexes" / "beir_nfcorpus" / "faiss_flat.index"),
    },
}


def load_queries_short(path: str, n: int) -> list:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("token_length", 0) <= 8:
                records.append(rec)
    random.seed(42)
    random.shuffle(records)
    return records[:n]


def build_queries() -> str:
    cfg = CORPUS_CONFIG["beir_nfcorpus"]
    queries = load_queries_short(cfg["queries_path"], QUERIES * 2)
    out = OUT_DIR / "queries.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    return str(out)


@dataclass
class RunResult:
    mode: str
    gpu_util: float
    batch_size: int
    actual_ms_q: float  # median actual wall time per query (ms)
    total_time: float    # total wall time (s)
    throughput: float    # queries per second
    dispatch_cost_avg: float  # predicted dispatch cost from trace
    n_batches: int
    n_queries: int
    feedback_trace: list
    dispatch_trace: list


def run_async_v2(queries_file: str, gpu_util: float, batch_size: int) -> RunResult:
    cfg = CORPUS_CONFIG["beir_nfcorpus"]
    out_json = OUT_DIR / f"async_v2_gpu{gpu_util}_b{batch_size}.json"
    gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])

    cmd = [
        PYTHON, str(ROOT / "async_rag_pipeline.py"),
        "--pipeline-mode", "async_v2",
        "--xE", "0", "--xR", "0",
        "--b", str(batch_size),
        "--index-path", cfg["index_path"],
        "--corpus-path", cfg["corpus_path"],
        "--queries-file", queries_file,
        "--generator-model", "Qwen/Qwen2.5-1.5B-Instruct",
        "--output-json", str(out_json),
        "--max-model-len", "8192",
        "--gpu-memory-utilization", str(gpu_util),
        "--gpu-id", str(gpu_id),
        "--sample-queries", str(QUERIES),
    ]
    env = os.environ.copy()
    env["HF_ENDPOINT"] = "https://hf-mirror.com"
    env["PYTHONUNBUFFERED"] = "1"

    label = f"async_v2 gpu={gpu_util} B={batch_size}"
    print(f"  {label}...", end=" ", flush=True)
    t0 = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT, env=env)
        elapsed = time.time() - t0
        if result.returncode == 0 and out_json.exists():
            with open(out_json) as f:
                data = json.load(f)
            fb = data.get("feedback_trace", [])
            dispatch = data.get("dispatch_trace", [])
            actual_times = [e.get("wall_time_ms", 0) for e in fb if e.get("wall_time_ms", 0) > 0]
            actual_ms_q = statistics.median(actual_times) if actual_times else 0
            n_batches = len(fb)
            n_queries = sum(e.get("batch_size", 0) for e in fb)
            dispatch_costs = [e.get("predicted_dispatch_cost_ms_per_query", 0) for e in dispatch if e.get("predicted_dispatch_cost_ms_per_query", 0) > 0]
            dispatch_cost_avg = statistics.mean(dispatch_costs) if dispatch_costs else 0
            throughput = n_queries / elapsed if elapsed > 0 else 0
            print(f"OK ({elapsed:.0f}s, {actual_ms_q:.1f}ms/q, {throughput:.1f} q/s)")
            return RunResult(
                mode="async_v2", gpu_util=gpu_util, batch_size=batch_size,
                actual_ms_q=actual_ms_q, total_time=elapsed, throughput=throughput,
                dispatch_cost_avg=dispatch_cost_avg, n_batches=n_batches, n_queries=n_queries,
                feedback_trace=fb, dispatch_trace=dispatch,
            )
        print(f"FAIL ({elapsed:.0f}s)")
        if result.stderr:
            for ln in result.stderr.strip().split("\n"):
                if "error" in ln.lower() or "cuda" in ln.lower():
                    print(f"    {ln[:120]}")
    except subprocess.TimeoutExpired:
        print("TIMEOUT")
    return RunResult(mode="async_v2", gpu_util=gpu_util, batch_size=batch_size,
                     actual_ms_q=0, total_time=0, throughput=0, dispatch_cost_avg=0,
                     n_batches=0, n_queries=0, feedback_trace=[], dispatch_trace=[])


def run_async_plain(queries_file: str, gpu_util: float, batch_size: int) -> RunResult:
    cfg = CORPUS_CONFIG["beir_nfcorpus"]
    out_json = OUT_DIR / f"async_plain_gpu{gpu_util}_b{batch_size}.json"
    gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])

    cmd = [
        PYTHON, str(ROOT / "async_rag_pipeline.py"),
        "--pipeline-mode", "async_plain",
        "--xE", "0", "--xR", "0",
        "--b", str(batch_size),
        "--index-path", cfg["index_path"],
        "--corpus-path", cfg["corpus_path"],
        "--queries-file", queries_file,
        "--generator-model", "Qwen/Qwen2.5-1.5B-Instruct",
        "--output-json", str(out_json),
        "--max-model-len", "8192",
        "--gpu-memory-utilization", str(gpu_util),
        "--gpu-id", str(gpu_id),
        "--sample-queries", str(QUERIES),
    ]
    env = os.environ.copy()
    env["HF_ENDPOINT"] = "https://hf-mirror.com"
    env["PYTHONUNBUFFERED"] = "1"

    label = f"async_plain gpu={gpu_util} B={batch_size}"
    print(f"  {label}...", end=" ", flush=True)
    t0 = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT, env=env)
        elapsed = time.time() - t0
        if result.returncode == 0 and out_json.exists():
            with open(out_json) as f:
                data = json.load(f)
            fb = data.get("feedback_trace", [])
            dispatch = data.get("dispatch_trace", [])
            actual_times = [e.get("wall_time_ms", 0) for e in fb if e.get("wall_time_ms", 0) > 0]
            actual_ms_q = statistics.median(actual_times) if actual_times else 0
            n_batches = len(fb)
            n_queries = sum(e.get("batch_size", 0) for e in fb)
            dispatch_costs = [e.get("predicted_dispatch_cost_ms_per_query", 0) for e in dispatch if e.get("predicted_dispatch_cost_ms_per_query", 0) > 0]
            dispatch_cost_avg = statistics.mean(dispatch_costs) if dispatch_costs else 0
            throughput = n_queries / elapsed if elapsed > 0 else 0
            print(f"OK ({elapsed:.0f}s, {actual_ms_q:.1f}ms/q, {throughput:.1f} q/s)")
            return RunResult(
                mode="async_plain", gpu_util=gpu_util, batch_size=batch_size,
                actual_ms_q=actual_ms_q, total_time=elapsed, throughput=throughput,
                dispatch_cost_avg=dispatch_cost_avg, n_batches=n_batches, n_queries=n_queries,
                feedback_trace=fb, dispatch_trace=dispatch,
            )
        print(f"FAIL ({elapsed:.0f}s)")
        if result.stderr:
            for ln in result.stderr.strip().split("\n"):
                if "error" in ln.lower() or "cuda" in ln.lower():
                    print(f"    {ln[:120]}")
    except subprocess.TimeoutExpired:
        print("TIMEOUT")
    return RunResult(mode="async_plain", gpu_util=gpu_util, batch_size=batch_size,
                     actual_ms_q=0, total_time=0, throughput=0, dispatch_cost_avg=0,
                     n_batches=0, n_queries=0, feedback_trace=[], dispatch_trace=[])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-utils", default="0.3,0.5,0.8", help="Comma-separated")
    parser.add_argument("--batch-sizes", default="8,16,32", help="Comma-separated")
    parser.add_argument("--modes", default="async_v2,async_plain", help="Comma-separated")
    args = parser.parse_args()
    gpu_utils = [float(x) for x in args.gpu_utils.split(",")]
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    modes = args.modes.split(",")

    queries_file = build_queries()

    results = {}
    for gpu_util in gpu_utils:
        results[gpu_util] = {}
        for mode in modes:
            results[gpu_util][mode] = {}
            for B in batch_sizes:
                if mode == "async_v2":
                    r = run_async_v2(queries_file, gpu_util, B)
                else:
                    r = run_async_plain(queries_file, gpu_util, B)
                results[gpu_util][mode][B] = r

    # ── Summary table ───────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print(f"  SUMMARY: Scheduling benefit by gpu_util")
    print(f"{'='*90}")
    print(f"  {'gpu_util':>8} | {'Mode':>12} | {'B':>4} | {'actual_ms/q':>12} | {'throughput':>10} | {'vs async_plain':>12}")
    print(f"  {'-'*70}")

    for gpu_util in sorted(results.keys()):
        for mode in modes:
            for B in batch_sizes:
                r = results[gpu_util][mode].get(B)
                if not r or r.actual_ms_q == 0:
                    continue
                # Find baseline (async_plain B=32 at same gpu_util)
                baseline = results[gpu_util].get("async_plain", {}).get(32)
                if baseline and baseline.actual_ms_q > 0:
                    improvement = (baseline.actual_ms_q - r.actual_ms_q) / baseline.actual_ms_q * 100
                    improvement_str = f"{improvement:+.1f}%"
                else:
                    improvement_str = "N/A"
                print(f"  {gpu_util:>8.1f} | {mode:>12} | {B:>4} | {r.actual_ms_q:>12.1f} | "
                      f"{r.throughput:>10.1f} | {improvement_str:>12}")

    # Save results
    serializable = {}
    for gpu_util, by_mode in results.items():
        serializable[str(gpu_util)] = {}
        for mode, by_B in by_mode.items():
            serializable[str(gpu_util)][mode] = {}
            for B, r in by_B.items():
                serializable[str(gpu_util)][mode][str(B)] = {
                    "actual_ms_q": r.actual_ms_q,
                    "total_time": r.total_time,
                    "throughput": r.throughput,
                    "dispatch_cost_avg": r.dispatch_cost_avg,
                    "n_batches": r.n_batches,
                    "n_queries": r.n_queries,
                }

    out = OUT_DIR / "comparison_results.json"
    with open(out, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()
