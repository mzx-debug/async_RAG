#!/usr/bin/env python3
"""
12-scenario × 3-pipeline comparison experiment.
"""

import argparse
import json
import os
import random
import shutil
import sys
import subprocess
import threading
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

BASE = ROOT

SCENARIOS: List[Dict[str, Any]] = [
    {
        "id": "S1", "corpus": "beir_nfcorpus", "index": "flat",
        "query_mode": "short", "gpu_util": 0.8,
        "desc": "nfcorpus + flat + short",
    },
    {
        "id": "S2", "corpus": "beir_nfcorpus", "index": "hnsw",
        "query_mode": "short", "gpu_util": 0.8,
        "desc": "nfcorpus + hnsw + short",
    },
    {
        "id": "S3", "corpus": "beir_fiqa", "index": "flat",
        "query_mode": "short", "gpu_util": 0.8,
        "desc": "fiqa + flat + short",
    },
    {
        "id": "S4", "corpus": "beir_fiqa", "index": "hnsw",
        "query_mode": "short", "gpu_util": 0.8,
        "desc": "fiqa + hnsw + short",
    },
    {
        "id": "S5", "corpus": "beir_fiqa", "index": "flat",
        "query_mode": "long", "gpu_util": 0.8,
        "desc": "fiqa + flat + long",
    },
    {
        "id": "S6", "corpus": "beir_fiqa", "index": "hnsw",
        "query_mode": "long", "gpu_util": 0.8,
        "desc": "fiqa + hnsw + long",
    },
    {
        "id": "S7", "corpus": "beir_nfcorpus", "index": "flat",
        "query_mode": "mixed", "gpu_util": 0.8,
        "desc": "nfcorpus+fiqa mixed + flat + mixed",
    },
    {
        "id": "S8", "corpus": "beir_nfcorpus", "index": "hnsw",
        "query_mode": "mixed", "gpu_util": 0.8,
        "desc": "nfcorpus+fiqa mixed + hnsw + mixed",
    },
    {
        "id": "S9", "corpus": "beir_nfcorpus", "index": "flat",
        "query_mode": "short", "gpu_util": 0.3,
        "desc": "nfcorpus + flat + short + gpu_util=0.3",
    },
    {
        "id": "S10", "corpus": "beir_nfcorpus", "index": "hnsw",
        "query_mode": "short", "gpu_util": 0.3,
        "desc": "nfcorpus + hnsw + short + gpu_util=0.3",
    },
    {
        "id": "S11", "corpus": "beir_fiqa", "index": "flat",
        "query_mode": "long", "gpu_util": 0.3,
        "desc": "fiqa + flat + long + gpu_util=0.3",
    },
    {
        "id": "S12", "corpus": "beir_fiqa", "index": "hnsw",
        "query_mode": "long", "gpu_util": 0.3,
        "desc": "fiqa + hnsw + long + gpu_util=0.3",
    },
]

PIPELINES = [
    {"id": "serial",  "pipeline_mode": "serial",      "xE": 0, "xR": 0, "bs": 1},
    {"id": "plain",   "pipeline_mode": "async_plain",  "xE": 0, "xR": 0, "bs": 32},
    {"id": "v3",     "pipeline_mode": "async_v2",      "xE": 0, "xR": 0, "bs": 32},
]

CORPUS_CONFIG = {
    "beir_nfcorpus": {
        "corpus_path": str(BASE / "data" / "beir_nfcorpus" / "corpus.jsonl"),
        "queries_path": str(BASE / "data" / "beir_nfcorpus" / "queries.jsonl"),
        "index_path": str(BASE / "indexes" / "beir_nfcorpus" / "faiss_{index}.index"),
        "docs": 3633,
    },
    "beir_fiqa": {
        "corpus_path": str(BASE / "data" / "beir_fiqa" / "corpus.jsonl"),
        "queries_path": str(BASE / "data" / "beir_fiqa" / "queries_with_length.jsonl"),
        "index_path": str(BASE / "indexes" / "beir_fiqa" / "faiss_{index}.index"),
        "docs": 57638,
    },
}


def load_queries(path: str, top_n: Optional[int] = None) -> List[Dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    random.shuffle(records)
    if top_n is not None and top_n < len(records):
        records = records[:top_n]
    return records


def filter_queries_by_length(queries: List[Dict[str, Any]], mode: str) -> List[Dict[str, Any]]:
    if mode == "short":
        return [q for q in queries if q.get("token_length", 0) <= 8]
    elif mode == "long":
        return [q for q in queries if q.get("token_length", 0) >= 12]
    else:
        return queries


def build_scenario_queries(scenario: Dict[str, Any], sample_queries: int) -> List[Dict[str, Any]]:
    corpus = scenario["corpus"]
    query_mode = scenario["query_mode"]

    if query_mode == "mixed":
        nfc_queries = load_queries(CORPUS_CONFIG["beir_nfcorpus"]["queries_path"])
        fiqa_queries = load_queries(CORPUS_CONFIG["beir_fiqa"]["queries_path"])
        merged = nfc_queries + fiqa_queries
        random.shuffle(merged)
        queries = merged[:sample_queries]
    else:
        queries = load_queries(CORPUS_CONFIG[corpus]["queries_path"])
        queries = filter_queries_by_length(queries, query_mode)
        queries = queries[:sample_queries]

    return queries


def extract_metrics(result: Dict[str, Any]) -> Dict[str, Any]:
    sched = result.get("scheduler", {})
    dispatch_trace = result.get("dispatch_trace", [])

    predicted_cost = None
    actual_cost = None
    if dispatch_trace:
        first = dispatch_trace[0]
        predicted_cost = first.get("predicted_dispatch_cost_ms_per_query")
        wall_ms = result.get("wall_time_ms", 0)
        n_q = result.get("num_queries", 1)
        if wall_ms and n_q:
            actual_cost = wall_ms / n_q

    action_counts: Dict[str, int] = {}
    candidate_bs_counts: Dict[int, int] = {}

    for entry in dispatch_trace:
        action_key = f"({entry.get('chosen_action', {}).get('xE', 0)},{entry.get('chosen_action', {}).get('xR', 0)})"
        action_counts[action_key] = action_counts.get(action_key, 0) + 1
        for bs in entry.get("candidate_batch_sizes", []):
            candidate_bs_counts[bs] = candidate_bs_counts.get(bs, 0) + 1

    return {
        "num_queries": result.get("num_queries", 0),
        "wall_time_ms": result.get("wall_time_ms", 0),
        "wall_throughput_qps": result.get("wall_throughput_qps", 0),
        "avg_emb_ms": result.get("avg_embedding_ms", 0),
        "avg_ret_ms": result.get("avg_retrieval_ms", 0),
        "avg_gen_ms": result.get("avg_generation_ms", 0),
        "dispatch_count": len(dispatch_trace),
        "action_counts": action_counts,
        "candidate_bs_counts": dict(sorted(candidate_bs_counts.items())),
        "predicted_cost": predicted_cost,
        "actual_cost": actual_cost,
        "cost_model_error_pct": (
            abs(predicted_cost - actual_cost) / actual_cost * 100
            if predicted_cost and actual_cost and actual_cost > 0 else None
        ),
    }


def run_one_pipeline(
    scenario_id: str,
    pipeline: Dict[str, Any],
    scenario: Dict[str, Any],
    sample_queries: int,
    out_dir: Path,
    timeout_sec: int = 600,
) -> Dict[str, Any]:
    corpus_cfg = CORPUS_CONFIG[scenario["corpus"]]
    index_path = corpus_cfg["index_path"].format(index=scenario["index"])
    queries = build_scenario_queries(scenario, sample_queries)

    tmp_q_path = out_dir / f"tmp_queries_{scenario_id}.jsonl"
    with open(tmp_q_path, "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    out_json = out_dir / f"result_{scenario_id}_{pipeline['id']}.json"

    cmd = [
        sys.executable,
        str(ROOT / "async_rag_pipeline.py"),
        "--index-path", index_path,
        "--corpus-path", corpus_cfg["corpus_path"],
        "--queries-file", str(tmp_q_path),
        "--generator-model", "Qwen/Qwen2.5-3B-Instruct",
        "--b", str(pipeline["bs"]),
        "--xE", str(pipeline["xE"]),
        "--xR", str(pipeline["xR"]),
        "--pipeline-mode", pipeline["pipeline_mode"],
        "--embedding-model", "sentence-transformers/all-MiniLM-L6-v2",
        "--embedding-max-length", "384",
        "--pooling-method", "mean",
        "--gpu-memory-utilization", str(scenario["gpu_util"]),
        "--max-output-len", "64",
        "--temperature", "0.0",
        "--top-k", "1",
        "--top-p", "1.0",
        "--output-json", str(out_json),
        "--seed", "2026",
    ]

    env = dict(os.environ)
    env["HF_ENDPOINT"] = "https://hf-mirror.com"
    env.pop("LD_LIBRARY_PATH", None)
    env.pop("CUDA_HOME", None)

    print(f"  [{scenario_id}] {pipeline['id']}: cmd = {' '.join(cmd[:10])} ...", flush=True)

    start = time.time()
    try:
        proc = subprocess.run(cmd, env=env, timeout=timeout_sec, text=True, capture_output=True)
        elapsed = time.time() - start
    except subprocess.TimeoutExpired:
        return {
            "scenario": scenario_id, "pipeline": pipeline["id"],
            "status": "TIMEOUT", "elapsed_sec": timeout_sec,
            "error": f"Timeout after {timeout_sec}s",
        }
    finally:
        if tmp_q_path.exists():
            tmp_q_path.unlink()

    if proc.returncode != 0:
        return {
            "scenario": scenario_id, "pipeline": pipeline["id"],
            "status": "FAIL", "elapsed_sec": time.time() - start,
            "stdout": proc.stdout[-2000:] if proc.stdout else "",
            "stderr": proc.stderr[-2000:] if proc.stderr else "",
        }

    if not out_json.exists():
        return {
            "scenario": scenario_id, "pipeline": pipeline["id"],
            "status": "FAIL", "elapsed_sec": time.time() - start,
            "error": "Output JSON not created",
        }

    result = json.loads(out_json.read_text(encoding="utf-8"))
    metrics = extract_metrics(result)

    return {
        "scenario": scenario_id,
        "pipeline": pipeline["id"],
        "status": "OK",
        "elapsed_sec": time.time() - start,
        "result_json": result,
        **metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="12-scenario × 3-pipeline comparison.")
    parser.add_argument("--sample-queries", type=int, default=256)
    parser.add_argument("--output-dir", type=str,
                        default=str(ROOT / "output" / "scenario_comparison"))
    parser.add_argument("--timeout-per-run", type=int, default=600)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--scenarios", type=str, default="S1-S12")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.scenarios == "S1-S12":
        run_scenarios = [s for s in SCENARIOS]
    else:
        ids = set()
        for part in args.scenarios.split(","):
            part = part.strip()
            if "-" in part:
                s, e = part.split("-")
                for i in range(int(s[1:]), int(e[1:]) + 1):
                    ids.add(f"S{i}")
            else:
                ids.add(part)
        run_scenarios = [s for s in SCENARIOS if s["id"] in ids]

    total_runs = len(run_scenarios) * len(PIPELINES)
    print(f"Running {len(run_scenarios)} scenarios × {len(PIPELINES)} pipelines = {total_runs} runs")
    print(f"Output dir: {out_dir}")
    print(f"Sample queries: {args.sample_queries}")
    print(f"Timeout per run: {args.timeout_per_run}s")
    print()

    all_results: List[Dict[str, Any]] = []
    run_idx = 0

    for scenario in run_scenarios:
        print(f"\n{'='*60}")
        print(f"  Scenario {scenario['id']}: {scenario['desc']}")
        print(f"{'='*60}")
        for pipeline in PIPELINES:
            run_idx += 1
            print(f"\n[Run {run_idx}/{total_runs}] {scenario['id']} × {pipeline['id']}")
            if args.dry_run:
                print(f"  [DRY RUN] Skipping execution")
                continue

            result = run_one_pipeline(
                scenario_id=scenario["id"],
                pipeline=pipeline,
                scenario=scenario,
                sample_queries=args.sample_queries,
                out_dir=out_dir,
                timeout_sec=args.timeout_per_run,
            )
            all_results.append(result)

            status_icon = "OK" if result.get("status") == "OK" else "FAIL"
            if result.get("status") == "OK":
                m = result
                print(f"  [{status_icon}] {m['elapsed_sec']:.1f}s | "
                      f"wall_time={m['wall_time_ms']:.1f}ms | "
                      f"qps={m['wall_throughput_qps']:.4f} | "
                      f"dispatch={m['dispatch_count']} | "
                      f"cost_err={m.get('cost_model_error_pct','N/A')}")
                if result.get("action_counts"):
                    print(f"         actions: {result['action_counts']}")
            else:
                print(f"  [{status_icon}] {result.get('error', result.get('stderr', 'unknown error'))[:120]}")

            checkpoint_path = out_dir / "checkpoint.json"
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)

    if args.dry_run:
        print("\nDry run complete.")
        return

    for scenario in run_scenarios:
        sid = scenario["id"]
        scenario_results = [r for r in all_results if r["scenario"] == sid]
        print(f"\n--- {sid}: {scenario['desc']} ---")
        header = "| pipeline | status | wall_ms | qps | dispatch | avg_emb | avg_ret | avg_gen | cost_err% | actions |"
        print(header)
        print("|" + "---|" * (header.count("|") - 1))
        for r in scenario_results:
            if r["status"] != "OK":
                print(f"| {r['pipeline']} | {r['status']} | - | - | - | - | - | - | - | - |")
                continue
            actions_str = ",".join(f"{k}={v}" for k, v in r.get("action_counts", {}).items())
            print(f"| {r['pipeline']} | {r['status']} | "
                  f"{r['wall_time_ms']:.1f} | {r['wall_throughput_qps']:.4f} | "
                  f"{r['dispatch_count']} | {r['avg_emb_ms']:.3f} | "
                  f"{r['avg_ret_ms']:.3f} | {r['avg_gen_ms']:.3f} | "
                  f"{r.get('cost_model_error_pct', 'N/A')} | "
                  f"{actions_str or '-'} |")

    print("\n--- Cross-scenario QPS comparison ---")
    print("| Scenario | serial_qps | plain_qps | v3_qps | v3_vs_serial% | v3_vs_plain% |")
    print("|---|---|---|---|---|---|")
    for scenario in run_scenarios:
        sid = scenario["id"]
        m = {r["pipeline"]: r for r in all_results if r["scenario"] == sid and r["status"] == "OK"}
        if all(k in m for k in ["serial", "plain", "v3"]):
            sq = m["serial"]["wall_throughput_qps"]
            pq = m["plain"]["wall_throughput_qps"]
            vq = m["v3"]["wall_throughput_qps"]
            vs = (vq / sq - 1) * 100 if sq > 0 else float("nan")
            vp = (vq / pq - 1) * 100 if pq > 0 else float("nan")
            print(f"| {sid} | {sq:.4f} | {pq:.4f} | {vq:.4f} | {vs:+.1f}% | {vp:+.1f}% |")

    results_path = out_dir / "all_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)

    checkpoint_path = out_dir / "checkpoint.json"
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
