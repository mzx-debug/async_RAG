#!/usr/bin/env python3
"""
Extended benchmark: demonstrate async_v2's advantage over serial across all B values.

The key question: async_v2 cannot know the optimal B offline.
Does it find it online? Is it faster than even the best serial?

Setup:
  serial runs at B=1, 4, 16, 32, 64 — show the full curve
  async_v2 runs with adaptive B — show where it lands on the curve
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
from typing import Dict, List, Optional

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
}


def load_queries(path: str, n: int, max_tokens: int = 8) -> List[Dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("token_length", 0) <= max_tokens:
                records.append(rec)
    random.seed(42)
    random.shuffle(records)
    return records[:n]


def build_queries(n: int) -> str:
    cfg = CORPUS_CONFIG["beir_nfcorpus"]
    queries = load_queries(cfg["queries_path"], n * 2)
    out = OUT_DIR / "queries.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    return str(out)


def run_serial(B: int, queries_path: str, n: int, gpu_util: float) -> Optional[Dict]:
    cfg = CORPUS_CONFIG["beir_nfcorpus"]
    gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    out_json = OUT_DIR / f"serial_B{B}.json"
    cmd = [
        PYTHON, str(ROOT / "async_rag_pipeline.py"),
        "--pipeline-mode", "serial",
        "--xE", "0", "--xR", "0",
        "--b", str(B),
        "--index-path", cfg["index_path"],
        "--corpus-path", cfg["corpus_path"],
        "--queries-file", queries_path,
        "--generator-model", "Qwen/Qwen2.5-1.5B-Instruct",
        "--output-json", str(out_json),
        "--max-model-len", "1024",
        "--gpu-memory-utilization", str(gpu_util),
        "--gpu-id", str(gpu_id),
        "--sample-queries", str(n),
    ]
    env = os.environ.copy()
    env["HF_ENDPOINT"] = "https://hf-mirror.com"
    print(f"  serial  B={B:<4}...", end=" ", flush=True)
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT, env=env)
        elapsed = time.time() - t0
        if r.returncode == 0 and out_json.exists():
            with open(out_json) as f:
                data = json.load(f)
            print(f"OK {elapsed:.0f}s")
            return data
        print(f"FAIL ({elapsed:.0f}s)")
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT")
    return None


def run_async_v2(initial_B: int, queries_path: str, n: int, gpu_util: float) -> Optional[Dict]:
    cfg = CORPUS_CONFIG["beir_nfcorpus"]
    gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    out_json = OUT_DIR / f"asyncv2.json"
    cmd = [
        PYTHON, str(ROOT / "async_rag_pipeline.py"),
        "--pipeline-mode", "async_v2",
        "--xE", "0", "--xR", "0",
        "--b", str(initial_B),
        "--index-path", cfg["index_path"],
        "--corpus-path", cfg["corpus_path"],
        "--queries-file", queries_path,
        "--generator-model", "Qwen/Qwen2.5-1.5B-Instruct",
        "--output-json", str(out_json),
        "--max-model-len", "1024",
        "--gpu-memory-utilization", str(gpu_util),
        "--gpu-id", str(gpu_id),
        "--sample-queries", str(n),
    ]
    env = os.environ.copy()
    env["HF_ENDPOINT"] = "https://hf-mirror.com"
    print(f"  async_v2 (init B={initial_B})...", end=" ", flush=True)
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT, env=env)
        elapsed = time.time() - t0
        if r.returncode == 0 and out_json.exists():
            with open(out_json) as f:
                data = json.load(f)
            print(f"OK {elapsed:.0f}s")
            return data
        print(f"FAIL ({elapsed:.0f}s)")
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT")
    return None


def extract(data: Optional[Dict]) -> Dict:
    if not data:
        return {}
    wall_ms = data.get("wall_time_ms", 0)
    qps = data.get("throughput_qps", 0)
    per_batch = data.get("per_batch", [])
    fb = data.get("feedback_trace", [])
    avg_B = statistics.mean(b.get("batch_size", 0) for b in per_batch) if per_batch else 0
    return {
        "wall_ms": wall_ms,
        "qps": qps,
        "n_batches": len(per_batch),
        "avg_B": avg_B,
        "feedback": fb,
    }


def plot_results(results: Dict, scenario: str):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: serial curve + async_v2 landing point ──────────────────────
    ax = axes[0]
    serial_Bs = sorted([k for k in results if k.startswith("serial_")])
    serial_walls = [results[k]["wall_ms"] for k in serial_Bs]
    serial_qps = [results[k]["qps"] for k in serial_Bs]
    serial_labels = [k.replace("serial_", "") for k in serial_Bs]

    ax.plot(serial_labels, serial_walls, "o-", color="#e74c3c", linewidth=2, markersize=8, label="serial (fixed B)")

    # Add theory curve
    gen_base = 1109.0
    gen_per_token = 25.62
    emb_ret = 0.5
    queue = 0.23
    B_theory = [1, 4, 16, 32, 64, 128]
    wall_theory = [gen_base / b + gen_per_token * 120 + emb_ret + queue for b in B_theory]
    ax.plot([str(b) for b in B_theory], [wall_theory[B_theory.index(b)] if b in [int(x) for x in serial_labels] else None for b in B_theory],
            "s--", color="#e74c3c", alpha=0.3, markersize=6, label="theory")

    # async_v2 point
    v2 = results.get("async_v2", {})
    if v2:
        avg_B_v2 = v2.get("avg_B", 0)
        ax.axhline(y=v2["wall_ms"], color="#27ae60", linewidth=1.5, linestyle="--", alpha=0.7)
        ax.scatter([f"B={avg_B_v2:.0f}"], [v2["wall_ms"]], color="#27ae60", s=150, zorder=5, label=f"async_v2 (avg B={avg_B_v2:.0f})")
        ax.annotate(f"async_v2\n{v2['wall_ms']:.0f}ms\n(QPS={v2['qps']:.1f})",
                    xy=(0, v2["wall_ms"]), xytext=(1.5, v2["wall_ms"] + 500),
                    fontsize=9, color="#27ae60",
                    arrowprops=dict(arrowstyle="->", color="#27ae60"))

    ax.set_xlabel("Batch Size (B)", fontsize=11)
    ax.set_ylabel("Wall Time (ms)", fontsize=11)
    ax.set_title("Serial Performance Curve vs async_v2\n(nfcorpus, 300 queries, gpu=0.8)", fontsize=12)
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    # ── Right: Speedup over serial B=1 ────────────────────────────────
    ax2 = axes[1]
    serial_wall_1 = results.get("serial_1", {}).get("wall_ms", None)
    if serial_wall_1:
        speedups = {}
        for k, v in results.items():
            if v.get("wall_ms"):
                speedups[k] = serial_wall_1 / v["wall_ms"]

        labels = list(speedups.keys())
        values = list(speedups.values())
        colors = ["#e74c3c" if "serial" in l else "#27ae60" for l in labels]
        ax2.bar(labels, values, color=colors, alpha=0.8)
        ax2.axhline(y=1.0, color="black", linewidth=1, linestyle="--", alpha=0.5)
        for i, (l, v) in enumerate(zip(labels, values)):
            ax2.text(i, v + 0.02, f"{v:.2f}x", ha="center", va="bottom", fontsize=9)
        ax2.set_ylabel("Speedup vs Serial (B=1)", fontsize=11)
        ax2.set_title("Speedup over Serial (B=1)", fontsize=12)
        ax2.set_ylim(0, max(values) * 1.15)
        ax2.yaxis.grid(True, alpha=0.3)

    plt.tight_layout()
    out = OUT_DIR / "serial_curve_vs_asyncv2.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=int, default=300)
    parser.add_argument("--gpu-util", type=float, default=0.8)
    args = parser.parse_args()

    queries_path = build_queries(args.queries)
    results = {}

    # Serial at multiple B values
    for B in [1, 4, 16, 32, 64]:
        data = run_serial(B, queries_path, args.queries, args.gpu_util)
        if data:
            results[f"serial_{B}"] = extract(data)

    # async_v2 with initial B=32
    data = run_async_v2(32, queries_path, args.queries, args.gpu_util)
    if data:
        results["async_v2"] = extract(data)

    # Print summary
    print(f"\n{'='*80}")
    print(f"  Results (nfcorpus, {args.queries} queries, gpu={args.gpu_util})")
    print(f"{'='*80}")
    print(f"  {'Config':<20} | {'Wall(ms)':>10} | {'QPS':>8} | {'Batches':>8} | {'Avg B':>8}")
    print(f"  {'-'*60}")

    serial_walls = {k.replace("serial_", ""): v["wall_ms"] for k, v in results.items() if "serial" in k and v.get("wall_ms")}
    if serial_walls:
        best_serial_B = min(serial_walls, key=serial_walls.get)
        best_serial_wall = serial_walls[best_serial_B]

    for k, v in sorted(results.items(), key=lambda x: x[1].get("wall_ms", 1e9)):
        wall = v.get("wall_ms", 0)
        best_spd = f" (best serial={best_serial_B}: {best_serial_wall:.0f}ms)" if "serial" in k and wall == best_serial_wall else ""
        print(f"  {k:<20} | {wall:>10.1f} | {v.get('qps', 0):>8.1f} | {v.get('n_batches', 0):>8} | {v.get('avg_B', 0):>8.0f}{best_spd}")

    if "async_v2" in results and best_serial_wall:
        v2_wall = results["async_v2"]["wall_ms"]
        spd_vs_best = best_serial_wall / v2_wall
        spd_vs_b1 = results.get("serial_1", {}).get("wall_ms", 0) / v2_wall
        print(f"\n  async_v2 vs best serial ({best_serial_B}): {spd_vs_best:.3f}x ({v2_wall:.1f}ms vs {best_serial_wall:.1f}ms)")
        print(f"  async_v2 vs serial B=1: {spd_vs_b1:.2f}x")

    plot_results(results, f"nfcorpus-gpu{args.gpu_util}")

    # Save
    out = OUT_DIR / "serial_curve_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()
