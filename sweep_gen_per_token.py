#!/usr/bin/env python3
"""
Measure gen_per_token at different gpu_util values.

For each gpu_util in {0.3, 0.5, 0.8}:
  Run async_plain (0,0), measure gen_ms and output tokens per batch,
  fit gen_per_token = f(gpu_util).

This tells us how much gen_per_token varies so the cost model can
be adjusted accordingly.
"""

import argparse
import json
import os
import random
import statistics
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
PYTHON = "/home/cloudteam/Software/conda/envs/p702/bin/python"
OUT_DIR = ROOT / "output" / "gen_token_sweep"
TIMEOUT = 600
QUERIES = 200

CORPUS_CONFIG = {
    "beir_nfcorpus": {
        "corpus_path": str(ROOT / "data" / "beir_nfcorpus" / "corpus.jsonl"),
        "queries_path": str(ROOT / "data" / "beir_nfcorpus" / "queries.jsonl"),
        "index_path": str(ROOT / "indexes" / "beir_nfcorpus" / "faiss_flat.index"),
    },
}


def load_queries_short(path: str, n: int) -> list:
    records = []
    with open(path) as f:
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
    with open(out, "w") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    return str(out)


def run_at_gpu_util(gpu_util: float, batch_size: int, queries_file: str) -> dict:
    cfg = CORPUS_CONFIG["beir_nfcorpus"]
    gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    out_json = OUT_DIR / f"gpu{gpu_util}_b{batch_size}.json"

    cmd = [
        PYTHON, str(ROOT / "async_rag_pipeline.py"),
        "--pipeline-mode", "async_plain",
        "--xE", "0", "--xR", "0",
        "--b", str(batch_size),
        "--fixed-action",
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

    label = f"gpu={gpu_util}, B={batch_size}"
    print(f"  {label}...", end=" ", flush=True)
    t0 = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT, env=env)
        elapsed = time.time() - t0
        if result.returncode == 0 and out_json.exists():
            with open(out_json) as f:
                data = json.load(f)
            print(f"OK ({elapsed:.0f}s)")
            return data
        print(f"FAIL ({elapsed:.0f}s)")
        if result.stderr:
            for ln in result.stderr.strip().split("\n"):
                if "error" in ln.lower() or "cuda" in ln.lower():
                    print(f"    {ln[:100]}")
    except subprocess.TimeoutExpired:
        print("TIMEOUT")
    return {}


def extract_gen_params(data: dict) -> dict:
    pb = data.get("per_batch", [])
    fb = data.get("feedback_trace", [])

    # Use per_batch for total gen time and tokens
    total_gen_ms = sum(b.get("generation_sec", 0) * 1000 for b in pb)
    total_tokens = sum(b.get("generated_tokens", 0) for b in pb)
    n_batches = len(pb)

    if total_tokens == 0:
        return {}

    # gen_per_token = total_gen_ms / total_tokens (ms per token)
    gen_pt = total_gen_ms / total_tokens
    avg_out = total_tokens / n_batches if n_batches else 0

    return {
        "gen_per_token": gen_pt,
        "avg_output": avg_out,
        "total_gen_ms": total_gen_ms,
        "total_tokens": total_tokens,
        "n_batches": n_batches,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-utils", default="0.3,0.5,0.8", help="Comma-separated gpu_util values")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    gpu_utils = [float(x) for x in args.gpu_utils.split(",")]

    global queries_path
    queries_path = build_queries()

    results = {}
    for gpu_util in gpu_utils:
        data = run_at_gpu_util(gpu_util, args.batch_size, queries_path)
        if data:
            params = extract_gen_params(data)
            results[gpu_util] = params
            print(f"    → gen_per_token={params.get('gen_per_token', 0):.4f}, "
                  f"avg_out={params.get('avg_output', 0):.1f}, n_batches={params.get('n_batches', 0)}")

    # ── Print summary ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  gen_per_token by gpu_util")
    print(f"{'='*60}")
    print(f"  {'gpu_util':>10} | {'gen_pt':>10} | {'avg_out':>10} | {'n':>5}")
    print(f"  {'-'*45}")
    for gpu_util, params in sorted(results.items()):
        print(f"  {gpu_util:>10.1f} | {params.get('gen_per_token', 0):>10.4f} | "
              f"{params.get('avg_output', 0):>10.1f} | {params.get('n', 0):>5}")

    # Save
    out = OUT_DIR / "gen_token_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    queries_path = ""
    main()
