#!/usr/bin/env python3

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


def run_one(mode: str, script: Path, base_args: List[str], out_dir: Path, prefix: str = "") -> Dict[str, Any]:
    out_name = f"summary_{prefix}_{mode}.json" if prefix else f"summary_{mode}.json"
    out_path = out_dir / out_name
    cmd = [sys.executable, str(script)] + base_args + ["--pipeline-mode", mode, "--output-json", str(out_path)]
    print("Running:", " ".join(cmd))
    import os
    env = os.environ.copy()
    env["HF_ENDPOINT"] = "https://hf-mirror.com"
    proc = subprocess.run(cmd, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"Mode {mode} failed with code {proc.returncode}")
    if not out_path.exists():
        raise FileNotFoundError(f"Missing output: {out_path}")
    return json.loads(out_path.read_text(encoding="utf-8"))


def make_table(rows: List[Dict[str, Any]]) -> str:
    header = [
        "group",
        "mode",
        "execution",
        "num_queries",
        "wall_time_ms",
        "wall_qps",
        "total_ms(sum stages)",
        "avg_emb_ms",
        "avg_ret_ms",
        "avg_gen_ms",
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("group", "")),
                    str(row["mode"]),
                    str(row["execution"]),
                    str(row["num_queries"]),
                    f"{row['wall_time_ms']:.2f}",
                    f"{row['wall_qps']:.4f}",
                    f"{row['total_ms']:.2f}",
                    f"{row['avg_emb_ms']:.4f}",
                    f"{row['avg_ret_ms']:.4f}",
                    f"{row['avg_gen_ms']:.4f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run serial/async_plain/async_v2 comparison (V1: resource-constrained defaults).")
    parser.add_argument("--workdir", type=str, default=".", help="Directory containing async_rag_pipeline.py")
    parser.add_argument("--index-path", type=str,
                        default="./indexes/beir_nfcorpus/faiss.index",
                        help="FAISS index path (default: ./indexes/beir_nfcorpus/faiss.index)")
    parser.add_argument("--corpus-path", type=str,
                        default="./data/beir_nfcorpus/corpus.jsonl",
                        help="Corpus path (default: ./data/beir_nfcorpus/corpus.jsonl)")
    parser.add_argument("--generator-model", type=str,
                        default="Qwen/Qwen2.5-3B-Instruct",
                        help="Generator model for vLLM (default: Qwen/Qwen2.5-1.5B-Instruct)")
    parser.add_argument("--embedding-model", type=str,
                        default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Embedding model path or HuggingFace id. Must match the index.")
    parser.add_argument("--embedding-max-length", type=int, default=384)
    parser.add_argument("--pooling-method", type=str, default="mean",
                        choices=["mean", "cls", "pooler"])
    parser.add_argument("--embedding-use-fp16", action="store_true", default=True)
    parser.add_argument("--queries-file", type=str, default=None,
                        help="Query file (queries.jsonl from generate_queries.py). "
                             "Example: ./data/beir_nfcorpus/queries.jsonl")
    parser.add_argument("--sample-queries", type=int, default=256)
    parser.add_argument("--b", type=int, default=32)
    parser.add_argument("--xE", type=int, default=1)
    parser.add_argument("--xR", type=int, default=0)
    parser.add_argument("--nprobe", type=int, default=1,
                        help="FAISS nprobe for IVF index, or 1 for Flat index (default: 1)")
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--gpu-id", type=str, default="0")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument("--output-dir", type=str, default="./output/comparison")
    parser.add_argument("--ema-params-path", type=str,
                        default=None,
                        help="Path to EMA parameters JSON from auto_tune.py. "
                             "If not set, EMA params are not loaded.")
    args = parser.parse_args()

    workdir = Path(args.workdir).expanduser().resolve()
    script = workdir / "async_rag_pipeline.py"
    if not script.exists():
        raise FileNotFoundError(script)

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    base_args: List[str] = [
        "--index-path",         str(args.index_path),
        "--corpus-path",        str(args.corpus_path),
        "--generator-model",    str(args.generator_model),
        "--b",                  str(args.b),
        "--xE",                 str(args.xE),
        "--xR",                 str(args.xR),
        "--nprobe",             str(args.nprobe),
        "--topk",               str(args.topk),
        "--sample-queries",     str(args.sample_queries),
        "--gpu-id",             str(args.gpu_id),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--embedding-model",       str(args.embedding_model),
        "--embedding-max-length",  str(args.embedding_max_length),
        "--pooling-method",       str(args.pooling_method),
        "--embedding-use-fp16",
    ]
    if args.queries_file:
        base_args += ["--queries-file", str(args.queries_file)]
    if args.ema_params_path:
        base_args += ["--ema-params-path", str(args.ema_params_path)]

    # Run with CPU retrieval (xR=0) and GPU retrieval (xR=1).
    # Each retrieval config runs serial, async_plain, async_v2.
    modes_by_retrieval = {
        "cpu_retrieval": {"xR": 0, "modes": ["serial", "async_plain", "async_v2"]},
        "gpu_retrieval": {"xR": 1, "modes": ["serial", "async_plain", "async_v2"]},
    }

    rows = []
    for group_name, group_cfg in modes_by_retrieval.items():
        xR_val = group_cfg["xR"]
        modes = group_cfg["modes"]
        print(f"\n{'='*60}")
        print(f"  {group_name.upper()} (xR={xR_val})")
        print(f"{'='*60}")
        group_args = base_args.copy()
        group_args[group_args.index("--xR") + 1] = str(xR_val)
        for mode in modes:
            print(f"\n>>> {mode}")
            s = run_one(mode, script, group_args, out_dir, prefix=group_name)
            rows.append(
                {
                    "group": group_name,
                    "mode": mode,
                    "execution": s.get("scheduler", {}).get("execution", ""),
                    "num_queries": s.get("num_queries", 0),
                    "wall_time_ms": float(s.get("wall_time_ms", 0.0)),
                    "wall_qps": float(s.get("wall_throughput_qps", 0.0)),
                    "total_ms": float(s.get("total_ms", 0.0)),
                    "avg_emb_ms": float(s.get("avg_embedding_ms", 0.0)),
                    "avg_ret_ms": float(s.get("avg_retrieval_ms", 0.0)),
                    "avg_gen_ms": float(s.get("avg_generation_ms", 0.0)),
                }
            )

    table_md = make_table(rows)
    (out_dir / "comparison_table.md").write_text(table_md, encoding="utf-8")
    (out_dir / "comparison_rows.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nComparison done.")
    print(table_md)
    print(f"\nSaved: {out_dir}")


if __name__ == "__main__":
    main()
