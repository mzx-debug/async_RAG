#!/usr/bin/env python3

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


def run_one(name: str, script: Path, base_args: List[str], extra_args: List[str], out_dir: Path) -> Dict[str, Any]:
    out_path = out_dir / f"summary_{name}.json"
    cmd = [sys.executable, str(script)] + base_args + extra_args + ["--output-json", str(out_path)]
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Ablation {name} failed with code {proc.returncode}")
    if not out_path.exists():
        raise FileNotFoundError(f"Missing output: {out_path}")
    return json.loads(out_path.read_text(encoding="utf-8"))


def make_table(rows: List[Dict[str, Any]]) -> str:
    header = [
        "name",
        "mode",
        "wall_time_ms",
        "wall_qps",
        "avg_emb_ms",
        "avg_ret_ms",
        "avg_gen_ms",
        "dispatch_ms",
        "feedback_ms",
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["name"]),
                    str(row["mode"]),
                    f"{row['wall_time_ms']:.2f}",
                    f"{row['wall_qps']:.4f}",
                    f"{row['avg_emb_ms']:.4f}",
                    f"{row['avg_ret_ms']:.4f}",
                    f"{row['avg_gen_ms']:.4f}",
                    f"{row['dispatch_ms']:.4f}",
                    f"{row['feedback_ms']:.4f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ablation variants for async_rag_pipeline.py (V1: resource-constrained defaults).")
    parser.add_argument("--workdir", type=str, default=".", help="Directory containing async_rag_pipeline.py")
    parser.add_argument("--index-path", type=str,
                        default="./indexes/flat/faiss.index")
    parser.add_argument("--corpus-path", type=str,
                        default="./data/corpus_small.jsonl")
    parser.add_argument("--generator-model", type=str,
                        default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--embedding-model", type=str,
                        default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Embedding model path or HuggingFace id. Must match the index.")
    parser.add_argument("--embedding-max-length", type=int, default=384)
    parser.add_argument("--pooling-method", type=str, default="mean",
                        choices=["mean", "cls", "pooler"])
    parser.add_argument("--embedding-use-fp16", action="store_true", default=True)
    parser.add_argument("--queries-file", type=str, default=None)
    parser.add_argument("--sample-queries", type=int, default=256)
    parser.add_argument("--b", type=int, default=32)
    parser.add_argument("--xE", type=int, default=1)
    parser.add_argument("--xR", type=int, default=0)
    parser.add_argument("--nprobe", type=int, default=1)
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--gpu-id", type=str, default="0")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument("--output-dir", type=str, default="./output/ablation")
    parser.add_argument("--scheduler-ema-alpha", type=float, default=0.25)
    args = parser.parse_args()

    workdir = Path(args.workdir).expanduser().resolve()
    script = workdir / "async_rag_pipeline.py"
    if not script.exists():
        raise FileNotFoundError(script)

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    common_args: List[str] = [
        "--index-path",             str(args.index_path),
        "--corpus-path",            str(args.corpus_path),
        "--generator-model",        str(args.generator_model),
        "--b",                      str(args.b),
        "--xE",                     str(args.xE),
        "--xR",                     str(args.xR),
        "--nprobe",                 str(args.nprobe),
        "--topk",                   str(args.topk),
        "--sample-queries",         str(args.sample_queries),
        "--gpu-id",                 str(args.gpu_id),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--scheduler-ema-alpha",    str(args.scheduler_ema_alpha),
        "--embedding-model",         str(args.embedding_model),
        "--embedding-max-length",  str(args.embedding_max_length),
        "--pooling-method",         str(args.pooling_method),
        "--embedding-use-fp16",
    ]
    if args.queries_file:
        common_args += ["--queries-file", str(args.queries_file)]

    variants: List[Dict[str, Any]] = [
        {"name": "plain_b16",  "args": ["--pipeline-mode", "async_plain", "--b", "16"]},
        {"name": "plain_b32",  "args": ["--pipeline-mode", "async_plain", "--b", "32"]},
        {"name": "v2_fixed_batch_fixed_action", "args": ["--pipeline-mode", "async_v2", "--ablate-online-batch", "--ablate-online-action"]},
        {"name": "v2_online_batch_fixed_action", "args": ["--pipeline-mode", "async_v2", "--ablate-online-action"]},
        {"name": "v2_online_batch_online_action", "args": ["--pipeline-mode", "async_v2"]},
        {"name": "v2_online_batch_online_action_no_chunk", "args": ["--pipeline-mode", "async_v2", "--ablate-chunking"]},
    ]

    rows: List[Dict[str, Any]] = []
    for variant in variants:
        summary = run_one(variant["name"], script, common_args, variant["args"], out_dir)
        timing = summary.get("timing_breakdown", {})
        rows.append(
            {
                "name": variant["name"],
                "mode": summary.get("scheduler", {}).get("mode", ""),
                "wall_time_ms": float(summary.get("wall_time_ms", 0.0)),
                "wall_qps": float(summary.get("wall_throughput_qps", 0.0)),
                "avg_emb_ms": float(summary.get("avg_embedding_ms", 0.0)),
                "avg_ret_ms": float(summary.get("avg_retrieval_ms", 0.0)),
                "avg_gen_ms": float(summary.get("avg_generation_ms", 0.0)),
                "dispatch_ms": float(timing.get("scheduler_dispatch_ms_total", 0.0)),
                "feedback_ms": float(timing.get("scheduler_feedback_ms_total", 0.0)),
            }
        )

    table_md = make_table(rows)
    (out_dir / "ablation_table.md").write_text(table_md, encoding="utf-8")
    (out_dir / "ablation_rows.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nAblation done.")
    print(table_md)
    print(f"\nSaved: {out_dir}")


if __name__ == "__main__":
    main()
