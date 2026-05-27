#!/usr/bin/env python3
"""
Post-process BEIR queries for the async pipeline.

BEIR queries are real expert-crafted queries. This script:
  1. Loads queries from a BEIR queries JSONL file (from corpus_builder.py).
  2. Tokenizes them with the embedding tokenizer.
  3. Writes a queries.jsonl with token lengths.

Usage:
    python generate_queries.py \
      --queries-file ./data/beir_nfcorpus/queries_beir.jsonl \
      --output ./data/beir_nfcorpus/queries.jsonl \
      --tokenizer-model sentence-transformers/all-MiniLM-L6-v2
"""

import argparse
import json
import random
import statistics
from pathlib import Path

from tqdm import tqdm
from transformers import AutoTokenizer


def load_queries(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-process BEIR queries: add token lengths."
    )
    parser.add_argument(
        "--queries-file",
        type=str,
        required=True,
        help="Path to queries_beir.jsonl (from corpus_builder.py).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for processed queries (e.g. queries.jsonl).",
    )
    parser.add_argument(
        "--tokenizer-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Tokenizer for token-length computation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)

    queries_path = Path(args.queries_file).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    print(f"Loading queries from {queries_path}...")
    queries = load_queries(queries_path)
    print(f"  Loaded {len(queries)} queries")

    print(f"Loading tokenizer: {args.tokenizer_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model, use_fast=True)

    print("Tokenizing queries...")
    token_lengths = []
    for q in tqdm(queries, desc="Tokenizing"):
        text = q.get("question") or q.get("text") or q.get("query") or ""
        tokens = tokenizer.encode(text, add_special_tokens=True)
        q["token_length"] = len(tokens)
        token_lengths.append(len(tokens))

    rng.shuffle(queries)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for q in queries:
            record = {
                "id": q.get("id", ""),
                "question": q.get("question") or q.get("text") or q.get("query", ""),
                "token_length": q["token_length"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    lens = token_lengths
    print(f"\nDone. Wrote {len(queries)} queries to {output_path}")
    print(f"  Token lengths: min={min(lens)}, max={max(lens)}, "
          f"avg={statistics.mean(lens):.1f}, median={statistics.median(lens):.0f}")


if __name__ == "__main__":
    main()
