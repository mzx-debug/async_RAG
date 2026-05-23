#!/usr/bin/env python3
"""
Post-process BEIR queries for the V1 async pipeline.

BEIR queries are real expert-crafted queries. This script:
  1. Loads queries from a BEIR queries JSONL file (from corpus_builder.py).
  2. Tokenizes them with the embedding tokenizer.
  3. Assigns bucket_hint (short/mid/long) based on token length.
  4. Writes a queries.jsonl in the same format as V0 queries_generated.jsonl.

The short/mid/long thresholds are derived from the actual distribution of the dataset,
so bucket boundaries reflect the real query-length profile of that domain.

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


# Thresholds: these define short / mid / long in terms of token count.
# They are matched to the real distribution of each dataset (printed after analysis).
# For nfcorpus (avg ~60-80 tokens), scifact (~80-120), arguana (~20-40).
DEFAULT_SHORT_THRESHOLD = 48
DEFAULT_LONG_THRESHOLD = 96


def load_queries(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_thresholds(token_lengths: list[int]) -> tuple[int, int]:
    """
    Derive short/long thresholds from the actual distribution.
    short: below 33rd percentile
    long:  above 67th percentile
    mid:   everything else
    """
    short_thr = int(statistics.quantiles(token_lengths, n=3)[0])
    long_thr = int(statistics.quantiles(token_lengths, n=3)[2])
    return short_thr, long_thr


def assign_bucket(token_len: int, short_thr: int, long_thr: int) -> str:
    if token_len <= short_thr:
        return "short"
    elif token_len >= long_thr:
        return "long"
    else:
        return "mid"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-process BEIR queries: add token lengths and bucket hints."
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
        "--short-threshold",
        type=int,
        default=DEFAULT_SHORT_THRESHOLD,
        help="Token length <= this is 'short'.",
    )
    parser.add_argument(
        "--long-threshold",
        type=int,
        default=DEFAULT_LONG_THRESHOLD,
        help="Token length >= this is 'long'.",
    )
    parser.add_argument(
        "--auto-threshold",
        action="store_true",
        help="Auto-compute short/long thresholds from query distribution.",
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

    # Step 1: tokenize all queries to get lengths
    print("Tokenizing queries...")
    token_lengths = []
    for q in tqdm(queries, desc="Tokenizing"):
        text = q.get("question") or q.get("text") or q.get("query") or ""
        tokens = tokenizer.encode(text, add_special_tokens=True)
        q["token_length"] = len(tokens)
        token_lengths.append(len(tokens))

    # Step 2: compute thresholds
    if args.auto_threshold:
        short_thr, long_thr = compute_thresholds(token_lengths)
        print(f"  Auto thresholds: short<={short_thr}, long>={long_thr}")
    else:
        short_thr = args.short_threshold
        long_thr = args.long_threshold
        print(f"  Fixed thresholds: short<={short_thr}, long>={long_thr}")

    # Step 3: assign buckets and finalise
    for q in queries:
        q["bucket_hint"] = assign_bucket(q["token_length"], short_thr, long_thr)

    # Shuffle to avoid ordering bias
    rng.shuffle(queries)

    # Step 4: write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for q in queries:
            record = {
                "id": q.get("id", ""),
                "question": q.get("question") or q.get("text") or q.get("query", ""),
                "bucket_hint": q["bucket_hint"],
                "token_length": q["token_length"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Report distribution
    lens = token_lengths
    bucket_counts = {"short": 0, "mid": 0, "long": 0}
    for q in queries:
        bucket_counts[q["bucket_hint"]] += 1

    print(f"\nDone. Wrote {len(queries)} queries to {output_path}")
    print(f"  Token lengths: min={min(lens)}, max={max(lens)}, "
          f"avg={statistics.mean(lens):.1f}, median={statistics.median(lens):.0f}")
    print(f"  Buckets: {bucket_counts}")


if __name__ == "__main__":
    main()
