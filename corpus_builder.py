#!/usr/bin/env python3
"""
Build corpus and queries for resource-constrained RAG experiments
from public BEIR benchmark datasets.

Available datasets (all have corpus + queries + qrels):
  beir/scifact    ~5K scientific articles, ~1K claims       (biomedical)
  beir/nfcorpus   ~3.6K papers,       ~323 queries           (biomedical)
  beir/trec-covid ~171K papers,       ~50 queries            (COVID-19)
  beir/arguana    ~8.7K arguments,    ~1,409 queries         (argument retrieval)

For tiny devices (<= 6GB VRAM): beir/nfcorpus (smallest, ~3.6K docs)
For mid-range (6-12GB VRAM):   beir/scifact (~5K docs)
For more capable devices:        beir/arguana (~8.7K docs)

Usage:
    python corpus_builder.py --dataset beir/nfcorpus --output ./data/beir_nfcorpus
"""

import argparse
import json
import time
from pathlib import Path

from datasets import load_dataset


# Field mapping: BEIR uses "_id", "title", "text"
# Pipeline expects "contents" or {"title": ..., "text": ...}
def beir_to_pipeline(doc: dict) -> dict:
    """Convert BEIR document format to pipeline format."""
    doc_id = doc.get("_id") or doc.get("id", "")
    title = doc.get("title") or ""
    text = doc.get("text") or doc.get("contents") or ""
    # Prefer title+text as contents for better embedding
    if title:
        contents = f"{title}\n{text}"
    else:
        contents = text
    return {"id": doc_id, "contents": contents.strip()}


def beir_query_to_pipeline(q: dict) -> dict:
    """Convert BEIR query format to pipeline format."""
    q_id = q.get("_id") or q.get("id", "")
    # BEIR queries often store the query text in "text" or "title"
    text = q.get("text") or q.get("title") or q.get("query") or ""
    return {"id": q_id, "question": text.strip()}


def download_dataset(dataset_name: str, output_dir: Path) -> dict[str, list]:
    """
    Download a BEIR dataset and convert to pipeline format.

    Returns {"corpus": [...], "queries": [...]} in pipeline format.
    """
    t0 = time.perf_counter()
    # dataset_name may be "beir/nfcorpus" or just "nfcorpus"
    # HuggingFace dataset id is always "beir/{name}" with lowercase 'b'
    beir_name = dataset_name.replace("BeIR/", "").replace("beir/", "")
    hf_id = f"beir/{beir_name}"
    print(f"Loading BEIR dataset: {dataset_name}  (HuggingFace: {hf_id})")

    corpus_ds = load_dataset(hf_id, name="corpus", split="corpus")
    queries_ds = load_dataset(hf_id, name="queries", split="queries")

    corpus = [beir_to_pipeline(dict(d)) for d in corpus_ds]
    queries = [beir_query_to_pipeline(dict(d)) for d in queries_ds]

    elapsed = time.perf_counter() - t0
    print(f"  Corpus:  {len(corpus):,} documents")
    print(f"  Queries: {len(queries):,} queries")
    print(f"  Download time: {elapsed:.1f}s")

    # Save corpus
    corpus_path = output_dir / "corpus.jsonl"
    with corpus_path.open("w", encoding="utf-8") as f:
        for doc in corpus:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"  Saved corpus to: {corpus_path}  ({corpus_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # Save queries
    queries_path = output_dir / "queries_beir.jsonl"
    with queries_path.open("w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"  Saved queries to: {queries_path}")

    # Also try to load qrels if available (for reference / evaluation)
    try:
        qrels_ds = load_dataset(hf_id, name="qrels", split="train")
        qrels_path = output_dir / "qrels.jsonl"
        with qrels_path.open("w", encoding="utf-8") as f:
            for row in qrels_ds:
                f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
        print(f"  Saved qrels to: {qrels_path}")
    except Exception:
        pass  # qrels optional

    return {"corpus": corpus, "queries": queries}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a BEIR dataset and save as corpus.jsonl + queries_beir.jsonl for V1 pipeline."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="nfcorpus",
        choices=["nfcorpus", "scifact", "trec-covid", "arguana", "fiqa", "scidocs"],
        help="BEIR dataset name (without 'beir/' prefix). Default: nfcorpus",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data",
        help="Output directory. Will contain corpus.jsonl, queries_beir.jsonl, and optionally qrels.jsonl.",
    )
    args = parser.parse_args()

    dataset_name = args.dataset
    if not dataset_name.startswith("beir/"):
        dataset_name = f"beir/{dataset_name}"

    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== BEIR Dataset Builder ===")
    print(f"Dataset:  {dataset_name}")
    print(f"Output:   {output_dir}")
    print()

    result = download_dataset(dataset_name, output_dir)

    # Print corpus stats
    total_chars = sum(len(d["contents"]) for d in result["corpus"])
    avg_len = total_chars / len(result["corpus"]) if result["corpus"] else 0
    print()
    print(f"Done. {len(result['corpus']):,} documents, {len(result['queries']):,} queries.")
    print(f"  Avg doc length: {avg_len:.0f} chars")


if __name__ == "__main__":
    main()
