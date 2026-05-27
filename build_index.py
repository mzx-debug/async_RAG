#!/usr/bin/env python3
"""
FAISS index builder for resource-constrained V1 experiments.

Changes vs V0 build_index.py:
  - Default embedding model: sentence-transformers/all-MiniLM-L6-v2 (384-dim, ~80MB)
  - Default FAISS type: Flat  (no IVF clustering needed for ~10K passages)
  - Default max-length: 384   (matches MiniLM's training context)
  - No --faiss-gpu flag      (Flat index is tiny, CPU is fine)

Build index:
    python ./build_index.py \
      --corpus-path ./data/corpus_small.jsonl \
      --output-dir ./indexes/flat \
      --model-path sentence-transformers/all-MiniLM-L6-v2 \
      --batch-size 256 \
      --max-length 384 \
      --pooling-method mean \
      --use-fp16 \
      --faiss-type Flat \
      --device cuda
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import faiss
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def load_corpus(corpus_path: str) -> List[Dict[str, Any]]:
    """Load corpus from a local file (JSONL/JSON/Arrow/Parquet) or a HuggingFace dataset id."""
    path = Path(os.path.expandvars(os.path.expanduser(corpus_path)))
    if path.exists():
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            records = []
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            return records
        if suffix == ".json":
            content = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(content, list):
                return content
            if isinstance(content, dict) and "data" in content:
                return content["data"]
            raise ValueError("JSON corpus must be a list or dict with 'data' key.")
        if suffix in (".arrow", ".parquet"):
            from datasets import load_from_disk
            return list(load_from_disk(str(path)))
        if path.is_dir():
            from datasets import load_from_disk
            return list(load_from_disk(str(path)))

    # Fallback: treat as HuggingFace dataset id
    from datasets import load_dataset
    dataset = load_dataset(corpus_path)
    split = "train" if "train" in dataset else next(iter(dataset.keys()))
    return list(dataset[split])


def extract_doc_text(doc: Dict[str, Any]) -> str:
    """Extract text from a corpus document."""
    if "contents" in doc and doc["contents"] is not None:
        return str(doc["contents"])
    if "text" in doc and doc["text"] is not None:
        if "title" in doc and doc["title"]:
            return f"{doc['title']}\n{doc['text']}"
        return str(doc["text"])
    if "title" in doc and doc["title"] is not None:
        return str(doc["title"])
    return json.dumps(doc, ensure_ascii=False)


def pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor, method: str) -> torch.Tensor:
    if method == "mean":
        masked = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return masked.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    if method == "cls":
        return last_hidden_state[:, 0]
    raise ValueError(f"Unsupported pooling method: {method}")


@torch.inference_mode()
def encode_corpus(
    corpus_texts: List[str],
    model_path: str,
    batch_size: int,
    max_length: int,
    pooling_method: str,
    use_fp16: bool,
    device: torch.device,
) -> np.ndarray:
    """Encode all corpus texts into embeddings."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device)
    if device.type == "cuda" and use_fp16:
        model = model.half()
    model.eval()

    all_embeddings = []
    for start in tqdm(range(0, len(corpus_texts), batch_size), desc="Encoding corpus"):
        batch = corpus_texts[start : start + batch_size]
        inputs = tokenizer(
            batch,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs, return_dict=True)
        emb = pooling(outputs.last_hidden_state, inputs["attention_mask"], pooling_method)
        emb = torch.nn.functional.normalize(emb, dim=-1)
        all_embeddings.append(emb.cpu().numpy().astype(np.float32))

    return np.concatenate(all_embeddings, axis=0)


def build_faiss_index(embeddings: np.ndarray, faiss_type: str, use_gpu: bool) -> faiss.Index:
    """Build a FAISS index from embeddings."""
    dim = embeddings.shape[1]
    index = faiss.index_factory(dim, faiss_type, faiss.METRIC_INNER_PRODUCT)

    if use_gpu:
        co = faiss.GpuMultipleClonerOptions()
        co.useFloat16 = True
        co.shard = True
        index = faiss.index_cpu_to_all_gpus(index, co)

    if not index.is_trained:
        print(f"Training index ({faiss_type}) on {embeddings.shape[0]} vectors...")
        index.train(embeddings)

    print("Adding vectors to index...")
    index.add(embeddings)

    if use_gpu:
        index = faiss.index_gpu_to_cpu(index)

    return index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a FAISS index from a corpus (V1: resource-constrained defaults).")
    parser.add_argument("--corpus-path", type=str, required=True,
                        help="Path to corpus file (JSONL/JSON) or HuggingFace dataset id.")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save the index file.")
    parser.add_argument("--model-path", type=str,
                        default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Embedding model path or HuggingFace id.")
    parser.add_argument("--batch-size", type=int, default=256, help="Encoding batch size.")
    parser.add_argument("--max-length", type=int, default=384,
                        help="Max token length for encoding (default: 384, matches MiniLM).")
    parser.add_argument("--pooling-method", type=str, default="mean",
                        choices=["mean", "cls"], help="Pooling method.")
    parser.add_argument("--use-fp16", action="store_true", help="Use FP16 for encoding.")
    parser.add_argument("--faiss-type", type=str, default="Flat",
                        help="FAISS index type (default: Flat; use IVF for larger corpora).")
    parser.add_argument("--faiss-gpu", action="store_true", help="Use GPU for FAISS index building.")
    parser.add_argument("--device", type=str, default=None,
                        help="Encoding device (cuda/cpu). Auto-detected if omitted.")
    parser.add_argument("--save-embeddings", action="store_true", help="Also save raw embeddings as .npy file.")
    args = parser.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available.")

    # Load corpus
    print("Loading corpus...")
    corpus = load_corpus(args.corpus_path)
    print(f"Corpus size: {len(corpus)}")

    corpus_texts = [extract_doc_text(doc) for doc in corpus]

    # Encode
    t0 = time.perf_counter()
    embeddings = encode_corpus(
        corpus_texts=corpus_texts,
        model_path=args.model_path,
        batch_size=args.batch_size,
        max_length=args.max_length,
        pooling_method=args.pooling_method,
        use_fp16=args.use_fp16,
        device=device,
    )
    encode_time = time.perf_counter() - t0
    print(f"Encoding done: {embeddings.shape}, took {encode_time:.1f}s")
    print(f"  Embedding memory: {embeddings.nbytes / 1024 / 1024:.1f} MB")

    # Build index
    t0 = time.perf_counter()
    index = build_faiss_index(embeddings, args.faiss_type, args.faiss_gpu)
    index_time = time.perf_counter() - t0
    print(f"Index built: {index.ntotal} vectors, took {index_time:.1f}s")

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    index_path = output_dir / "faiss.index"
    faiss.write_index(index, str(index_path))
    print(f"Index saved to: {index_path}")
    print(f"  Index file size: {index_path.stat().st_size / 1024 / 1024:.2f} MB")

    if args.save_embeddings:
        emb_path = output_dir / "embeddings.npy"
        np.save(str(emb_path), embeddings)
        print(f"Embeddings saved to: {emb_path}")

    print(f"\nDone.")
    print(f"  Corpus:     {len(corpus)} passages")
    print(f"  Embedding:  {args.model_path}  dim={embeddings.shape[1]}")
    print(f"  Index:      {args.faiss_type}  {index.ntotal} vectors")


if __name__ == "__main__":
    main()
