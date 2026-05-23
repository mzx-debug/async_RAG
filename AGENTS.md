# Repository Guidelines — Async RAG Pipeline V1 (Resource-Constrained)

## Project Scope

V1 is a **minimal fork of V0** for resource-constrained hardware. The scheduling and pipeline logic in `async_rag_pipeline.py` must remain **identical to V0** — only data, model defaults, and CLI defaults change.

## What must stay identical to V0

- `async_rag_pipeline.py` — copy verbatim from V0, no modifications
- Scheduling logic (bucket dispatch, EMA feedback, action selection)
- Output JSON schema
- Argument names and semantics for scheduler-related flags

## What may differ from V0

- Default values for `--generator-model`, `--model-path`, `--b`, `--nprobe`, etc.
- Corpus source and size
- FAISS index type
- Query set generation scripts
- Documentation and guides

## Key differences from V0

| | V0 | V1 |
|-|----|----|
| Generation | Llama-3.1-8B-Instruct | Qwen2.5-3B-Instruct |
| Embedding | intfloat/e5-large-v2 (1024-dim) | all-MiniLM-L6-v2 (384-dim) |
| Corpus | 8.8M passages | 10,000 passages |
| Index | IVF4096,Flat (~34 GB) | Flat (~15 MB) |
| Target hardware | Server GPU (24–80 GB) | Edge GPU (4–16 GB) |

## Adding new code

Follow V0 conventions:
- 4-space indentation, type hints, docstrings
- `snake_case` for functions, `PascalCase` for classes
- Keep scripts importable
- New scripts belong in the root (not `docs/` or `data/`)

## Build, Test, and Development Commands

```bash
# Setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Build corpus (one-time)
python corpus_builder.py --num-passages 10000 --output ./data/corpus_small.jsonl

# Build index
python build_index.py \
  --corpus-path ./data/corpus_small.jsonl \
  --output-dir ./indexes/flat \
  --model-path sentence-transformers/all-MiniLM-L6-v2 \
  --batch-size 256 --max-length 384 \
  --pooling-method mean --use-fp16 --faiss-type Flat --device cuda

# Generate queries
python generate_queries.py

# Run comparison
python run_comparison.py --workdir . \
  --index-path ./indexes/flat/faiss.index \
  --corpus-path ./data/corpus_small.jsonl \
  --generator-model Qwen/Qwen2.5-3B-Instruct \
  --b 32 --nprobe 1 --topk 1 \
  --gpu-memory-utilization 0.6 \
  --output-dir ./output/comparison
```

## Testing

Run the smallest possible path first:

```bash
# Just build the corpus (no GPU needed)
python corpus_builder.py --num-passages 1000 --output ./data/corpus_test.jsonl

# Build a tiny index
python build_index.py --corpus-path ./data/corpus_test.jsonl \
  --output-dir ./indexes/test --model-path sentence-transformers/all-MiniLM-L6-v2 \
  --max-length 384 --faiss-type Flat --device cpu

# One serial run with 16 queries
python async_rag_pipeline.py \
  --pipeline-mode serial \
  --index-path ./indexes/test/faiss.index \
  --corpus-path ./data/corpus_test.jsonl \
  --generator-model Qwen/Qwen2.5-3B-Instruct \
  --sample-queries 16 --b 8 --nprobe 1 --topk 1 \
  --output-json ./output/test_serial.json
```

Check the output JSON for schema stability and sensible values before running full experiments.
