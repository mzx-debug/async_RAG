# Async RAG Pipeline V1 — Resource-Constrained Edition

This is a fork of V0, re-targeted for **resource-constrained devices** (laptops, embedded inference boxes, edge GPUs with 4–16 GB VRAM).

The core research question: when the entire tech stack is downsized to match real constrained hardware, does the async_bucket scheduler still outperform static baselines?

## What's different from V0

| Component | V0 (server) | V1 (constrained) |
|-----------|-------------|-----------------|
| Generation model | Llama-3.1-8B-Instruct | Qwen2.5-3B-Instruct |
| Embedding model | e5-large-v2 (1024-dim) | all-MiniLM-L6-v2 (384-dim) |
| Corpus | 8.8M passages (Arrow) | BEIR benchmarks (3.6K–8.7K real docs) |
| FAISS index | IVF4096 (~34 GB) | Flat (~1–2 MB) |
| Vector dimension | 1024 | 384 |
| Default nprobe | 128 | 1 (Flat index) |
| Default batch size | 64 | 32 |
| Default vLLM util | 0.3 (simulated) | 0.6 (real constrained) |

The **scheduling logic itself (`async_rag_pipeline.py`) is unchanged** — only the data, models, and CLI defaults differ.

## Files

- `async_rag_pipeline.py` — main pipeline (copied verbatim from V0)
- `corpus_builder.py` — downloads a BEIR dataset (corpus + queries + qrels)
- `build_index.py` — builds the FAISS index with MiniLM embeddings
- `generate_queries.py` — post-processes BEIR queries (adds bucket hints + token lengths)
- `run_comparison.py` — serial vs async_plain vs async_bucket
- `run_ablation.py` — named ablation variants
- `run_generation_target_eval.py` — generation_target_v1 vs baseline
- `data/` — corpus and queries (populate with `corpus_builder.py`)
- `indexes/` — FAISS indexes (build with `build_index.py`)
- `docs/` — experiment guide and parameter reference

## BEIR Datasets Available

| Dataset | Docs | Queries | Domain | Index size | Recommended for |
|---------|------|---------|--------|-----------|---------------|
| `nfcorpus` | ~3,600 | ~323 | Biomedical (diet/health) | ~0.7 MB | <= 6 GB VRAM |
| `scifact` | ~5,180 | ~1,109 | Scientific claims | ~1.0 MB | 6–12 GB VRAM |
| `arguana` | ~8,700 | ~1,409 | Arguments | ~1.7 MB | 12+ GB VRAM |
| `fiqa` | ~56,000 | ~6,600 | Finance QA | ~11 MB | Larger devices |
| `scidocs` | ~25,000 | ~1,000 | Scientific papers | ~5 MB | Larger devices |

All datasets have **real queries written by experts**, **real relevance judgments (qrels)**, and **real distractor documents** — unlike randomly sampled corpora which can make retrieval trivially easy.

## Setup

```bash
cd /path/to/async_rag_pipeline_v1
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

**Additional vLLM installation** (match your CUDA version):
```bash
pip install vllm  # or: pip install vllm --index-url https://wheels.mycuda.example.com
```

## Step 1: Download BEIR corpus

```bash
# Tiny corpus (recommended for <= 6GB GPU)
python corpus_builder.py --dataset nfcorpus --output ./data/beir_nfcorpus

# Small corpus (6–12GB GPU)
python corpus_builder.py --dataset scifact --output ./data/beir_scifact
```

Downloads corpus + queries + qrels from HuggingFace BeIR benchmark. Output: `data/beir_*/corpus.jsonl`, `queries_beir.jsonl`.

## Step 2: Build index

```bash
python build_index.py \
  --corpus-path ./data/beir_nfcorpus/corpus.jsonl \
  --output-dir ./indexes/beir_nfcorpus \
  --model-path sentence-transformers/all-MiniLM-L6-v2 \
  --batch-size 256 \
  --max-length 384 \
  --pooling-method mean \
  --use-fp16 \
  --faiss-type Flat \
  --device cuda
```

Output: `indexes/beir_nfcorpus/faiss.index` (~0.7 MB)

## Step 3: Post-process queries

BEIR queries are real expert queries, but lack bucket hints. This step adds them:

```bash
python generate_queries.py \
  --queries-file ./data/beir_nfcorpus/queries_beir.jsonl \
  --output ./data/beir_nfcorpus/queries.jsonl \
  --tokenizer-model sentence-transformers/all-MiniLM-L6-v2 \
  --auto-threshold
```

`--auto-threshold` computes short/mid/long boundaries from the actual token-length distribution.

## Step 4: Run comparison

```bash
python run_comparison.py \
  --workdir . \
  --index-path ./indexes/beir_nfcorpus/faiss.index \
  --corpus-path ./data/beir_nfcorpus/corpus.jsonl \
  --generator-model Qwen/Qwen2.5-3B-Instruct \
  --queries-file ./data/beir_nfcorpus/queries.jsonl \
  --sample-queries 256 \
  --b 32 --xE 1 --xR 0 \
  --nprobe 1 --topk 1 \
  --gpu-memory-utilization 0.6 \
  --output-dir ./output/comparison_nfcorpus
```

## One-shot build + run

```bash
# Linux / macOS
DATASET=nfcorpus bash build_and_run.sh

# Or with environment variables:
DATASET=scifact GEN_MODEL=Qwen/Qwen2.5-3B-Instruct GPU_UTIL=0.6 bash build_and_run.sh
```

Or manually:
```bash
python corpus_builder.py --dataset nfcorpus --output ./data/beir_nfcorpus
python build_index.py --corpus-path ./data/beir_nfcorpus/corpus.jsonl \
  --output-dir ./indexes/beir_nfcorpus \
  --model-path sentence-transformers/all-MiniLM-L6-v2 \
  --batch-size 256 --max-length 384 \
  --pooling-method mean --use-fp16 --faiss-type Flat --device cuda
python generate_queries.py --queries-file ./data/beir_nfcorpus/queries_beir.jsonl \
  --output ./data/beir_nfcorpus/queries.jsonl \
  --tokenizer-model sentence-transformers/all-MiniLM-L6-v2 --auto-threshold
python run_comparison.py --workdir . \
  --index-path ./indexes/beir_nfcorpus/faiss.index \
  --corpus-path ./data/beir_nfcorpus/corpus.jsonl \
  --generator-model Qwen/Qwen2.5-3B-Instruct \
  --b 32 --nprobe 1 --gpu-memory-utilization 0.6 \
  --output-dir ./output/comparison_nfcorpus
```

## Expected bottleneck distribution

```
V0 (server):
  Embedding:   ~2ms  ( 1.1%)
  Retrieval:  ~21ms  (11.3%)
  Generation: ~163ms (87.6%)  ← dominated by generation

V1 (constrained, expected):
  Embedding:   ~1ms  ( 5-10%)
  Retrieval:   ~3ms  (15-25%)  ← retrieval占比上升, xR=1开始可行
  Generation:  ~15ms (65-80%)  ← generation不再是绝对主导
```

The **key research question** is: does async_bucket's scheduling advantage survive when all three stages are faster and more balanced?

## Key metrics to compare

```python
wall_throughput_qps   # primary: real throughput
wall_time_ms          # wall-clock latency
avg_embedding_ms
avg_retrieval_ms
avg_generation_ms
action_counts          # xE/xR distribution — does scheduler use xR=1 more?
bucket_counts          # short/mid/long dispatch counts
```

## Choosing a generation model for your GPU

| GPU VRAM | Recommended model | vLLM util |
|----------|-----------------|------------|
| 4–6 GB | Qwen2.5-1.5B-Instruct | 0.7–0.8 |
| 6–8 GB | Qwen2.5-3B-Instruct | 0.6–0.7 |
| 8–12 GB | Qwen2.5-3B-Instruct | 0.8–0.9 |
| 12–16 GB | Qwen2.5-7B-Instruct | 0.5–0.7 |

Override with `--generator-model` on any run script:
```bash
GEN_MODEL=Qwen/Qwen2.5-1.5B-Instruct bash build_and_run.sh
```
