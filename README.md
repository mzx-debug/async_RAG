# Async RAG Pipeline — Resource-Constrained Edition

Async RAG with **EMA-calibrated online dispatch** on resource-constrained devices (laptops, edge GPUs with 4–16 GB VRAM).

The core idea: embedding, retrieval, and generation are scheduled in a three-stage async pipeline. A `GreedyScheduler` uses Exponential Moving Average (EMA) cost models — continuously calibrated from live execution feedback — to make dispatch decisions online.

## Table of contents

- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Data preparation](#step-1--download-corpus)
- [Index building](#step-2--build-faiss-index)
- [Query preprocessing](#step-3--preprocess-queries)
- [Running the pipeline](#step-4--run-the-pipeline)
- [Three execution modes](#three-execution-modes)
- [EMA cost model](#ema-cost-model)
- [CLI reference](#cli-reference)
- [Output format](#output-format)
- [BEIR datasets](#beir-datasets)
- [GPU memory and device selection](#gpu-memory-and-device-selection)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

| Component | Version / Note |
|-----------|---------------|
| Python | >= 3.10 |
| CUDA | 12.x (required by vLLM and PyTorch) |
| GPU | 4–16 GB VRAM (tested on A10G 24 GB, RTX 3090 24 GB, T4 16 GB) |
| Disk | ~5 GB for models + ~500 MB for BEIR datasets |
| OS | Linux (tested on Ubuntu 22.04) |

**Python environment must be activated with `source`** (not `conda activate`) to ensure `LD_LIBRARY_PATH` points to the correct CUDA version:

```bash
source /home/cloudteam/Software/conda/bin/activate p702
```

> If you use a different conda environment, make sure `LD_LIBRARY_PATH` includes your CUDA `lib64` directory, e.g. `export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH`.

---

## Setup

```bash
cd /home/cloudteam/rag_mzx/async_RAG
pip install -U pip
pip install -r requirements.txt
```

> **China mainland note:** HuggingFace downloads are slow. Set `HF_ENDPOINT=https://hf-mirror.com` before running any script that downloads models. The `async_rag_pipeline.py` sets this automatically at startup.

---

## Step 1 — Download corpus

```bash
python corpus_builder.py --dataset nfcorpus --output ./data/beir_nfcorpus
```

This downloads corpus, queries, and relevance judgments (qrels) from the BEIR benchmark. It produces:

```
data/beir_nfcorpus/
  corpus.jsonl          # one JSON object per line: {"_id": ..., "text": ...}
  queries_beir.jsonl    # BEIR-format queries: {"id": ..., "text": ...}
  qrels/               # relevance judgments
```

Supported datasets: `nfcorpus`, `scifact`, `arguana`, `fiqa`, `scidocs`.

---

## Step 2 — Build FAISS index

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

Output: `indexes/beir_nfcorpus/faiss.index` (~1 MB for nfcorpus).

- `--faiss-type Flat`: exact search (required for nfcorpus; for large corpora use `IVF` with `--nprobe`)
- `--use-fp16`: reduces memory footprint, no accuracy loss for retrieval
- `--device cuda`: generates embeddings on GPU (faster), then builds index in memory

---

## Step 3 — Preprocess queries

BEIR queries lack token-length metadata needed for the scheduler's cost model. This step adds it:

```bash
python generate_queries.py \
  --queries-file ./data/beir_nfcorpus/queries_beir.jsonl \
  --output ./data/beir_nfcorpus/queries.jsonl \
  --tokenizer-model sentence-transformers/all-MiniLM-L6-v2 \
  --auto-threshold
```

Output: `data/beir_nfcorpus/queries.jsonl` — same queries enriched with `"token_length"` and `"token_length_bucket"`.

---

## Step 4 — Run the pipeline

### One-shot script (recommended)

```bash
# Defaults: nfcorpus dataset, xE=0 xR=0, Qwen2.5-1.5B-Instruct, 256 queries
DATASET=nfcorpus bash build_and_run.sh

# Customize via environment variables
DATASET=scifact \
GEN_MODEL=Qwen/Qwen2.5-1.5B-Instruct \
BATCH=64 \
GPU_UTIL=0.7 \
SAMPLE_QUERIES=512 \
bash build_and_run.sh
```

### Direct pipeline invocation

For single-mode runs (debugging or targeted experiments):

```bash
source /home/cloudteam/Software/conda/bin/activate p702

HF_ENDPOINT=https://hf-mirror.com python async_rag_pipeline.py \
  --xE 0 --xR 0 \
  --b 64 \
  --sample-queries 256 \
  --pipeline-mode async_v2 \
  --index-path ./indexes/beir_nfcorpus/faiss.index \
  --corpus-path ./data/beir_nfcorpus/corpus.jsonl \
  --queries-file ./data/beir_nfcorpus/queries.jsonl \
  --generator-model Qwen/Qwen2.5-1.5B-Instruct \
  --output-json ./output/run.json \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.6 \
  --gpu-id 0
```

---

## Three execution modes

| Mode | Description | Use case |
|------|-------------|---------|
| `serial` | E→R→G sequentially, fixed batches | Baseline |
| `async_plain` | E+R pre-built batches, G runs after all E+R done | Simpler async baseline |
| `async_v2` | Online dispatch, EMA cost model, greedy batch shaping | **Production use** |

`serial` and `async_plain` both pre-build all microbatches before execution. `async_v2` schedules each batch dynamically as the pipeline runs, using the cost model to choose `(batch_size, xE, xR)` that minimizes `wall_time / batch_size`.

### Run comparison across all modes

```bash
python run_comparison.py \
  --workdir . \
  --index-path ./indexes/beir_nfcorpus/faiss.index \
  --corpus-path ./data/beir_nfcorpus/corpus.jsonl \
  --queries-file ./data/beir_nfcorpus/queries.jsonl \
  --generator-model Qwen/Qwen2.5-1.5B-Instruct \
  --sample-queries 256 \
  --b 64 \
  --xE 0 --xR 0 \
  --nprobe 1 --topk 1 \
  --gpu-memory-utilization 0.6 \
  --output-dir ./output/comparison
```

This runs `serial`, `async_plain`, and `async_v2` and prints a Markdown comparison table.

---

## EMA cost model

The scheduler models wall-clock time per microbatch as:

```
wall_time = max(gen_time, emb_ret_time)      # xE=0: E+R parallel with G
wall_time = gen_time + emb_ret_time × f       # xE=1: GPU contention

gen_time[(xE,xR)] = gen_base[(xE,xR)] + gen_per_query[(xE,xR)] × batch_size
emb_ret_time        = er_base + (emb_per_query + ret_per_query) × batch_size
```

The **dispatch score** (lower is better) is:

```
score = wall_time / batch_size     # estimated ms per query
```

The scheduler picks the `(batch_size, xE, xR)` with minimum score.

Warm-start defaults for `(0, 0)` (CPU embed, CPU retrieval):

| Parameter | Value | Source |
|-----------|-------|--------|
| `gen_base` | 1170 ms | measured, vLLM Qwen2.5-1.5B-Instruct (prefill + kernel launch) |
| `gen_per_query` | 28.2 ms/q | measured, linear fit across bs=32/64/256 |
| `emb_per_query` | 1.31 ms/q | measured, CPU sentence-transformers/all-MiniLM-L6-v2 |
| `ret_per_query` | 0.05 ms/q | measured, CPU FAISS Flat search |

> The linear model (`gen = gen_base + gen_per_q × bs`) fits with R²=0.999999 (MAE=3ms). This is correct because LLM prefill and decode both scale linearly with batch size under vLLM's continuous batching.

The candidate batch size list is `[1, 2, 4, 8, 16, 32, 64, 128, 256]`, filtered by GPU memory feasibility per action.

---

## EMA persistence

Save calibrated parameters after the first run to warm-start subsequent runs:

```bash
# First run: calibrate and save
python async_rag_pipeline.py \
  --xE 0 --xR 0 --b 64 --sample-queries 256 \
  --pipeline-mode async_v2 \
  --index-path ./indexes/beir_nfcorpus/faiss.index \
  --corpus-path ./data/beir_nfcorpus/corpus.jsonl \
  --queries-file ./data/beir_nfcorpus/queries.jsonl \
  --generator-model Qwen/Qwen2.5-1.5B-Instruct \
  --output-json ./output/run.json \
  --gpu-memory-utilization 0.6 \
  --ema-params-path ./output/params/ema.json \
  --save-ema-params

# Subsequent runs: load pre-calibrated parameters
python async_rag_pipeline.py \
  ... (same args, minus --save-ema-params) ...
  --ema-params-path ./output/params/ema.json
```

The saved JSON contains: `gen_base_overhead_ema`, `gen_per_query_ema`, `embedding_latency_ema`, `retrieval_latency_ema`, `overlap_factor_ema`, `wall_time_measurements`, `best_batch_size_by_action`.

### Calibrate all (xE, xR) combinations

For accurate scheduling across all device configurations, run the calibration sweep:

```bash
# Full sweep: 4 (xE,xR) combinations × multiple batch sizes
python calibrate_sweep.py

# Custom sweep
python calibrate_sweep.py --action 0 1 --batch-sizes 1 32 128
```

This runs `async_v2` with `--fixed-action` so each `(xE, xR)` is independently profiled.

---

## Auto-tune for new devices

**`auto_tune.py`** is the recommended first step on any new device. It:

1. Auto-detects your GPU model and VRAM
2. Selects an appropriate generator model and GPU utilization
3. Feasibility-tests each `(xE, xR)` combination (catches OOM before wasting time)
4. Runs a calibration sweep (multiple batch sizes) for each feasible action
5. Fits the linear cost model `gen = gen_base + gen_per_q × batch_size`
6. Saves a ready-to-use EMA parameters file

```bash
# Full auto-tune (all 4 actions × 5 batch sizes, ~20 runs)
python auto_tune.py

# Faster: one action, fewer batch sizes
python auto_tune.py --action 0 0 --batch-sizes 16 64 128

# Check plan without running
python auto_tune.py --dry-run
```

Output files:

```
output/auto_tune/ema_params_<model>_<timestamp>.json   # load with --ema-params-path
output/auto_tune/tuning_report_<model>_<timestamp>.txt  # human-readable report
```

The tuning report shows per-action R², the fitted coefficients, and a predicted score table at every batch size. Use the recommended `--ema-params-path` to warm-start subsequent runs.

### Example output

```
CALIBRATION RESULTS

Action (0,0) — CPU embed + CPU retrieval
  Status         OK
  gen_base       1170 ms
  gen_per_q      28.2 ms/q
  emb_per_q      1.31 ms/q
  ret_per_q      0.051 ms/q
  R²             0.999999

  b=256: gen=8389ms, score=32.8ms/q  ← best

RECOMMENDATIONS

  Best overall action: (0,0) with bs=256
  Score: 32.8 ms/q

Next step: use the tuned parameters
  python async_rag_pipeline.py --ema-params-path output/auto_tune/ema_params_...
```

### Device recommendation table

`auto_tune.py` auto-selects model and GPU util based on your VRAM:

| VRAM | Model | GPU util | Notes |
|------|-------|----------|-------|
| < 5 GB | Qwen2.5-1.5B-Instruct | 0.5 | Conservative |
| 5–7 GB | Qwen2.5-1.5B-Instruct | 0.7 | Comfortable |
| 7–10 GB | Qwen2.5-3B-Instruct | 0.6 | Moderate |
| 10–16 GB | Qwen2.5-3B-Instruct | 0.8 | Generous KV |
| > 16 GB | Qwen2.5-7B-Instruct | 0.6 | Large model |

Override with `--model` and `--gpu-util`.

---

## CLI reference

### Required arguments

| Flag | Type | Description |
|------|------|-------------|
| `--index-path` | str | Path to the FAISS `.index` file |
| `--corpus-path` | str | Path to corpus JSONL |
| `--generator-model` | str | HuggingFace model ID or local path for vLLM |
| `--b` | int | Base batch size for query dispatch |
| `--xE` | int | Embedding device: `0`=CPU, `1`=GPU |
| `--xR` | int | Retrieval device: `0`=CPU, `1`=GPU (requires faiss-gpu) |

### Pipeline control

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--pipeline-mode` | str | `async_v2` | `serial` \| `async_plain` \| `async_v2` |
| `--fixed-action` | flag | off | Lock (xE, xR) to CLI values; disables online action selection |
| `--sample-queries` | int | 256 | Number of queries to sample from the query file |
| `--queries-file` | str | auto | Query file path (`.jsonl`); auto-selects from `--dataset-name` if omitted |
| `--output-json` | str | — | Write summary JSON here |

### Embedding model

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--embedding-model` | str | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model ID |
| `--embedding-max-length` | int | 384 | Max token length for embedding |
| `--pooling-method` | str | `mean` | `mean` \| `cls` \| `pooler` |
| `--embedding-use-fp16` | flag | off | Use FP16 for embedding (faster, lower memory) |

### Generation model (vLLM)

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--max-model-len` | int | auto | Max sequence length for vLLM |
| `--max-output-len` | int | 128 | Max generated tokens per query |
| `--temperature` | float | 0.0 | Generation temperature (0 = greedy) |
| `--top-p` | float | 1.0 | Nucleus sampling threshold |
| `--top-k` | int | auto | Top-k sampling (default: no limit) |
| `--gpu-memory-utilization` | float | 0.8 | Fraction of GPU memory for vLLM KV cache |
| `--tensor-parallel-size` | int | 1 | Tensor parallelism degree (multi-GPU) |
| `--vllm-enforce-eager` | flag | off | Disable CUDA graph acceleration (more accurate profiling) |

### Retrieval

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--nprobe` | int | 128 | FAISS IVF search depth (set to 1 for Flat index) |
| `--topk` | int | 1 | Number of documents retrieved per query |

### Scheduler (async_v2 only)

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--initial-batch-size` | int | 32 | Starting batch size for EMA; scheduler adapts from here |
| `--scheduler-ema-alpha` | float | 0.25 | EMA smoothing factor (higher = faster adaptation, more noise) |
| `--backpressure-high` | int | 8 | Max pending items in pipeline queue before blocking dispatch |
| `--gpu-mem-low-threshold-gb` | float | 4.0 | Free GPU memory below this → "high" pressure |
| `--gpu-mem-medium-threshold-gb` | float | 10.0 | Free GPU memory below this → "medium" pressure |
| `--gpu-mem-high-batch-penalty` | float | 50.0 | Score penalty for GPU-heavy actions under high memory pressure |
| `--faiss-index-gb` | float | 2.0 | Estimated FAISS GPU index memory footprint |
| `--generator-model-layers` | int | auto | Generator model layers (for memory estimation; auto-detected) |
| `--generator-model-hidden` | int | auto | Generator hidden dim (for memory estimation; auto-detected) |

### EMA persistence

| Flag | Type | Description |
|------|------|-------------|
| `--ema-params-path` | str | Load/save EMA parameters JSON |
| `--save-ema-params` | flag | Save EMA params to `--ema-params-path` after run |

### Ablation flags

| Flag | Description |
|------|-------------|
| `--ablate-online-batch` | Disable online batch size selection (use fixed `--b`) |
| `--ablate-online-action` | Disable online (xE, xR) action selection |
| `--ablate-chunking` | Disable token-length-based query chunking |

### Logging

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--log-interval` | int | 1 | Log every N batches (set high like 9999 to suppress) |
| `--show-samples` | int | 3 | Number of example (query, answer) pairs to show |
| `--seed` | int | 2026 | Random seed for query sampling |

---

## Output format

Each run produces a JSON summary (at `--output-json`). Key fields:

```json
{
  "wall_time_ms": 9057,
  "wall_throughput_qps": 28.27,
  "avg_embedding_ms": 1.36,
  "avg_retrieval_ms": 0.07,
  "avg_generation_ms": 31.7,
  "total_generated_tokens": 32012,
  "generation_ms_per_token": 0.25,
  "scheduler": {
    "mode": "online_dispatch_ema_v1",
    "action_counts": {"xE0_xR0": 1},
    "max_q_er": 1,
    "max_q_rg": 1
  },
  "ema_params": {
    "gen_base_overhead_ema": {"(0,0)": 1170.0},
    "gen_per_query_ema": {"(0,0)": 28.2}
  },
  "per_batch": [
    {
      "batch_index": 1,
      "batch_size": 256,
      "embedding_sec": 0.340,
      "retrieval_sec": 0.018,
      "generation_sec": 8.092
    }
  ],
  "dispatch_trace": [...],
  "feedback_trace": [...],
  "samples": [...]
}
```

### Key metrics

| Field | Description |
|-------|-------------|
| `wall_time_ms` | Actual wall-clock time in ms |
| `wall_throughput_qps` | Primary metric: queries per second |
| `avg_*_ms` | Average stage time per query |
| `generation_ms_per_token` | Generation throughput |
| `action_counts` | Distribution of (xE, xR) choices — confirms scheduler behavior |
| `max_q_er` / `max_q_rg` | Maximum queue depth — indicates pipeline buffering |
| `ema_params` | Calibrated cost model parameters (persist and reuse) |

---

## BEIR datasets

| Dataset | Docs | Queries | Domain | Index size | Recommended VRAM |
|---------|------|---------|--------|-----------|-----------------|
| `nfcorpus` | 3,633 | 323 | Biomedical (diet/health) | ~1 MB | <= 6 GB |
| `scifact` | 5,183 | 1,109 | Scientific claims | ~1 MB | 6–12 GB |
| `arguana` | 8,674 | 1,409 | Arguments | ~2 MB | 12+ GB |
| `fiqa` | 56,380 | 6,648 | Finance QA | ~11 MB | Large devices |
| `scidocs` | 25,000 | 1,000 | Scientific papers | ~5 MB | Large devices |

All datasets have real expert-written queries, real relevance judgments, and real distractor documents.

---

## GPU memory and device selection

### (xE, xR) device combinations

| (xE, xR) | Embedding | Retrieval | Overlap with G |
|-----------|-----------|-----------|----------------|
| (0, 0) | CPU | CPU | E+R on CPU parallel to G on GPU — **recommended for constrained devices** |
| (1, 0) | GPU | CPU | Embed competes for GPU memory with vLLM — **may be slower than (0,0)** |
| (0, 1) | CPU | GPU | Both parallel to G — good if FAISS GPU search is available |
| (1, 1) | GPU | GPU | Full GPU contention — only for large VRAM (> 16 GB) |

### Generator model selection

| GPU VRAM | Model | `--gpu-memory-utilization` |
|----------|-------|--------------------------|
| 4–6 GB | `Qwen/Qwen2.5-1.5B-Instruct` | 0.70–0.80 |
| 6–8 GB | `Qwen/Qwen2.5-3B-Instruct` | 0.60–0.70 |
| 8–12 GB | `Qwen/Qwen2.5-3B-Instruct` | 0.80–0.90 |
| 12–16 GB | `Qwen/Qwen2.5-7B-Instruct` | 0.50–0.70 |
| 16–24 GB | `Qwen/Qwen2.5-7B-Instruct` | 0.80–0.90 |

### Memory-aware scheduling

`--enable-memory-aware-scheduling` (default: on) uses `torch.cuda.mem_get_info()` to detect free GPU memory before each dispatch decision. If an action would exceed available memory, it is skipped. You will see a warning:

```
[sched WARNING] requested (xE=1,xR=0) but scheduler chose (xE=0,xR=0) due to memory constraints.
```

Disable it with `--disable-memory-aware-scheduling` if you want to force a specific action regardless of memory.

---

## Troubleshooting

### vLLM out of memory (OOM)

- Reduce `--gpu-memory-utilization` (e.g., 0.5 instead of 0.8)
- Switch embed to CPU: `--xE 0`
- Use a smaller generator model
- Reduce `--max-model-len` if your queries are short

### "No module named 'faiss'" or "faiss.swigfaiss_avx2"

```bash
# Python < 3.13: use conda-forge
conda install -c conda-forge faiss-gpu   # GPU retrieval (xR=1)
conda install -c conda-forge faiss-cpu   # CPU-only

# Python >= 3.13: CPU only from PyPI
pip install faiss-cpu
# GPU retrieval requires building from source: https://github.com/facebookresearch/faiss
```

### Model downloads failing (China mainland)

```bash
export HF_ENDPOINT=https://hf-mirror.com
python async_rag_pipeline.py ...
```

The pipeline sets this automatically at startup. For other scripts:

```bash
HF_ENDPOINT=https://hf-mirror.com python build_index.py ...
HF_ENDPOINT=https://hf-mirror.com python generate_queries.py ...
```

### Slow embedding on CPU

If `--xE 0` is too slow and you have free GPU memory, try `--xE 1`. Note that the scheduler may automatically downgrade to `--xE 0` if GPU memory is insufficient; a WARNING will be printed.

### "WARNING: destroy_process_group() was not called"

This is a vLLM internal warning about process group cleanup. It is harmless and does not affect results. You can suppress it with `torch.distributed.destroy_process_group()` at the end of your script.

### Serial is faster than async_plain

This is expected for small query sets or when generation dominates total time. The async pipeline shows its advantage with larger query sets (>= 64 queries) where batching aggregation outweighs thread overhead.

---

## File inventory

| File | Purpose |
|------|---------|
| `async_rag_pipeline.py` | **Main entry point.** Single script for all modes. |
| `run_comparison.py` | Runs serial / async_plain / async_v2 and renders a comparison table. |
| `build_and_run.sh` | One-shot script: download → index → run comparison. |
| `auto_tune.py` | **Recommended first step.** Auto-detect GPU, calibrate all (xE,xR), save EMA params. |
| `calibrate_sweep.py` | Profile all (xE, xR) combinations across batch sizes. |
| `build_index.py` | Build FAISS index from corpus embeddings. |
| `corpus_builder.py` | Download BEIR dataset (corpus + queries + qrels). |
| `generate_queries.py` | Add token-length metadata to queries. |
| `run_ablation.py` | Ablation study variants. |
| `run_generation_target_eval.py` | Generation target strategy evaluation. |
