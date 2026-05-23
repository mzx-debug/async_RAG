# Async RAG Pipeline V0

This repository compares three standalone RAG execution modes:

- `serial`: strict `Embedding -> Retrieval -> Generation`
- `async_plain`: fixed microbatch threaded pipeline
- `async_bucket`: three-bucket greedy dispatch plus threaded pipeline

The current implementation is intentionally narrow:

- `nprobe` is fixed by CLI and is not part of scheduling.
- Query compression is not implemented.
- `async_bucket` now uses tokenized query lengths cached at load time.
- `async_bucket` re-selects `xE/xR` online for each outgoing microbatch using stage-specific EMA latency feedback.
- `async_bucket` also chooses batch size online from a small candidate set instead of using only fixed per-bucket constants.
- `xE=1, xR=1` keeps embeddings on GPU between embedding and retrieval when possible.

## Files

- `async_rag_pipeline.py`: main executable pipeline
- `run_comparison.py`: runs the three modes and writes comparison artifacts
- `run_ablation.py`: runs named ablation variants and writes ablation artifacts
- `run_generation_target_eval.py`: compares `generation_target_v1` directly against the `plain_b64` throughput baseline
- `build_index.py`: builds embeddings and FAISS indexes
- `data/`: input corpus and queries
- `comparison/`, `comparison_large/`: saved experiment outputs
- `docs/`: run notes and execution docs

## Setup

```bash
cd /path/to/async_rag_pipeline_v0
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

If generation uses vLLM, install a vLLM build compatible with your CUDA environment.

## Input formats

Corpus files accept:

- `{"contents": "..."}`
- `{"title": "...", "text": "..."}`
- `{"text": "..."}`

Query files accept `.txt`, `.jsonl`, or `.json`. The loader prefers the `question` field and falls back to `query`, `question_text`, or `text`.

## Example: build index

```bash
python ./build_index.py \
  --corpus-path ./data/corpus.jsonl \
  --output-dir ./indexes/ivf4096_flat \
  --model-path intfloat/e5-large-v2 \
  --batch-size 256 \
  --max-length 512 \
  --pooling-method mean \
  --use-fp16 \
  --faiss-type IVF4096,Flat \
  --device cuda
```

Output:

- `indexes/ivf4096_flat/faiss.index`

## Example: run one mode

```bash
python ./async_rag_pipeline.py \
  --pipeline-mode async_bucket \
  --index-path ./indexes/ivf4096_flat/faiss.index \
  --corpus-path ./data/corpus.jsonl \
  --generator-model meta-llama/Llama-3.1-8B-Instruct \
  --queries-file ./data/queries.jsonl \
  --sample-queries 256 \
  --b 64 --xE 1 --xR 0 \
  --nprobe 128 --topk 1 \
  --output-json ./output/summary_async_bucket.json
```

## Example: compare all modes

```bash
python ./run_comparison.py \
  --workdir /path/to/async_rag_pipeline_v0 \
  --index-path /path/to/async_rag_pipeline_v0/indexes/ivf4096_flat/faiss.index \
  --corpus-path /path/to/async_rag_pipeline_v0/data/corpus.jsonl \
  --generator-model meta-llama/Llama-3.1-8B-Instruct \
  --queries-file /path/to/async_rag_pipeline_v0/data/queries.jsonl \
  --sample-queries 256 \
  --b 64 --xE 1 --xR 0 \
  --nprobe 128 --topk 1 \
  --output-dir /path/to/async_rag_pipeline_v0/output/comparison
```

Outputs:

- `summary_serial.json`
- `summary_async_plain.json`
- `summary_async_bucket.json`
- `comparison_rows.json`
- `comparison_table.md`

## Current scheduler parameters

The live scheduler-related CLI surface in `async_rag_pipeline.py` is:

### Basic bucket / batch parameters
- `--length-short-threshold`
- `--length-long-threshold`
- `--bucket-batch-short`
- `--bucket-batch-mid`
- `--bucket-batch-long`
- `--embed-long-gpu-threshold`
- `--retrieve-gpu-batch-threshold`
- `--backpressure-high`
- `--scheduler-ema-alpha`

### Memory-aware scheduling (resource-constrained scenarios)
- `--enable-memory-aware-scheduling` — enable GPU memory-aware action selection and batch shaping (default: True). Use `--disable-memory-aware-scheduling` to fall back to the old static threshold behavior.
- `--gpu-mem-low-threshold-gb` — free memory below this triggers "high" pressure (default: 4.0 GiB)
- `--gpu-mem-medium-threshold-gb` — free memory below this triggers "medium" pressure (default: 10.0 GiB)
- `--gpu-mem-high-batch-penalty` — score penalty for GPU-heavy actions under high pressure (default: 50.0 ms)
- `--faiss-index-gb` — estimated FAISS GPU memory footprint for scheduling decisions (default: 2.0 GiB)

### Removed zombie parameters

- `--length-hard-threshold`
- `--max-processed-length`
- `--embed-mid-gpu-threshold`
- `--backpressure-low`

## Ablation

`run_ablation.py` runs a fixed set of named variants that separate:

- plain large-batch gain
- bucketing gain
- online batch-size gain
- online action-selection gain
- chunking gain

Example:

```bash
python ./run_ablation.py \
  --workdir /path/to/async_rag_pipeline_v0 \
  --index-path /path/to/async_rag_pipeline_v0/indexes/ivf4096_flat/faiss.index \
  --corpus-path /path/to/async_rag_pipeline_v0/data/corpus.jsonl \
  --generator-model meta-llama/Llama-3.1-8B-Instruct \
  --queries-file /path/to/async_rag_pipeline_v0/data/queries.jsonl \
  --sample-queries 256 \
  --b 16 --xE 1 --xR 0 \
  --nprobe 128 --topk 1 \
  --output-dir /path/to/async_rag_pipeline_v0/ablation_output
```

## Generation-target evaluation

If you only want to answer whether `generation_target_v1` can beat the current strongest practical baseline, use:

```bash
python ./run_generation_target_eval.py \
  --workdir /path/to/async_rag_pipeline_v0 \
  --index-path /path/to/async_rag_pipeline_v0/indexes/ivf4096_flat/faiss.index \
  --corpus-path /path/to/async_rag_pipeline_v0/data/corpus.jsonl \
  --generator-model meta-llama/Llama-3.1-8B-Instruct \
  --queries-file /path/to/async_rag_pipeline_v0/data/queries.jsonl \
  --sample-queries 256 \
  --xE 1 --xR 0 \
  --nprobe 128 --topk 1 \
  --output-dir /path/to/async_rag_pipeline_v0/generation_target_eval
```

## Reading results

Prioritize:

- `wall_time_ms`
- `wall_throughput_qps`
- `avg_embedding_ms`
- `avg_retrieval_ms`
- `avg_generation_ms`
- `scheduler.bucket_counts`
- `scheduler.action_counts`

Use `wall_*` metrics for cross-mode performance claims. They reflect real elapsed time; `total_ms` is just the sum of stage times.
