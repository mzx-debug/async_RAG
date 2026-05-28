#!/bin/bash
#
# build_and_run.sh — V1 resource-constrained experiment (BEIR corpus)
#
# Run this ON THE TARGET DEVICE (the resource-constrained machine).
#
#   Step 1: Download a BEIR dataset (corpus + queries + qrels)
#   Step 2: Build FAISS index
#   Step 3: Post-process queries
#   Step 4: Run comparison
#
# Prerequisites:
#   - Python >= 3.10 with venv set up (see README)
#   - vLLM installed and compatible with your CUDA version
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Auto-activate conda env that has vllm installed.
# Tries in order: env from CONDA_DEFAULT_ENV, env from CONDA_ENV_PATH,
# named envs that have vllm, or the first env with torch+cuda.
if [ -z "${VLLM_ENV_SKIP_ACTIVATE:-}" ] && [ -z "${CONDA_PREFIX:-}" ]; then
    if command -v conda &>/dev/null; then
        _found_env=""
        # 1. Look for an env that has vllm
        for _env_path in $(conda env list --json 2>/dev/null | python3 -c "import sys,json; [print(e) for e in json.load(sys.stdin)['envs']]" 2>/dev/null); do
            if [ -f "$_env_path/bin/python" ] && "$_env_path/bin/python" -c "import vllm" 2>/dev/null; then
                _found_env="$_env_path"
                break
            fi
        done
        # 2. Fallback: try common env names
        if [ -z "$_found_env" ]; then
            for _name in p702 rag async_rag vllm pytorch; do
                _env_path="$(conda env list --json 2>/dev/null | python3 -c "import sys,json; envs=json.load(sys.stdin)['envs']; print(next((e for e in envs if e.endswith('/$_name') or e == '$_name'), ''))" 2>/dev/null)"
                if [ -n "$_env_path" ] && [ -f "$_env_path/bin/python" ]; then
                    _found_env="$_env_path"
                    break
                fi
            done
        fi
        if [ -n "$_found_env" ]; then
            echo "[ENV] Auto-activating conda env: $_found_env"
            source activate "$_found_env"
        else
            echo "[ENV] WARNING: Could not auto-detect a conda env with vllm."
            echo "        Please activate your vllm environment manually, e.g.:"
            echo "        conda env list"
            echo "        conda activate <your-env-name>"
            echo "        Then re-run this script."
            exit 1
        fi
    else
        echo "[ENV] WARNING: conda not found. Ensure vllm, torch (cuda), and faiss-cpu are installed in your current Python environment."
    fi
fi

echo "=============================================="
echo "  Async RAG Pipeline V1 — BEIR Corpus Edition"
echo "=============================================="
echo ""

# ── Configuration ────────────────────────────────────────────────
# Choose BEIR dataset: nfcorpus (~3.6K), scifact (~5K), arguana (~8.7K)
DATASET="${DATASET:-nfcorpus}"

CORPUS_PATH="./data/beir_${DATASET}/corpus.jsonl"
QUERIES_BEIR="./data/beir_${DATASET}/queries_beir.jsonl"
QUERIES_OUT="./data/beir_${DATASET}/queries.jsonl"
INDEX_PATH="./indexes/beir_${DATASET}/faiss.index"

EMBED_MODEL="sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL="${GEN_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
BATCH="${BATCH:-32}"
GPU_UTIL="${GPU_UTIL:-0.6}"
GPU_ID="${GPU_ID:-0}"
SAMPLE_QUERIES="${SAMPLE_QUERIES:-256}"
# ─────────────────────────────────────────────────────────────────

echo "Configuration:"
echo "  DATASET       = $DATASET"
echo "  EMBED_MODEL   = $EMBED_MODEL"
echo "  GEN_MODEL     = $GEN_MODEL"
echo "  BATCH         = $BATCH"
echo "  GPU_UTIL      = $GPU_UTIL"
echo "  GPU_ID        = $GPU_ID"
echo ""

# Step 1: Download BEIR corpus + queries
if [ -f "$CORPUS_PATH" ] && [ -f "$QUERIES_BEIR" ]; then
    echo "[SKIP] BEIR corpus already downloaded: $CORPUS_PATH"
else
    echo "[1/4] Downloading BEIR dataset: $DATASET ..."
    python corpus_builder.py \
        --dataset "$DATASET" \
        --output "./data/beir_${DATASET}"
fi
echo ""

# Step 2: Build FAISS index
if [ -f "$INDEX_PATH" ]; then
    echo "[SKIP] FAISS index already exists: $INDEX_PATH"
else
    echo "[2/4] Building FAISS index (Flat, MiniLM-L6) ..."
    python build_index.py \
        --corpus-path "$CORPUS_PATH" \
        --output-dir "./indexes/beir_${DATASET}" \
        --model-path "$EMBED_MODEL" \
        --batch-size 256 \
        --max-length 384 \
        --pooling-method mean \
        --use-fp16 \
        --faiss-type Flat \
        --device cuda
fi
echo ""

# Step 3: Post-process queries
if [ -f "$QUERIES_OUT" ]; then
    echo "[SKIP] Queries already post-processed: $QUERIES_OUT"
else
    echo "[3/4] Post-processing queries (token lengths) ..."
    python generate_queries.py \
        --queries-file "$QUERIES_BEIR" \
        --output "$QUERIES_OUT" \
        --tokenizer-model "$EMBED_MODEL" \
        --auto-threshold
fi
echo ""

# Step 4: Run comparison
echo "[4/4] Running comparison (serial / async_plain / async_v2) ..."
mkdir -p "./output/comparison_${DATASET}"
python run_comparison.py \
    --workdir . \
    --index-path "$INDEX_PATH" \
    --corpus-path "$CORPUS_PATH" \
    --generator-model "$GEN_MODEL" \
    --queries-file "$QUERIES_OUT" \
    --sample-queries "$SAMPLE_QUERIES" \
    --b "$BATCH" \
    --xE 0 --xR 0 \
    --nprobe 1 --topk 1 \
    --gpu-id "$GPU_ID" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --output-dir "./output/comparison_${DATASET}"

echo ""
echo "=============================================="
echo "  Done. Results in ./output/comparison_${DATASET}/"
echo "=============================================="
echo ""
echo "Dataset sizes:"
echo "  nfcorpus  ~3,600 docs  (~0.7 MB index)"
echo "  scifact   ~5,180 docs  (~1.0 MB index)"
echo "  arguana   ~8,700 docs  (~1.7 MB index)"
echo ""
echo "To switch dataset:"
echo "  DATASET=scifact bash build_and_run.sh"
echo "  DATASET=arguana bash build_and_run.sh"
