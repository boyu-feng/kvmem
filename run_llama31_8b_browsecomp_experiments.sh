#!/bin/bash
# Run BrowseComp main experiments with Llama-3.1-8B-Instruct
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
unset HF_ENDPOINT

export HF_HOME=/root/autodl-tmp/hf_cache
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_cache
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/hf_cache/hub

PYTHON=$(which python)
SCRIPT=run_all_browsecomp_experiments_v2.py
LOGDIR=logs

MODEL_REPO="meta-llama/Meta-Llama-3.1-8B-Instruct"
LOCAL_MODEL_DIR="/root/autodl-tmp/hf_cache/models/Meta-Llama-3.1-8B-Instruct"
MODEL_PATH="$LOCAL_MODEL_DIR"
OUTPUT_DIR="results/browsecomp_llama31_8b_v2"
HF_DATASET_NAME="Tevatron/browsecomp-plus"
RETRIEVER_BACKEND="web"
MAX_STEPS=40
EXPERIMENT="all"

mkdir -p "$LOGDIR"
LOG_FILE="${LOGDIR}/logs_${EXPERIMENT}_browsecomp_llama31_8b.log"

if [ -f "$LOCAL_MODEL_DIR/config.json" ] && [ -f "$LOCAL_MODEL_DIR/tokenizer_config.json" ]; then
  echo "$(date): Found local model at ${LOCAL_MODEL_DIR}, skip download."
elif [ -d "$LOCAL_MODEL_DIR/original" ]; then
  echo "[ERROR] Found ${LOCAL_MODEL_DIR}/original, but Transformers model files are missing."
  echo "[ERROR] Please download full HF format files (do NOT use --include \"original/*\")."
  echo "[ERROR] Example: huggingface-cli download ${MODEL_REPO} --local-dir ${LOCAL_MODEL_DIR}"
  exit 1
else
  echo "$(date): Local model not found, downloading ${MODEL_REPO} to ${LOCAL_MODEL_DIR}..."
  mkdir -p "$(dirname "$LOCAL_MODEL_DIR")"
  if command -v hf >/dev/null 2>&1; then
    hf download "$MODEL_REPO" \
      --local-dir "$LOCAL_MODEL_DIR"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "$MODEL_REPO" \
      --local-dir "$LOCAL_MODEL_DIR"
  else
    echo "[ERROR] Neither 'hf' nor 'huggingface-cli' is installed."
    exit 1
  fi
fi

echo "$(date): Starting BrowseComp ${EXPERIMENT} with local model ${MODEL_PATH}..."
$PYTHON -u "$SCRIPT" \
  --experiment "$EXPERIMENT" \
  --model_path "$MODEL_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --hf_dataset_name "$HF_DATASET_NAME" \
  --retriever_backend "$RETRIEVER_BACKEND" \
  --max_steps "$MAX_STEPS" 2>&1 | tee "$LOG_FILE"
echo "$(date): BrowseComp ${EXPERIMENT} done."
