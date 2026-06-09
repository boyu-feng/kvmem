#!/bin/bash
# Run HotpotQA main experiments with Llama-3.1-8B-Instruct
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
unset HF_ENDPOINT

export HF_HOME=/root/autodl-tmp/hf_cache
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_cache
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/hf_cache/hub

PYTHON=$(which python)
SCRIPT=run_all_wiki_experiments_v2.py
LOGDIR=logs

MODEL_REPO="meta-llama/Llama-3.1-8B-Instruct"
LOCAL_MODEL_DIR="/root/autodl-tmp/hf_cache/models/Llama-3.1-8B-Instruct"
MODEL_PATH="$LOCAL_MODEL_DIR"
OUTPUT_DIR="results/wiki_llama31_8b_v2"
EXPERIMENT="all"

mkdir -p "$LOGDIR"
LOG_FILE="${LOGDIR}/logs_${EXPERIMENT}_wiki_llama31_8b.log"

echo "$(date): Downloading model ${MODEL_REPO} to ${LOCAL_MODEL_DIR}..."
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

echo "$(date): Starting HotpotQA ${EXPERIMENT} with local model ${MODEL_PATH}..."
$PYTHON -u "$SCRIPT" \
  --experiment "$EXPERIMENT" \
  --model_path "$MODEL_PATH" \
  --output_dir "$OUTPUT_DIR" 2>&1 | tee "$LOG_FILE"
echo "$(date): HotpotQA ${EXPERIMENT} done."
