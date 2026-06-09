#!/bin/bash
# Run HotpotQA main experiments with Llama-3.1-8B-Instruct
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export HF_ENDPOINT=https://hf-mirror.com

export HF_HOME=/root/autodl-tmp/hf_cache
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_cache
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/hf_cache/hub

PYTHON=$(which python)
SCRIPT=run_all_wiki_experiments_v2.py
LOGDIR=logs

MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"
OUTPUT_DIR="results/wiki_llama31_8b_v2"
EXPERIMENT="all"

mkdir -p "$LOGDIR"
LOG_FILE="${LOGDIR}/logs_${EXPERIMENT}_wiki_llama31_8b.log"

echo "$(date): Starting HotpotQA ${EXPERIMENT} with ${MODEL_PATH}..."
$PYTHON -u "$SCRIPT" \
  --experiment "$EXPERIMENT" \
  --model_path "$MODEL_PATH" \
  --output_dir "$OUTPUT_DIR" 2>&1 | tee "$LOG_FILE"
echo "$(date): HotpotQA ${EXPERIMENT} done."
