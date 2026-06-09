#!/bin/bash
# Run MuSiQue main experiments with Llama-3.1-8B-Instruct
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export HF_ENDPOINT=https://hf-mirror.com

export HF_HOME=/root/autodl-tmp/hf_cache
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_cache
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/hf_cache/hub

PYTHON=$(which python)
SCRIPT=run_all_musique_experiments_v2.py
LOGDIR=logs

MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"
OUTPUT_DIR="results/musique_llama31_8b_v2"
EXPERIMENT="all"
MAX_STEPS=12

mkdir -p "$LOGDIR"
LOG_FILE="${LOGDIR}/logs_${EXPERIMENT}_musique_llama31_8b.log"

echo "$(date): Starting MuSiQue ${EXPERIMENT} with ${MODEL_PATH}..."
$PYTHON -u "$SCRIPT" \
  --experiment "$EXPERIMENT" \
  --model_path "$MODEL_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --max_steps "$MAX_STEPS" 2>&1 | tee "$LOG_FILE"
echo "$(date): MuSiQue ${EXPERIMENT} done."
