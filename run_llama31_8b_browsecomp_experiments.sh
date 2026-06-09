#!/bin/bash
# Run BrowseComp main experiments with Llama-3.1-8B-Instruct
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export HF_ENDPOINT=https://hf-mirror.com

export HF_HOME=/root/autodl-tmp/hf_cache
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_cache
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/hf_cache/hub

PYTHON=$(which python)
SCRIPT=run_all_browsecomp_experiments_v2.py
LOGDIR=logs

MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"
OUTPUT_DIR="results/browsecomp_llama31_8b_v2"
HF_DATASET_NAME="Tevatron/browsecomp-plus"
RETRIEVER_BACKEND="web"
MAX_STEPS=40
EXPERIMENT="all"

mkdir -p "$LOGDIR"
LOG_FILE="${LOGDIR}/logs_${EXPERIMENT}_browsecomp_llama31_8b.log"

echo "$(date): Starting BrowseComp ${EXPERIMENT} with ${MODEL_PATH}..."
$PYTHON -u "$SCRIPT" \
  --experiment "$EXPERIMENT" \
  --model_path "$MODEL_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --hf_dataset_name "$HF_DATASET_NAME" \
  --retriever_backend "$RETRIEVER_BACKEND" \
  --max_steps "$MAX_STEPS" 2>&1 | tee "$LOG_FILE"
echo "$(date): BrowseComp ${EXPERIMENT} done."
