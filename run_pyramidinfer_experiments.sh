#!/bin/bash
# Run PyramidInfer baseline on HotpotQA (wiki runner)

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_cache
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_cache
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/hf_cache/hub

PYTHON=$(which python)
SCRIPT=run_all_wiki_experiments_v2.py
LOGDIR=logs

mkdir -p "$LOGDIR"

echo "$(date): Starting ReAct-KV (PyramidInfer baseline) on HotpotQA..."
$PYTHON $SCRIPT --experiment react_kv_pyramidinfer > "${LOGDIR}/logs_react_kv_pyramidinfer_wiki_0512.log" 2>&1
echo "$(date): ReAct-KV (PyramidInfer baseline) done."
