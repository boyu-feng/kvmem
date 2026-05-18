#!/bin/bash
# Run v2 experiments on BrowseComp dataset
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export HF_ENDPOINT=https://hf-mirror.com

export HF_HOME=/root/autodl-tmp/hf_cache
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_cache

PYTHON=$(which python)
SCRIPT=run_all_browsecomp_experiments_v2.py
LOGDIR=logs
MODEL_PATH=/root/autodl-tmp/hf_cache/models/Qwen3.5-9B
HF_DATASET_NAME=Tevatron/browsecomp-plus

mkdir -p "$LOGDIR"
LOG_FILE="${LOGDIR}/logs_react_browsecomp.log"

# echo "$(date): Starting BrowseComp Single experiment..."
# $PYTHON $SCRIPT --experiment single > "${LOGDIR}/logs_single_browsecomp.log" 2>&1
# echo "$(date): BrowseComp Single done."

# echo "$(date): Starting BrowseComp RAG experiment..."
# $PYTHON $SCRIPT --experiment rag > "${LOGDIR}/logs_rag_browsecomp.log" 2>&1
# echo "$(date): BrowseComp RAG done."

echo "$(date): Starting BrowseComp ReAct experiment..."
$PYTHON -u $SCRIPT --experiment react --hf_dataset_name "$HF_DATASET_NAME" --model_path "$MODEL_PATH" 2>&1 | tee "$LOG_FILE"
echo "$(date): BrowseComp ReAct done."

# echo "$(date): Starting BrowseComp ReAct-KV (none) experiment..."
# $PYTHON $SCRIPT --experiment react_kv_none > "${LOGDIR}/logs_react_kv_none_browsecomp.log" 2>&1
# echo "$(date): BrowseComp ReAct-KV (none) done."

# echo "$(date): Starting BrowseComp ReAct-KV (Step-Aware H2O) experiment..."
# $PYTHON $SCRIPT --experiment react_kv_step_aware_h2o --max_steps 40 --retriever_backend web --model_path "$MODEL_PATH" > "${LOGDIR}/logs_react_kv_step_aware_h2o_browsecomp.log" 2>&1
# echo "$(date): BrowseComp ReAct-KV (Step-Aware H2O) done."

# echo "$(date): Starting BrowseComp ReAct-KV (H2O) experiment..."
# $PYTHON $SCRIPT --experiment react_kv_h2o > "${LOGDIR}/logs_react_kv_h2o_browsecomp.log" 2>&1
# echo "$(date): BrowseComp ReAct-KV (H2O) done."

# echo "$(date): Starting BrowseComp ReAct-KV (Step-Anchor H2O) experiment..."
# $PYTHON $SCRIPT --experiment react_kv_step_anchor_h2o > "${LOGDIR}/logs_react_kv_step_anchor_h2o_browsecomp.log" 2>&1
# echo "$(date): BrowseComp ReAct-KV (Step-Anchor H2O) done."

# echo "$(date): Starting BrowseComp ReAct-KV (Step-Inter) experiment..."
# $PYTHON $SCRIPT --experiment react_kv_step_inter > "${LOGDIR}/logs_react_kv_step_inter_browsecomp.log" 2>&1
# echo "$(date): BrowseComp ReAct-KV (Step-Inter) done."

# echo "$(date): Starting BrowseComp ReAct-KV (SnapKV) experiment..."
# $PYTHON $SCRIPT --experiment react_kv_snapkv > "${LOGDIR}/logs_react_kv_snapkv_browsecomp.log" 2>&1
# echo "$(date): BrowseComp ReAct-KV (SnapKV) done."

# echo "$(date): Starting BrowseComp ReAct-KV (Ours) experiment..."
# $PYTHON $SCRIPT --experiment ours > "${LOGDIR}/logs_react_kv_ours_browsecomp.log" 2>&1
# echo "$(date): BrowseComp ReAct-KV (Ours) done."

# echo "$(date): Starting all BrowseComp experiments..."
# $PYTHON $SCRIPT --experiment all > "${LOGDIR}/logs_all_browsecomp.log" 2>&1
# echo "$(date): All BrowseComp experiments done."
