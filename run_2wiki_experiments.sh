#!/bin/bash
# Run v2 experiments on 2Wiki dataset

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export HF_ENDPOINT=https://hf-mirror.com

export HF_HOME=/root/autodl-tmp/hf_cache
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_cache

PYTHON=$(which python)
SCRIPT=run_all_2wiki_experiments_v2.py
LOGDIR=logs

mkdir -p "$LOGDIR"

# echo "$(date): Starting 2Wiki Single experiment..."
# $PYTHON $SCRIPT --experiment single > "${LOGDIR}/logs_single_2wiki.log" 2>&1
# echo "$(date): 2Wiki Single done."

# echo "$(date): Starting 2Wiki RAG experiment..."
# $PYTHON $SCRIPT --experiment rag > "${LOGDIR}/logs_rag_2wiki.log" 2>&1
# echo "$(date): 2Wiki RAG done."

# echo "$(date): Starting 2Wiki ReAct experiment..."
# $PYTHON $SCRIPT --experiment react > "${LOGDIR}/logs_react_2wiki.log" 2>&1
# echo "$(date): 2Wiki ReAct done."

# echo "$(date): Starting 2Wiki ReAct-KV (none) experiment..."
# $PYTHON $SCRIPT --experiment react_kv_none > "${LOGDIR}/logs_react_kv_none_2wiki.log" 2>&1
# echo "$(date): 2Wiki ReAct-KV (none) done."

echo "$(date): Starting 2Wiki ReAct-KV (H2O) experiment..."
$PYTHON $SCRIPT --experiment react_kv_h2o > "${LOGDIR}/logs_react_kv_h2o_2wiki.log" 2>&1
echo "$(date): 2Wiki ReAct-KV (H2O) done."

# echo "$(date): Starting 2Wiki ReAct-KV (Step-Anchor H2O) experiment..."
# $PYTHON $SCRIPT --experiment react_kv_step_anchor_h2o > "${LOGDIR}/logs_react_kv_step_anchor_h2o_2wiki.log" 2>&1
# echo "$(date): 2Wiki ReAct-KV (Step-Anchor H2O) done."

# echo "$(date): Starting 2Wiki ReAct-KV (Step-Aware H2O) experiment..."
# $PYTHON $SCRIPT --experiment react_kv_step_aware_h2o > "${LOGDIR}/logs_react_kv_step_aware_h2o_2wiki.log" 2>&1
# echo "$(date): 2Wiki ReAct-KV (Step-Aware H2O) done."

# echo "$(date): Starting 2Wiki ReAct-KV (SnapKV) experiment..."
# $PYTHON $SCRIPT --experiment react_kv_snapkv > "${LOGDIR}/logs_react_kv_snapkv_2wiki.log" 2>&1
# echo "$(date): 2Wiki ReAct-KV (SnapKV) done."

# echo "$(date): Starting 2Wiki ReAct-KV (Ours) experiment..."
# $PYTHON $SCRIPT --experiment ours > "${LOGDIR}/logs_react_kv_ours_2wiki.log" 2>&1
# echo "$(date): 2Wiki ReAct-KV (Ours) done."

# echo "$(date): Starting all 2Wiki experiments..."
# $PYTHON $SCRIPT --experiment all > "${LOGDIR}/logs_all_2wiki.log" 2>&1
# echo "$(date): All 2Wiki experiments done."
