#!/bin/bash
# Run v2 experiments with improved search logic
# GPU 0 has ~80GB free VRAM - enough to share with the currently running task

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export HF_ENDPOINT=https://hf-mirror.com

export HF_HOME=/root/autodl-tmp/hf_cache
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_cache
PYTHON=$(which python)
SCRIPT=run_all_wiki_experiments_v2.py
LOGDIR=logs

# mkdir -p $LOGDIR

# echo "$(date): Starting v2 RAG experiment (top_k=5, 1500 char context)..."
# $PYTHON $SCRIPT --experiment rag > ${LOGDIR}/logs_rag_wiki_0318_v2.log 2>&1
# echo "$(date): v2 RAG done."

# echo "$(date): Starting v2 ReAct experiment (title-match first, improved search)..."
# $PYTHON $SCRIPT --experiment react > ${LOGDIR}/logs_react_wiki_0329_v2.log 2>&1
# echo "$(date): v2 ReAct done."

# echo "$(date): Starting v2 ReAct-KV (H2O) experiment..."
# $PYTHON $SCRIPT --experiment react_kv_h2o > ${LOGDIR}/logs_react_kv_h2o_wiki_0414_true.log 2>&1
# echo "$(date): v2 ReAct-KV (H2O) done."

# echo "$(date): Starting v2 ReAct-KV (Step-Anchor H2O) experiment..."
# $PYTHON $SCRIPT --experiment react_kv_step_anchor_h2o > ${LOGDIR}/logs_react_kv_step_anchor_h2o_wiki_0415_v2.log 2>&1
# echo "$(date): v2 ReAct-KV (Step-Anchor H2O) done."

echo "$(date): Starting v2 ReAct-KV (Step-Aware H2O) experiment..."
$PYTHON $SCRIPT --experiment react_kv_step_aware_h2o > ${LOGDIR}/logs_react_kv_step_aware_h2o_wiki_0415_0.9.log 2>&1
echo "$(date): v2 ReAct-KV (Step-Aware H2O) done."

# echo "$(date): Starting v2 ReAct_ours experiment ..."
# $PYTHON $SCRIPT --experiment ours > ${LOGDIR}/logs_react_ours_wiki_0404.log 2>&1
# echo "$(date): v2 ReAct_ours done."

# echo "$(date): Starting v2 ReAct-KV (none) experiment..."
# $PYTHON $SCRIPT --experiment react_kv_none > ${LOGDIR}/logs_react_kv_none_wiki_0318_v2.log 2>&1
# echo "$(date): v2 ReAct-KV (none) done."

# echo "$(date): Starting v2 ReAct-KV (snapkv) experiment..."
# $PYTHON $SCRIPT --experiment react_kv_snapkv > ${LOGDIR}/logs_react_kv_snapkv_wiki_0318_v2.log 2>&1
# echo "$(date): v2 ReAct-KV (snapkv) done."

# echo "$(date): Collecting v2 results..."
# $PYTHON $SCRIPT --experiment collect > ${LOGDIR}/logs_collect_0318_v2.log 2>&1
# echo "$(date): All v2 experiments complete!"
