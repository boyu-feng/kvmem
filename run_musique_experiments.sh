#!/bin/bash
# Run v2 experiments on MuSiQue dataset

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export HF_ENDPOINT=https://hf-mirror.com

export HF_HOME=/root/autodl-tmp/hf_cache
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_cache

PYTHON=$(which python)
SCRIPT=run_all_musique_experiments_v2.py
LOGDIR=logs

mkdir -p "$LOGDIR"

# echo "$(date): Starting MuSiQue Single experiment..."
# $PYTHON $SCRIPT --experiment single > "${LOGDIR}/logs_single_musique.log" 2>&1
# echo "$(date): MuSiQue Single done."

# echo "$(date): Starting MuSiQue RAG experiment..."
# $PYTHON $SCRIPT --experiment rag > "${LOGDIR}/logs_rag_musique.log" 2>&1
# echo "$(date): MuSiQue RAG done."

echo "$(date): Starting MuSiQue ReAct experiment..."
$PYTHON $SCRIPT --experiment react > "${LOGDIR}/logs_react_musique.log" 2>&1
echo "$(date): MuSiQue ReAct done."

echo "$(date): Starting MuSiQue ReAct-KV (none, FullKV) experiment..."
$PYTHON $SCRIPT --experiment react_kv_none > "${LOGDIR}/logs_react_kv_none_musique.log" 2>&1
echo "$(date): MuSiQue ReAct-KV (none, FullKV) done."

echo "$(date): Starting MuSiQue ReAct-KV (H2O) experiment..."
$PYTHON $SCRIPT --experiment react_kv_h2o > "${LOGDIR}/logs_react_kv_h2o_musique.log" 2>&1
echo "$(date): MuSiQue ReAct-KV (H2O) done."

echo "$(date): Starting MuSiQue ReAct-KV (TOVA) experiment..."
$PYTHON $SCRIPT --experiment react_kv_tova > "${LOGDIR}/logs_react_kv_tova_musique.log" 2>&1
echo "$(date): MuSiQue ReAct-KV (TOVA) done."

# echo "$(date): Starting MuSiQue ReAct-KV (PyramidInfer) experiment..."
# $PYTHON $SCRIPT --experiment react_kv_pyramidinfer > "${LOGDIR}/logs_react_kv_pyramidinfer_musique.log" 2>&1
# echo "$(date): MuSiQue ReAct-KV (PyramidInfer) done."

# echo "$(date): Starting MuSiQue ReAct-KV (Step-Anchor H2O) experiment..."
# $PYTHON $SCRIPT --experiment react_kv_step_anchor_h2o > "${LOGDIR}/logs_react_kv_step_anchor_h2o_musique.log" 2>&1
# echo "$(date): MuSiQue ReAct-KV (Step-Anchor H2O) done."

# echo "$(date): Starting MuSiQue ReAct-KV (Step-Aware H2O) experiment..."
# $PYTHON $SCRIPT --experiment react_kv_step_aware_h2o > "${LOGDIR}/logs_react_kv_step_aware_h2o_musique.log" 2>&1
# echo "$(date): MuSiQue ReAct-KV (Step-Aware H2O) done."

# echo "$(date): Starting MuSiQue ReAct-KV (Step-Inter) experiment..."
# $PYTHON $SCRIPT --experiment react_kv_step_inter > "${LOGDIR}/logs_react_kv_step_inter_musique.log" 2>&1
# echo "$(date): MuSiQue ReAct-KV (Step-Inter) done."

# echo "$(date): Starting MuSiQue ReAct-KV (SnapKV) experiment..."
# $PYTHON $SCRIPT --experiment react_kv_snapkv > "${LOGDIR}/logs_react_kv_snapkv_musique.log" 2>&1
# echo "$(date): MuSiQue ReAct-KV (SnapKV) done."

# echo "$(date): Starting MuSiQue ReAct-KV (Ours) experiment..."
# $PYTHON $SCRIPT --experiment ours > "${LOGDIR}/logs_react_kv_ours_musique.log" 2>&1
# echo "$(date): MuSiQue ReAct-KV (Ours) done."

# echo "$(date): Starting all MuSiQue experiments..."
# $PYTHON $SCRIPT --experiment all > "${LOGDIR}/logs_all_musique.log" 2>&1
# echo "$(date): All MuSiQue experiments done."
