#!/bin/bash
# Run v2 experiments on BrowseComp with local BrowseComp BM25 index

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export HF_ENDPOINT=https://hf-mirror.com

export HF_HOME=/root/autodl-tmp/hf_cache
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_cache

PYTHON=$(which python)
SCRIPT=run_all_browsecomp_experiments_v2.py
LOGDIR=logs
INDEX_DIR=browsecomp_index
DATA_PATH=browsecomp/decrypted.jsonl
MODEL_PATH=Qwen/Qwen3-4B-Thinking-2507

mkdir -p "$LOGDIR"

# Build index first if needed:
# $PYTHON build_browsecomp_index.py --index_dir "$INDEX_DIR"

# echo "$(date): Starting BrowseComp Single experiment..."
# $PYTHON $SCRIPT --experiment single --data_path "$DATA_PATH" > "${LOGDIR}/logs_single_browsecomp_bm25.log" 2>&1
# echo "$(date): BrowseComp Single done."

# echo "$(date): Starting BrowseComp RAG experiment..."
# $PYTHON $SCRIPT --experiment rag --data_path "$DATA_PATH" --retriever_backend browsecomp_bm25 --browsecomp_index_dir "$INDEX_DIR" > "${LOGDIR}/logs_rag_browsecomp_bm25.log" 2>&1
# echo "$(date): BrowseComp RAG done."

echo "$(date): Starting BrowseComp ReAct experiment..."
$PYTHON $SCRIPT --experiment react --data_path "$DATA_PATH" --retriever_backend browsecomp_bm25 --browsecomp_index_dir "$INDEX_DIR" --model_path "$MODEL_PATH" > "${LOGDIR}/logs_react_browsecomp_bm25.log" 2>&1
echo "$(date): BrowseComp ReAct done."

# echo "$(date): Starting BrowseComp ReAct-KV (Step-Aware H2O, BrowseComp BM25) experiment..."
# $PYTHON $SCRIPT --experiment react_kv_step_aware_h2o --max_steps 40 --data_path "$DATA_PATH" --retriever_backend browsecomp_bm25 --browsecomp_index_dir "$INDEX_DIR" > "${LOGDIR}/logs_react_kv_step_aware_h2o_browsecomp_bm25.log" 2>&1
# echo "$(date): BrowseComp ReAct-KV (Step-Aware H2O, BrowseComp BM25) done."

# echo "$(date): Starting BrowseComp ReAct-KV (Step-Inter, BrowseComp BM25) experiment..."
# $PYTHON $SCRIPT --experiment react_kv_step_inter --data_path "$DATA_PATH" --retriever_backend browsecomp_bm25 --browsecomp_index_dir "$INDEX_DIR" > "${LOGDIR}/logs_react_kv_step_inter_browsecomp_bm25.log" 2>&1
# echo "$(date): BrowseComp ReAct-KV (Step-Inter, BrowseComp BM25) done."

# echo "$(date): Starting BrowseComp ReAct-KV (TOVA, BrowseComp BM25) experiment..."
# $PYTHON $SCRIPT --experiment react_kv_tova --data_path "$DATA_PATH" --retriever_backend browsecomp_bm25 --browsecomp_index_dir "$INDEX_DIR" --model_path "$MODEL_PATH" > "${LOGDIR}/logs_react_kv_tova_browsecomp_bm25.log" 2>&1
# echo "$(date): BrowseComp ReAct-KV (TOVA, BrowseComp BM25) done."

# echo "$(date): Starting all BrowseComp experiments (BrowseComp BM25)..."
# $PYTHON $SCRIPT --experiment all --data_path "$DATA_PATH" --retriever_backend browsecomp_bm25 --browsecomp_index_dir "$INDEX_DIR" > "${LOGDIR}/logs_all_browsecomp_bm25.log" 2>&1
# echo "$(date): All BrowseComp experiments (BrowseComp BM25) done."
