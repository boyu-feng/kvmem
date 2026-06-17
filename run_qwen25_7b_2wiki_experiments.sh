#!/bin/bash
# Run 2Wiki main experiments with Qwen2.5-7B-Instruct
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
unset HF_ENDPOINT
export HF_HUB_DISABLE_XET=1

export HF_HOME=/root/autodl-tmp/hf_cache
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_cache
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/hf_cache/hub

PYTHON=$(which python)
SCRIPT=run_all_2wiki_experiments_v2.py
METRICS_SCRIPT=record_experiment_metrics.py
LOGDIR=logs

MODEL_REPO="Qwen/Qwen2.5-7B-Instruct"
LOCAL_MODEL_DIR="/root/autodl-tmp/hf_cache/models/Qwen2.5-7B-Instruct"
MODEL_PATH="$LOCAL_MODEL_DIR"
OUTPUT_ROOT="results/2wiki_qwen25_7b_v2"
DATA_PATH="/root/autodl-tmp/kvmem/data/2wiki/dev.json"
# Repeat the full suite N times into separate dirs; previous results are untouched.
# Each repeat uses a different sampling seed (run1/original used seed 233).
RUN_TAGS=("run2" "run3")
RUN_SEEDS=(42 3407)
RUN=""
SEED=""

mkdir -p "$LOGDIR"

if [ -f "$LOCAL_MODEL_DIR/config.json" ] && [ -f "$LOCAL_MODEL_DIR/tokenizer_config.json" ]; then
  echo "$(date): Found local model at ${LOCAL_MODEL_DIR}, skip download."
else
  echo "$(date): Local model not found, downloading ${MODEL_REPO} to ${LOCAL_MODEL_DIR}..."
  mkdir -p "$(dirname "$LOCAL_MODEL_DIR")"
  if command -v hf >/dev/null 2>&1; then
    hf download "$MODEL_REPO" --local-dir "$LOCAL_MODEL_DIR"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "$MODEL_REPO" --local-dir "$LOCAL_MODEL_DIR"
  else
    echo "[ERROR] Neither 'hf' nor 'huggingface-cli' is installed."
    exit 1
  fi
fi

run_exp() {
  local exp_name="$1"
  local output_dir="$2"
  local cache_ratio="${3:-}"
  local tag="$exp_name"
  if [ -n "$cache_ratio" ]; then
    tag="${exp_name}_r${cache_ratio/./}"
  fi
  local log_file="${LOGDIR}/logs_${tag}_2wiki_qwen25_7b_${RUN}.log"
  local result_json=""

  echo "$(date): Starting 2Wiki ${exp_name} ..."
  local data_args=""
  if [ -f "$DATA_PATH" ]; then
    data_args="--data_path $DATA_PATH"
  fi

  if [ -n "$cache_ratio" ]; then
    $PYTHON -u "$SCRIPT" \
      --experiment "$exp_name" \
      --model_path "$MODEL_PATH" \
      --output_dir "$output_dir" \
      --seed "$SEED" \
      --cache_ratio "$cache_ratio" \
      ${data_args} 2>&1 | tee "$log_file"
  else
    $PYTHON -u "$SCRIPT" \
      --experiment "$exp_name" \
      --model_path "$MODEL_PATH" \
      --output_dir "$output_dir" \
      --seed "$SEED" \
      ${data_args} 2>&1 | tee "$log_file"
  fi

  case "$exp_name" in
    single) result_json="${output_dir}/single_2wiki.json" ;;
    react) result_json="${output_dir}/react_2wiki.json" ;;
    react_kv_none) result_json="${output_dir}/react_kv_none_2wiki.json" ;;
    react_kv_h2o) result_json="${output_dir}/react_kv_h2o_2wiki.json" ;;
    react_kv_tova) result_json="${output_dir}/react_kv_tova_2wiki.json" ;;
    react_kv_step_aware_h2o) result_json="${output_dir}/react_kv_step_aware_h2o_2wiki.json" ;;
  esac

  if [ -n "$result_json" ]; then
    $PYTHON "$METRICS_SCRIPT" \
      --result_json "$result_json" \
      --dataset "2wiki" \
      --method "$exp_name" \
      --cache_ratio "$cache_ratio" \
      --output_file "${output_dir}/metrics_${tag}.md"
  fi
  echo "$(date): 2Wiki ${exp_name} done."
}

for i in "${!RUN_TAGS[@]}"; do
  RUN="${RUN_TAGS[$i]}"
  SEED="${RUN_SEEDS[$i]}"
  OUTPUT_BASE="${OUTPUT_ROOT}/${RUN}"
  echo "$(date): ===== Repeat ${RUN} (seed=${SEED}) -> ${OUTPUT_BASE} ====="

  run_exp "single" "${OUTPUT_BASE}/single"
  run_exp "react" "${OUTPUT_BASE}/react"
  run_exp "react_kv_none" "${OUTPUT_BASE}/fullkv"
  run_exp "react_kv_h2o" "${OUTPUT_BASE}/h2o_r50" "0.5"
  run_exp "react_kv_h2o" "${OUTPUT_BASE}/h2o_r20" "0.2"
  run_exp "react_kv_tova" "${OUTPUT_BASE}/tova_r50" "0.5"
  run_exp "react_kv_tova" "${OUTPUT_BASE}/tova_r20" "0.2"
  run_exp "react_kv_step_aware_h2o" "${OUTPUT_BASE}/stepaware_r50" "0.5"
  run_exp "react_kv_step_aware_h2o" "${OUTPUT_BASE}/stepaware_r20" "0.2"
done
