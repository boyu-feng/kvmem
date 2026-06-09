#!/bin/bash
# Run Llama-3.1-8B experiments on HotpotQA + 2Wiki + MuSiQue
set -euo pipefail

LOGDIR=logs
mkdir -p "$LOGDIR"
MASTER_LOG="${LOGDIR}/logs_llama31_8b_hotpot_2wiki_musique.log"

run_one() {
  local script="$1"
  echo "$(date): Starting ${script}" | tee -a "$MASTER_LOG"
  bash "$script" 2>&1 | tee -a "$MASTER_LOG"
  echo "$(date): Finished ${script}" | tee -a "$MASTER_LOG"
}

run_one "run_llama31_8b_wiki_experiments.sh"
run_one "run_llama31_8b_2wiki_experiments.sh"
run_one "run_llama31_8b_musique_experiments.sh"

echo "$(date): All three datasets complete." | tee -a "$MASTER_LOG"
