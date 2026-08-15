#!/usr/bin/env bash
# Phase 1 — Experiment 1b: Role-prompt single-agent (SA), 4 models × run1.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python3}"

MODEL="${MODEL:-}"

run_sa() {
  local model="$1"
  local slug="${model//\//_}"
  slug="${slug//./-}"
  export MARC_CACHE_DIR="${ROOT}/results/cache/sa/${slug}/run1"
  echo "==> SA model=${model} cache=${MARC_CACHE_DIR}"
  if [[ "$model" == "qwen-plus" ]]; then
    "$PY" src/experiments/run_qwen_sa_full.py --run 1
  else
    "$PY" src/experiments/run_sa_full.py --model "$model" --run 1
  fi
}

MODELS=(gemini-2.5-flash-lite claude-3-haiku-20240307 gpt-4o qwen-plus)
if [[ -n "$MODEL" ]]; then MODELS=("$MODEL"); fi

for model in "${MODELS[@]}"; do
  run_sa "$model"
done

echo "Done: SA inference. Outputs under results/sa/"
