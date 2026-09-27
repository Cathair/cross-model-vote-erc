#!/usr/bin/env bash
# Phase 1 — Experiment 1d: InsideOut baseline, 4 models × run1.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python3}"

MODEL="${MODEL:-}"

run_io() {
  local model="$1"
  local slug="${model//\//_}"
  slug="${slug//./-}"
  export MARC_CACHE_DIR="${ROOT}/results/cache/insideout/${slug}/run1"
  echo "==> InsideOut model=${model} cache=${MARC_CACHE_DIR}"
  "$PY" src/experiments/run_insideout_full.py --model "$model" --run 1
}

MODELS=(gemini-2.5-flash-lite claude-3-haiku-20240307 qwen-plus gpt-4o)
if [[ -n "$MODEL" ]]; then MODELS=("$MODEL"); fi

for model in "${MODELS[@]}"; do
  run_io "$model"
done

echo "Done: InsideOut inference. Outputs under results/insideout/"
