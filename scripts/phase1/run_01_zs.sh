#!/usr/bin/env bash
# Phase 1 — Experiment 1a: Zero-shot (ZS), 4 models × 3 runs.
# API cache + checkpoint resume enabled by default (omit --no-save-cache).
# Re-run safely: interrupted runs auto-resume from partial JSON.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python3}"

RUN="${RUN:-}"          # optional: 1, 2, or 3
MODEL="${MODEL:-}"      # optional: gemini-2.5-flash-lite, claude-3-haiku-20240307, gpt-4o, qwen-plus

run_zs() {
  local model="$1" run="$2"
  local slug="${model//\//_}"
  slug="${slug//./-}"
  export MARC_CACHE_DIR="${ROOT}/results/cache/zs/${slug}/run${run}"
  echo "==> ZS model=${model} run=${run} cache=${MARC_CACHE_DIR}"
  if [[ "$model" == "qwen-plus" ]]; then
    "$PY" src/experiments/run_qwen_zs_full.py --run "$run"
  else
    "$PY" src/experiments/run_zs_full.py --model "$model" --run "$run"
  fi
}

MODELS=(gemini-2.5-flash-lite claude-3-haiku-20240307 gpt-4o qwen-plus)
RUNS=(1 2 3)

if [[ -n "$RUN" ]]; then RUNS=("$RUN"); fi
if [[ -n "$MODEL" ]]; then MODELS=("$MODEL"); fi

for run in "${RUNS[@]}"; do
  for model in "${MODELS[@]}"; do
    run_zs "$model" "$run"
  done
done

echo "Done: ZS inference. Outputs under results/zs/"
