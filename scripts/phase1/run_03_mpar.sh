#!/usr/bin/env bash
# Phase 1 — Experiment 1c: MPAR (3 runs). Auto-resumes when checkpoints exist.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python3}"

RUN="${RUN:-}"  # optional: 1, 2, or 3

run_mpar() {
  local run="$1"
  export MARC_CACHE_DIR="${ROOT}/results/cache/mpar/run${run}"
  export MARC_V37_OUT_DIR="${ROOT}/results/mpar/run${run}"
  local -a extra=()
  if compgen -G "${MARC_V37_OUT_DIR}/checkpoint_v37_*_at_*.json" > /dev/null; then
    extra+=(--resume)
    echo "==> MPAR run=${run}: checkpoint found, resuming"
  fi
  echo "==> MPAR run=${run} out=${MARC_V37_OUT_DIR} cache=${MARC_CACHE_DIR}"
  "$PY" src/experiments/run_mpar_full.py "${extra[@]}"
}

RUNS=(1 2 3)
if [[ -n "$RUN" ]]; then RUNS=("$RUN"); fi

for run in "${RUNS[@]}"; do
  run_mpar "$run"
done

echo "Done: MPAR inference. Outputs under results/mpar/"
