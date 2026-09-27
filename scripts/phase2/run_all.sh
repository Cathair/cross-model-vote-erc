#!/usr/bin/env bash
# Phase 2 index — run analysis scripts one by one (recommended).
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$DIR/../.." && pwd)"
cd "$ROOT"

run() {
  echo "======== $(basename "$1") ========"
  bash "$1"
}

if [[ "${1:-}" == "--all" ]]; then
  run "$DIR/run_01_plot_zs_sa.sh"
  run "$DIR/run_02_plot_mv_k.sh"
  run "$DIR/run_03_plot_within_vs_cross.sh"
  run "$DIR/run_04_table_prompt_ablation.sh"
  run "$DIR/run_05_table_main_result.sh"
  echo "Phase 2 complete. See results/tables/ and results/figures/"
else
  cat <<'EOF'
Phase 2 — paper tables & figures (run separately after Phase 1):

  bash scripts/phase2/run_01_plot_zs_sa.sh
  bash scripts/phase2/run_02_plot_mv_k.sh
  bash scripts/phase2/run_03_plot_within_vs_cross.sh
  bash scripts/phase2/run_04_table_prompt_ablation.sh
  bash scripts/phase2/run_05_table_main_result.sh

Or: bash scripts/phase2/run_all.sh --all
EOF
fi
