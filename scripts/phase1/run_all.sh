#!/usr/bin/env bash
# Phase 1 index — run experiments individually (recommended).
# Do NOT run all at once unless you intend full API reproduction.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
echo "Phase 1 experiments (run separately):"
echo "  bash scripts/phase1/run_01_zs.sh         # ZS: 4 models × 3 runs"
echo "  bash scripts/phase1/run_02_sa.sh         # SA: 4 models × run1"
echo "  bash scripts/phase1/run_03_mpar.sh       # MPAR: 3 runs"
echo "  bash scripts/phase1/run_04_insideout.sh  # InsideOut: 4 models × run1"
echo ""
echo "Optional env filters: RUN=2 MODEL=gpt-4o bash scripts/phase1/run_01_zs.sh"
echo ""
if [[ "${1:-}" == "--all" ]]; then
  bash "$DIR/run_01_zs.sh"
  bash "$DIR/run_02_sa.sh"
  bash "$DIR/run_03_mpar.sh"
  bash "$DIR/run_04_insideout.sh"
else
  echo "Pass --all to execute every Phase 1 script sequentially."
fi
