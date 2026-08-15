#!/usr/bin/env bash
# Phase 2 — Fig. ZS vs SA + table_rq1_zs_sa (prerequisite for Table I).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
"${PYTHON:-python3}" src/analysis/plot_zs_sa.py
