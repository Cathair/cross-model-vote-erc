#!/usr/bin/env bash
# Phase 2 — Fig. MV-K + Table II (MV-K summary).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
"${PYTHON:-python3}" src/analysis/plot_mv_k.py
