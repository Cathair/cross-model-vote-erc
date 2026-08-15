#!/usr/bin/env bash
# Phase 2 — Fig. within-model vs cross-model MV gain.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
"${PYTHON:-python3}" src/analysis/plot_within_vs_cross.py
