#!/usr/bin/env bash
# Phase 2 — Table: prompt ablation (MV-3-Phase0 vs MPAR-Phase0).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
"${PYTHON:-python3}" src/analysis/table_prompt_ablation.py
