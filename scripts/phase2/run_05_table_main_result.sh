#!/usr/bin/env bash
# Phase 2 — Table I main results (requires run_01 + run_02 + Phase 1 MPAR/InsideOut).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
"${PYTHON:-python3}" src/analysis/table_paper_main.py
