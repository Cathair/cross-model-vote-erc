#!/usr/bin/env bash
# Deprecated wrapper — use scripts/phase2/run_all.sh --all instead.
set -euo pipefail
exec bash "$(dirname "$0")/phase2/run_all.sh" "$@"
