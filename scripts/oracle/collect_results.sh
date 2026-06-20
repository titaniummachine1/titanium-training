#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
RUN_DIR="${1:-training/runs/value_oracle}"
OUT="${2:-dist/oracle_results}"
mkdir -p "$OUT"
if [[ -d "$RUN_DIR" ]]; then
  cp -a "$RUN_DIR/." "$OUT/"
  echo "Collected run artifacts from $RUN_DIR -> $OUT"
else
  echo "Run directory not found: $RUN_DIR" >&2
  exit 1
fi
