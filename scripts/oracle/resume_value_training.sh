#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python3}"
if [[ -d .venv ]]; then source .venv/bin/activate; fi
exec "$PY" training/nnue_cli.py resume --checkpoint "$1"
