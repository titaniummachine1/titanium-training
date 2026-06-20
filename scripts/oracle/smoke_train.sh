#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python3}"
if [[ -d .venv ]]; then source .venv/bin/activate; fi
"$PY" training/nnue_cli.py smoke --config training/configs/smoke.yaml
