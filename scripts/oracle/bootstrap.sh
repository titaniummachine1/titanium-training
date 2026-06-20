#!/usr/bin/env bash
# Bootstrap a fresh Oracle (Linux/ARM) machine for Titanium value-NNUE training.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

echo "=== Titanium Oracle bootstrap ==="
echo "Root: $ROOT"
echo "OS: $(uname -s)  Arch: $(uname -m)"
echo "CPU: $(nproc 2>/dev/null || echo '?') cores"
echo "RAM: $(awk '/MemTotal/ {print $2/1024 " MB"}' /proc/meminfo 2>/dev/null || echo 'unknown')"
echo "Disk free: $(df -h . | tail -1 | awk '{print $4}')"

need_gb=50
free_kb=$(df -k . | tail -1 | awk '{print $4}')
if [[ "${free_kb:-0}" -lt $((need_gb * 1024 * 1024)) ]]; then
  echo "WARN: less than ${need_gb}GB free on $(pwd)" >&2
fi

PY="${PYTHON:-python3}"
ver=$("$PY" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Python: $ver"
case "$ver" in
  3.11|3.12) ;;
  *) echo "WARN: supported Python is 3.11 or 3.12" >&2 ;;
esac

if [[ ! -d .venv ]]; then
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip
pip install -r training/requirements.txt
pip install -r training/requirements-teacher-dataset.txt

mkdir -p training/runs training/checkpoints training/logs

if [[ -f training/data/teacher_dataset/manifest.json ]]; then
  echo "Active teacher dataset: present"
else
  echo "NOTE: teacher dataset not present — transfer bundle with --include-active-dataset or copy separately"
fi

echo "Bootstrap complete. Next: bash scripts/oracle/doctor.sh"
