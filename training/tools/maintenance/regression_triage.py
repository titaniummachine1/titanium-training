#!/usr/bin/env python3
"""Classify an engine strength drop before blaming NNUE training.

Exit codes:
  1 = eval mismatch / schema mismatch
  2 = search regression smoke failed
  3 = rollout distribution shift probe failed
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "engine" / "target" / "release" / "titanium.exe"


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, **kw)


def check_eval() -> None:
    parity = run([sys.executable, "training/titanium_training/validation/parity_check.py"], timeout=120)
    if parity.returncode != 0 or "6/6 match" not in parity.stdout:
        print("DIAGNOSIS: eval mismatch")
        print((parity.stdout + parity.stderr).strip())
        raise SystemExit(1)

    schema = subprocess.run(
        [str(BIN), "eval-batch"],
        input=b"\n",
        capture_output=True,
        check=True,
        timeout=30,
    )
    rec = json.loads(schema.stdout.decode("utf-8").splitlines()[0])
    if rec.get("legal_wall_count") != 0:
        print(f"DIAGNOSIS: eval mismatch (legal_wall_count={rec.get('legal_wall_count')!r})")
        raise SystemExit(1)
    print("eval: OK (parity 6/6, retired legal_wall_count is zero)")


def check_search() -> None:
    match = run([
        str(BIN), "match",
        "--a", "titanium-v15",
        "--b", "ace-v13-ti-pure",
        "--games", "4",
        "--time", "0.2",
        "--threads", "1",
        "--open", "2",
        "--no-early-stop",
    ], timeout=180)
    text = match.stdout + match.stderr
    m = re.search(r"A score ([0-9.]+)/([0-9]+)", text)
    if match.returncode != 0 or not m:
        print("DIAGNOSIS: search regression (match smoke failed to run)")
        print(text.strip())
        raise SystemExit(2)
    score = float(m.group(1))
    total = float(m.group(2))
    rate = score / total if total else 0.0
    print(f"search smoke: titanium-v15 scored {score:g}/{total:g} vs ti-pure")
    if rate < 0.35:
        print("DIAGNOSIS: search regression")
        raise SystemExit(2)


def check_rollout() -> None:
    rollout = run([
        str(BIN), "rollout",
        "e2", "e8", "e3", "e7", "e4", "e6", "d3h", "c6h",
        "--sims", "32",
        "--plies", "8",
        "--cmp-depth", "3",
        "--time", "0.2",
    ], timeout=120)
    text = rollout.stdout + rollout.stderr
    if rollout.returncode != 0:
        print("DIAGNOSIS: rollout distribution shift")
        print(text.strip())
        raise SystemExit(3)
    if "top1=" in text and "top1=true" not in text and "top3=true" not in text:
        print("DIAGNOSIS: rollout distribution shift")
        print(text.strip())
        raise SystemExit(3)
    print("rollout: OK (probe completed)")


def main() -> None:
    check_eval()
    check_search()
    check_rollout()
    print("DIAGNOSIS: no eval/search/rollout smoke regression found")


if __name__ == "__main__":
    main()
