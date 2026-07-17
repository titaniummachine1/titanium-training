#!/usr/bin/env python3
"""Preflight before NNUE training — binary, parity, eval-batch, packed-state path."""
from __future__ import annotations

import json
import struct
import subprocess
import sys
from pathlib import Path

from titanium_training.paths import REPO_ROOT, TRAINING_ROOT
from titanium_training.store.state import PositionState

ROOT = REPO_ROOT
BIN = REPO_ROOT / "engine" / "target" / "release" / "titanium.exe"
PACKED_RECORD = struct.Struct("<I24s")


def fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    raise SystemExit(1)


def pass_(msg: str) -> None:
    print(f"PASS: {msg}")


def _check_eval_packed_batch() -> None:
    packed = PositionState.initial().packed_state()
    payload = PACKED_RECORD.pack(0, packed)
    batch = subprocess.run(
        [str(BIN), "eval-packed-batch"],
        input=payload,
        capture_output=True,
        timeout=60,
        check=True,
    )
    lines = [ln for ln in batch.stdout.decode().splitlines() if ln.strip()]
    if not lines:
        fail("eval-packed-batch returned no JSON lines")
    rec = json.loads(lines[0])
    if not rec.get("ok"):
        fail(f"eval-packed-batch failed: {rec.get('error')}")
    if "legal_wall_count" not in rec:
        fail("eval-packed-batch record missing legal_wall_count")
    if "legal_path_cross_p0" not in rec or "legal_path_cross_p1" not in rec:
        fail("eval-packed-batch record missing legal_path_cross_p0/p1 (rebuild engine)")
    if "cat_best_p0" not in rec or "cat_best_p1" not in rec:
        fail("eval-packed-batch record missing cat_best_p0/p1 (rebuild engine for ws20 CAT)")
    pass_(
        f"eval-packed-batch legal_wall_count={rec['legal_wall_count']} "
        f"legal_path_cross=({rec['legal_path_cross_p0']},{rec['legal_path_cross_p1']}) "
        f"cat_best=({rec['cat_best_p0']},{rec['cat_best_p1']})"
    )


def _readiness_level() -> str:
    try:
        from titanium_training.data.teacher_value import scan_packed_state_coverage
        from titanium_training.paths import ACTIVE_TEACHER_DATASET

        if not (ACTIVE_TEACHER_DATASET / "manifest.json").is_file():
            return "SMOKE READY (dataset absent locally)"
        stats = scan_packed_state_coverage(ACTIVE_TEACHER_DATASET, max_scan=2_048, batch_size=256)
        cov = float(stats.get("coverage_percentage", 0))
        if cov >= 99.9:
            return "FULL-CORPUS READY (sampled coverage >= 99.9%)"
        return f"SMOKE READY (sampled coverage {cov:.2f}%)"
    except Exception as e:
        return f"SMOKE READY (coverage probe skipped: {e})"


def main() -> int:
    if not BIN.exists():
        fail(f"missing binary: {BIN}")
    pass_(f"binary exists ({BIN.name})")

    parity = subprocess.run(
        [sys.executable, str(TRAINING_ROOT / "titanium_training" / "validation" / "parity_check.py")],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    if parity.stdout:
        print(parity.stdout.rstrip())
    if parity.stderr:
        print(parity.stderr.rstrip(), file=sys.stderr)
    if parity.returncode != 0:
        fail("parity_check.py failed")
    pass_("parity_check.py passed")

    try:
        batch = subprocess.run(
            [str(BIN), "eval-batch"],
            input="\n",
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        fail(f"eval-batch failed (exit {e.returncode}): {(e.stderr or '')[:400]}")
    lines = [ln for ln in batch.stdout.splitlines() if ln.strip()]
    if not lines:
        fail("eval-batch returned no JSON lines")
    rec = json.loads(lines[0])
    if "legal_wall_count" not in rec:
        fail("eval-batch record missing legal_wall_count")
    if "legal_path_cross_p0" not in rec or "legal_path_cross_p1" not in rec:
        fail("eval-batch record missing legal_path_cross_p0/p1 (rebuild engine)")
    pass_(f"eval-batch legal_wall_count={rec['legal_wall_count']} legal_path_cross=({rec['legal_path_cross_p0']},{rec['legal_path_cross_p1']})")

    _check_eval_packed_batch()

    level = _readiness_level()
    print(f"\nREADINESS: {level}")
    if "FULL-CORPUS READY" in level:
        print("\nREADY: full-corpus value-NNUE preflight passed")
    else:
        print("\nREADY: smoke / bounded teacher preflight passed (not full-corpus)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
