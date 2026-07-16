"""Engine eval-packed-batch wrapper — canonical 24-byte packed states."""
from __future__ import annotations

import json
import struct
import subprocess
import time
from pathlib import Path
from typing import Any

from titanium_training.paths import ENGINE_BIN, REPO_ROOT
from titanium_training.validation.engine_identity import assert_engine_ready

PACKED_STATE_LEN = 24
PACKED_RECORD = struct.Struct("<I24s")
FEATURE_SCHEMA = "halfpw-sparse-route5-ws20-catv5-normalized5-v1"
PROTOCOL = "eval-packed-v1"


def _packed_payload(items: list[tuple[int, bytes]]) -> bytes:
    out = bytearray()
    for row, packed in items:
        blob = bytes(packed)
        if len(blob) != PACKED_STATE_LEN:
            raise ValueError(f"packed_state must be {PACKED_STATE_LEN} bytes, got {len(blob)}")
        out.extend(PACKED_RECORD.pack(row, blob))
    return bytes(out)


def eval_packed_batch(
    items: list[tuple[int, bytes]],
    *,
    timeout_sec: float | None = None,
) -> list[dict[str, Any]]:
    """Run packed states through `titanium eval-packed-batch`; one JSON dict per item."""
    if not items:
        return []
    assert_engine_ready(write_if_missing=False, parity=False)
    from tools.datagen.datagen import EVAL_BATCH_TIMEOUT_SEC, _eval_batch_lock

    payload = _packed_payload(items)
    per_row = max(30.0, len(items) * 0.05)
    timeout = timeout_sec or max(EVAL_BATCH_TIMEOUT_SEC, per_row)
    with _eval_batch_lock():
        proc = subprocess.run(
            [str(ENGINE_BIN), "eval-packed-batch"],
            input=payload,
            capture_output=True,
            cwd=str(REPO_ROOT),
            timeout=timeout,
        )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or b"").decode(errors="replace")[:500]
        raise RuntimeError(f"eval-packed-batch exited {proc.returncode}: {err}")
    lines = [ln for ln in proc.stdout.decode().splitlines() if ln.strip()]
    if len(lines) != len(items):
        raise RuntimeError(
            f"eval-packed-batch count mismatch: {len(lines)} lines vs {len(items)} inputs"
        )
    out: list[dict[str, Any]] = []
    for ln in lines:
        rec = json.loads(ln)
        if not rec.get("ok", False):
            raise RuntimeError(f"eval-packed-batch row {rec.get('row')}: {rec.get('error')}")
        out.append(rec)
    return out


def eval_packed_batch_allow_errors(
    items: list[tuple[int, bytes]],
    *,
    timeout_sec: float | None = None,
) -> list[dict[str, Any]]:
    """Like eval_packed_batch but preserves per-row ok/error without raising."""
    if not items:
        return []
    assert_engine_ready(write_if_missing=False, parity=False)
    from tools.datagen.datagen import EVAL_BATCH_TIMEOUT_SEC, _eval_batch_lock

    payload = _packed_payload(items)
    per_row = max(30.0, len(items) * 0.05)
    timeout = timeout_sec or max(EVAL_BATCH_TIMEOUT_SEC, per_row)
    t0 = time.perf_counter()
    with _eval_batch_lock():
        proc = subprocess.run(
            [str(ENGINE_BIN), "eval-packed-batch"],
            input=payload,
            capture_output=True,
            cwd=str(REPO_ROOT),
            timeout=timeout,
        )
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or b"").decode(errors="replace")[:500]
        raise RuntimeError(f"eval-packed-batch exited {proc.returncode}: {err}")
    lines = [ln for ln in proc.stdout.decode().splitlines() if ln.strip()]
    results = [json.loads(ln) for ln in lines]
    if len(results) != len(items):
        raise RuntimeError(
            f"eval-packed-batch count mismatch: {len(results)} vs {len(items)}"
        )
    return results
def eval_cat_packed_batch_allow_errors(
    items: list[tuple[int, bytes]],
    *,
    timeout_sec: float | None = None,
) -> list[dict[str, Any]]:
    """Extract CATv5 precise witnesses and propagated heat from packed states."""
    if not items:
        return []
    assert_engine_ready(write_if_missing=False, parity=False)
    from tools.datagen.datagen import EVAL_BATCH_TIMEOUT_SEC, _eval_batch_lock

    payload = _packed_payload(items)
    per_row = max(30.0, len(items) * 0.01)
    timeout = timeout_sec or max(EVAL_BATCH_TIMEOUT_SEC, per_row)
    with _eval_batch_lock():
        proc = subprocess.run(
            [str(ENGINE_BIN), "cat-packed-batch"],
            input=payload,
            capture_output=True,
            cwd=str(REPO_ROOT),
            timeout=timeout,
        )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or b"").decode(errors="replace")[:500]
        raise RuntimeError(f"cat-packed-batch exited {proc.returncode}: {err}")
    lines = [ln for ln in proc.stdout.decode().splitlines() if ln.strip()]
    results = [json.loads(ln) for ln in lines]
    if len(results) != len(items):
        raise RuntimeError(
            f"cat-packed-batch count mismatch: {len(results)} vs {len(items)}"
        )
    return results
