"""Checkpoint read/write/validate for resumable feature-cache rebuilds."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CHECKPOINT_NAME = "rebuild_checkpoint.json"

DETERMINISTIC_STDERR_MARKERS = (
    "TypeError",
    "NameError",
    "AssertionError",
    "schema mismatch",
    "parity validation failure",
    "row_delta_integrity",
    "row-integrity failure",
    "cannot unpack non-iterable",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def checkpoint_path(cache_dir: Path) -> Path:
    return cache_dir / CHECKPOINT_NAME


def file_fingerprint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"exists": False, "size": 0, "sha256": None}
    data = path.read_bytes()
    return {
        "exists": True,
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def positions_fingerprint(cache_dir: Path, expected_rows: int, fv_len: int) -> dict[str, Any]:
    p = cache_dir / "positions.bin"
    if not p.is_file():
        return {"exists": False, "size": 0, "rows": 0}
    size = p.stat().st_size
    return {"exists": True, "size": size, "rows": size // (fv_len * 4) if fv_len else 0}


@dataclass
class RebuildCheckpoint:
    dataset_fingerprint: str
    fv_len: int
    expected_rows: int
    last_completed_batch: int
    next_row: int
    batch_size: int
    rows_written: int
    rows_failed: int
    state: str
    updated_at: str
    positions: dict[str, Any]
    packed_specs: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_fingerprint": self.dataset_fingerprint,
            "fv_len": self.fv_len,
            "expected_rows": self.expected_rows,
            "last_completed_batch": self.last_completed_batch,
            "next_row": self.next_row,
            "batch_size": self.batch_size,
            "rows_written": self.rows_written,
            "rows_failed": self.rows_failed,
            "state": self.state,
            "updated_at": self.updated_at,
            "positions": self.positions,
            "packed_specs": self.packed_specs,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RebuildCheckpoint:
        return cls(
            dataset_fingerprint=str(d["dataset_fingerprint"]),
            fv_len=int(d["fv_len"]),
            expected_rows=int(d["expected_rows"]),
            last_completed_batch=int(d["last_completed_batch"]),
            next_row=int(d["next_row"]),
            batch_size=int(d["batch_size"]),
            rows_written=int(d.get("rows_written", 0)),
            rows_failed=int(d.get("rows_failed", 0)),
            state=str(d.get("state", "BUILD_RUNNING")),
            updated_at=str(d.get("updated_at", "")),
            positions=dict(d.get("positions") or {}),
            packed_specs=dict(d.get("packed_specs") or {}),
        )


def read_checkpoint(cache_dir: Path) -> RebuildCheckpoint | None:
    path = checkpoint_path(cache_dir)
    if not path.is_file():
        return None
    try:
        return RebuildCheckpoint.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def write_checkpoint_atomic(cache_dir: Path, cp: RebuildCheckpoint) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cp.updated_at = _utc_now()
    path = checkpoint_path(cache_dir)
    tmp = path.with_suffix(".json.tmp")
    payload = json.dumps(cp.to_dict(), indent=2) + "\n"
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)
    if sys.platform != "win32":
        with open(path, "rb") as f:
            os.fsync(f.fileno())


def validate_checkpoint_for_resume(
    cache_dir: Path,
    *,
    dataset_fingerprint: str,
    fv_len: int,
    expected_rows: int,
    batch_size: int,
) -> tuple[bool, str, RebuildCheckpoint | None]:
    cp = read_checkpoint(cache_dir)
    if cp is None:
        return False, "no checkpoint", None
    if cp.dataset_fingerprint != dataset_fingerprint:
        return False, "dataset_fingerprint mismatch", cp
    if cp.fv_len != fv_len:
        return False, "fv_len mismatch", cp
    if cp.expected_rows != expected_rows:
        return False, "expected_rows mismatch", cp
    if cp.batch_size != batch_size:
        return False, "batch_size mismatch", cp
    pos = positions_fingerprint(cache_dir, expected_rows, fv_len)
    if int(pos.get("rows", 0)) != expected_rows and pos.get("exists"):
        return False, f"positions.bin row count {pos.get('rows')} != {expected_rows}", cp
    if cp.positions and int(cp.positions.get("rows", -1)) != int(pos.get("rows", -2)):
        return False, "positions.bin size drift", cp
    packed = cache_dir / "featurize_packed.bin"
    if not packed.is_file():
        return False, "featurize_packed.bin missing", cp
    if cp.packed_specs and int(cp.packed_specs.get("size", -1)) != packed.stat().st_size:
        return False, "packed specs size mismatch", cp
    expected_next = (cp.last_completed_batch + 1) * batch_size
    if cp.next_row != expected_next and cp.last_completed_batch >= 0:
        return False, f"next_row {cp.next_row} != expected {expected_next}", cp
    return True, "ok", cp


def stderr_is_deterministic(stderr_text: str) -> bool:
    text = stderr_text or ""
    return any(m in text for m in DETERMINISTIC_STDERR_MARKERS)
