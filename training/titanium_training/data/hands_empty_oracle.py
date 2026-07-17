"""Strict client for exact hands-empty labels from Titanium.

The ``titanium oracle-packed-batch`` command consumes the canonical packed
wire format used by :mod:`eval_packed`: little-endian ``u32`` row followed by
the 24-byte state.  It produces one JSON object per input in exactly the same
order.  This module deliberately treats that row ordering as part of the
protocol: accepting reordered answers would silently attach a proven outcome
to the wrong position.

Eligible answers use ``exact_value_plies`` as the signed exact value from the
side to move.  ``value_stm`` is its sign (-1, 0, or 1) and
``distance_plies`` is a non-negative reported distance.  A non-eligible
answer may only be used by callers that explicitly opt out of exact labels.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import struct
import subprocess
from typing import Any, Mapping, Sequence

from titanium_training.paths import ENGINE_BIN, REPO_ROOT
from titanium_training.validation.engine_identity import assert_engine_ready


PACKED_STATE_LEN = 24
PACKED_RECORD = struct.Struct("<I24s")
PROTOCOL = "hands-empty-oracle-packed-v1"


@dataclass(frozen=True)
class HandsEmptyOracleResult:
    """One validated response from ``oracle-packed-batch``.

    Exact fields are ``None`` only for an explicitly permitted non-eligible
    record.  Consumers requesting training labels should use the default
    ``require_exact_labels=True`` and will therefore only receive exact rows.
    """

    row: int
    eligible: bool
    exact_value_plies: int | None
    value_stm: int | None
    distance_plies: int | None


def _packed_payload(items: Sequence[tuple[int, bytes]]) -> bytes:
    """Serialize canonical packed states, rejecting malformed input early."""
    out = bytearray()
    for row, packed in items:
        if isinstance(row, bool) or not isinstance(row, int) or not 0 <= row <= 0xFFFF_FFFF:
            raise ValueError(f"row must be a u32 integer, got {row!r}")
        blob = bytes(packed)
        if len(blob) != PACKED_STATE_LEN:
            raise ValueError(f"packed_state must be {PACKED_STATE_LEN} bytes, got {len(blob)}")
        out.extend(PACKED_RECORD.pack(row, blob))
    return bytes(out)


def _require_int(record: Mapping[str, Any], key: str) -> int:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"oracle-packed-batch row {record.get('row')!r}: {key} must be an integer")
    return value


def _sign(value: int) -> int:
    return (value > 0) - (value < 0)


def _parse_record(
    raw: Any,
    *,
    expected_row: int,
    require_exact_labels: bool,
) -> HandsEmptyOracleResult:
    if not isinstance(raw, dict):
        raise RuntimeError("oracle-packed-batch response must be a JSON object")
    row = _require_int(raw, "row")
    if row != expected_row:
        raise RuntimeError(
            f"oracle-packed-batch row order mismatch: expected {expected_row}, got {row}"
        )
    if raw.get("protocol") not in (None, PROTOCOL):
        raise RuntimeError(
            f"oracle-packed-batch row {row}: unexpected protocol {raw.get('protocol')!r}"
        )
    if raw.get("ok") is not True:
        detail = raw.get("error", "missing or false ok")
        raise RuntimeError(f"oracle-packed-batch row {row}: {detail}")
    eligible = raw.get("eligible")
    if not isinstance(eligible, bool):
        raise RuntimeError(f"oracle-packed-batch row {row}: eligible must be a boolean")
    if not eligible:
        if require_exact_labels:
            raise RuntimeError(f"oracle-packed-batch row {row}: position is not exact-label eligible")
        return HandsEmptyOracleResult(row, False, None, None, None)

    exact_value_plies = _require_int(raw, "exact_value_plies")
    value_stm = _require_int(raw, "value_stm")
    distance_plies = _require_int(raw, "distance_plies")
    if value_stm not in (-1, 0, 1):
        raise RuntimeError(f"oracle-packed-batch row {row}: value_stm must be -1, 0, or 1")
    if distance_plies < 0:
        raise RuntimeError(f"oracle-packed-batch row {row}: distance_plies must be non-negative")
    if value_stm != _sign(exact_value_plies):
        raise RuntimeError(
            f"oracle-packed-batch row {row}: value_stm={value_stm} disagrees with "
            f"exact_value_plies={exact_value_plies}"
        )
    return HandsEmptyOracleResult(
        row=row,
        eligible=True,
        exact_value_plies=exact_value_plies,
        value_stm=value_stm,
        distance_plies=distance_plies,
    )


def oracle_packed_batch(
    items: Sequence[tuple[int, bytes]],
    *,
    require_exact_labels: bool = True,
    timeout_sec: float | None = None,
) -> list[HandsEmptyOracleResult]:
    """Return strict, ordered hands-empty oracle answers for packed states.

    Set ``require_exact_labels=False`` only for coverage/audit callers that
    intentionally retain non-eligible positions.  Training callers should
    retain the default so no unknown state can enter an exact-label dataset.
    """
    if not items:
        return []
    payload = _packed_payload(items)
    assert_engine_ready(write_if_missing=False, parity=False)
    # Oracle solving is more expensive than feature dumping; this is a floor,
    # while still scaling with a caller's batch size.
    timeout = timeout_sec if timeout_sec is not None else max(30.0, len(items) * 0.25)
    proc = subprocess.run(
        [str(ENGINE_BIN), "oracle-packed-batch"],
        input=payload,
        capture_output=True,
        cwd=str(REPO_ROOT),
        timeout=timeout,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or b"").decode(errors="replace")[:500]
        raise RuntimeError(f"oracle-packed-batch exited {proc.returncode}: {detail}")

    lines = [line for line in proc.stdout.decode(errors="strict").splitlines() if line.strip()]
    if len(lines) != len(items):
        raise RuntimeError(
            f"oracle-packed-batch count mismatch: {len(lines)} lines vs {len(items)} inputs"
        )

    results: list[HandsEmptyOracleResult] = []
    for (expected_row, _packed), line in zip(items, lines, strict=True):
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"oracle-packed-batch invalid JSON for row {expected_row}: {exc.msg}") from exc
        results.append(
            _parse_record(
                raw,
                expected_row=expected_row,
                require_exact_labels=require_exact_labels,
            )
        )
    return results


__all__ = [
    "HandsEmptyOracleResult",
    "PACKED_RECORD",
    "PACKED_STATE_LEN",
    "PROTOCOL",
    "oracle_packed_batch",
]
