"""Unit tests for the strict hands-empty packed-oracle bridge."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

TRAINING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING))

from titanium_training.data.hands_empty_oracle import (  # noqa: E402
    HandsEmptyOracleResult,
    PACKED_RECORD,
    PROTOCOL,
    oracle_packed_batch,
)


def _response(row: int, **overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "row": row,
        "ok": True,
        "protocol": PROTOCOL,
        "eligible": True,
        "exact_value_plies": 7,
        "value_stm": 1,
        "distance_plies": 7,
    }
    record.update(overrides)
    return record


def _run_with(stdout_records: list[dict[str, object]], items: list[tuple[int, bytes]]):
    completed = subprocess.CompletedProcess(
        args=["titanium", "oracle-packed-batch"],
        returncode=0,
        stdout=("\n".join(json.dumps(record) for record in stdout_records) + "\n").encode(),
        stderr=b"",
    )
    with patch(
        "titanium_training.data.hands_empty_oracle.assert_engine_ready"
    ), patch(
        "titanium_training.data.hands_empty_oracle.subprocess.run", return_value=completed
    ) as run:
        result = oracle_packed_batch(items, timeout_sec=1)
    return result, run


def test_oracle_packed_batch_serializes_and_returns_typed_ordered_results():
    items = [(4, b"a" * 24), (99, b"b" * 24)]
    result, run = _run_with(
        [_response(4), _response(99, exact_value_plies=-3, value_stm=-1, distance_plies=3)],
        items,
    )

    assert result == [
        HandsEmptyOracleResult(4, True, 7, 1, 7),
        HandsEmptyOracleResult(99, True, -3, -1, 3),
    ]
    assert run.call_args.args[0][1] == "oracle-packed-batch"
    assert run.call_args.kwargs["input"] == PACKED_RECORD.pack(4, b"a" * 24) + PACKED_RECORD.pack(
        99, b"b" * 24
    )


def test_oracle_packed_batch_rejects_reordered_rows():
    with pytest.raises(RuntimeError, match="row order mismatch"):
        _run_with([_response(99), _response(4)], [(4, b"a" * 24), (99, b"b" * 24)])


@pytest.mark.parametrize(
    ("record", "message"),
    [
        (_response(4, eligible=False), "not exact-label eligible"),
        (_response(4, exact_value_plies=-2, value_stm=1), "disagrees"),
        (_response(4, value_stm=2), "must be -1, 0, or 1"),
        (_response(4, distance_plies=-1), "must be non-negative"),
    ],
)
def test_oracle_packed_batch_rejects_invalid_exact_records(record, message):
    with pytest.raises(RuntimeError, match=message):
        _run_with([record], [(4, b"a" * 24)])


def test_noneligible_record_is_available_only_when_explicitly_requested():
    completed = subprocess.CompletedProcess(
        args=["titanium", "oracle-packed-batch"],
        returncode=0,
        stdout=(json.dumps(_response(4, eligible=False)) + "\n").encode(),
        stderr=b"",
    )
    with patch("titanium_training.data.hands_empty_oracle.assert_engine_ready"), patch(
        "titanium_training.data.hands_empty_oracle.subprocess.run", return_value=completed
    ):
        result = oracle_packed_batch([(4, b"a" * 24)], require_exact_labels=False, timeout_sec=1)
    assert result == [HandsEmptyOracleResult(4, False, None, None, None)]


def test_oracle_packed_batch_rejects_bad_input_before_invoking_engine():
    with pytest.raises(ValueError, match="24 bytes"):
        oracle_packed_batch([(4, b"too short")])
