"""Tests for packed-state teacher featurization, splits, and export parity."""
from __future__ import annotations

import json
import struct
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TRAINING = ROOT / "training"
BIN = ROOT / "engine" / "target" / "release" / "titanium.exe"
PACKED_RECORD = struct.Struct("<I24s")


def test_packed_state_python_round_trip() -> None:
    from titanium_training.store.state import POSITION_SCHEMA_VERSION, PositionState

    state = PositionState.initial()
    packed = state.packed_state()
    assert len(packed) == 24
    assert packed[0] == POSITION_SCHEMA_VERSION
    restored = PositionState.unpack_state(packed)
    assert restored.packed_state() == packed


def test_packed_state_malformed_rejected() -> None:
    from titanium_training.store.state import PositionState

    with pytest.raises(ValueError):
        PositionState.unpack_state(b"\x00" * 8)
    bad = bytearray(PositionState.initial().packed_state())
    bad[0] = 2
    with pytest.raises(ValueError):
        PositionState.unpack_state(bytes(bad))


@pytest.mark.skipif(not BIN.is_file(), reason="titanium.exe not built")
def test_engine_eval_packed_batch_startpos() -> None:
    from titanium_training.store.state import PositionState

    packed = PositionState.initial().packed_state()
    proc = subprocess.run(
        [str(BIN), "eval-packed-batch"],
        input=PACKED_RECORD.pack(0, packed),
        capture_output=True,
        timeout=60,
        check=True,
    )
    rec = json.loads(proc.stdout.decode().splitlines()[0])
    assert rec["ok"] is True
    assert rec["legal_wall_count"] == 128
    assert rec["feature_schema"] == "halfpw-sparse-route5-ws14-v1"


@pytest.mark.skipif(not BIN.is_file(), reason="titanium.exe not built")
def test_move_prefix_vs_packed_eval_equivalence() -> None:
    from tools.datagen.datagen import eval_batch
    from titanium_training.data.eval_packed import eval_packed_batch
    from titanium_training.store.state import PositionState

    # Positions replayed through standard Ace algebraic (same physical board as Python pack).
    moves_list = [
        [],
        ["e2", "e8", "e3", "e7", "d3h", "f5v"],
        ["e2", "e8", "e3", "e7", "e4", "e6", "a3h", "d4v"],
    ]
    for moves in moves_list:
        prefix_eval = eval_batch([moves])[0]
        # Reconstruct Python packed state from Ace eval geometry (inverse of packed decode map).
        packed = bytearray(24)
        packed[0] = 1
        packed[1] = prefix_eval["pawn1"]
        packed[2] = prefix_eval["pawn0"]
        packed[3] = prefix_eval["wl1"]
        packed[4] = prefix_eval["wl0"]
        packed[5] = 0 if prefix_eval["turn"] == 1 else 1
        hw = prefix_eval["hw"]
        vw = prefix_eval["vw"]
        hw_mask = sum((1 << i) for i, b in enumerate(hw) if str(b) == "1" or b == 1)
        vw_mask = sum((1 << i) for i, b in enumerate(vw) if str(b) == "1" or b == 1)
        packed[8:16] = int(hw_mask).to_bytes(8, "little")
        packed[16:24] = int(vw_mask).to_bytes(8, "little")
        packed_eval = eval_packed_batch([(0, bytes(packed))])[0]
        assert packed_eval["eval"] == prefix_eval["eval"]
        assert packed_eval["pawn0"] == prefix_eval["pawn0"]
        assert packed_eval["pawn1"] == prefix_eval["pawn1"]
        assert int(packed_eval["turn"]) == int(prefix_eval["turn"])


def test_deterministic_split_non_empty() -> None:
    from titanium_training.data.split import deterministic_train_val_split

    records = [{"_position_key": f"k{i}".encode()} for i in range(200)]
    train, val, meta = deterministic_train_val_split(
        records, val_fraction=0.1, seed=7, min_val=10, min_train=10
    )
    assert len(train) >= 10
    assert len(val) >= 10
    assert meta["validation_count"] == len(val)


def test_zero_validation_rejected() -> None:
    from titanium_training.data.split import ValidationSplitError, deterministic_train_val_split

    records = [{"_position_key": b"a"}]
    with pytest.raises(ValidationSplitError):
        deterministic_train_val_split(records, val_fraction=0.1, seed=0, min_val=1, min_train=1)


def test_teacher_manifest_binding() -> None:
    from titanium_training.data.teacher_value import load_manifest, verify_manifest_identity
    from titanium_training.paths import ACTIVE_TEACHER_DATASET

    if not ACTIVE_TEACHER_DATASET.is_file():
        pytest.skip("active dataset not present")
    manifest = load_manifest(ACTIVE_TEACHER_DATASET)
    verify_manifest_identity(manifest)


@pytest.mark.integration
def test_teacher_packed_featurization_nonzero() -> None:
    from titanium_training.data.teacher_value import load_teacher_value_training_records
    from titanium_training.paths import ACTIVE_TEACHER_DATASET

    if not (ACTIVE_TEACHER_DATASET / "manifest.json").is_file():
        pytest.skip("active dataset not present")
    if not BIN.is_file():
        pytest.skip("titanium.exe not built")
    records, meta = load_teacher_value_training_records(
        ACTIVE_TEACHER_DATASET,
        max_samples=32,
        min_samples=8,
        seed=1,
        coverage_min=1.0,
    )
    assert len(records) >= 8
    assert meta["synthetic_fallback_used"] is False
    assert meta["featurization_mode"] == "packed-state-direct"
    assert meta["coverage_percentage"] == 100.0


def test_split_module_import() -> None:
    from titanium_training.data import split as split_mod

    assert split_mod.SPLIT_ALGORITHM
