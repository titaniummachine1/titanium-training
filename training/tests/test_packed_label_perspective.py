#!/usr/bin/env python3
"""Tests for dataset-STM unchanged label perspective on packed rows."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

TRAINING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING))

from label_perspective import (
    LABEL_PERSPECTIVE_CONVENTION,
    dataset_stm_to_outcome_p0,
    json_row_target_prob,
    packed_row_outcome_p0,
    packed_row_target_prob,
    stm_to_target_prob,
    value_i16_to_dataset_stm,
)
from streaming_db_loader import _featurize_records, LabeledPosition
from titanium_training.data.eval_packed import eval_packed_batch_allow_errors
from titanium_training.paths import ENGINE_BIN


def test_convention_is_dataset_stm_unchanged():
    assert LABEL_PERSPECTIVE_CONVENTION == "dataset_stm_unchanged_v1"


def test_packed_row_keeps_dataset_label_when_turn_flipped():
    v = 0.8
    target = packed_row_target_prob(
        value_dataset_stm=v, engine_turn=1, dataset_side_to_move=0
    )
    assert target == pytest.approx(stm_to_target_prob(v))
    assert target != pytest.approx(stm_to_target_prob(-v))


def test_packed_row_keeps_dataset_label_stm1_engine_turn0():
    v = -0.4
    target = packed_row_target_prob(
        value_dataset_stm=v, engine_turn=0, dataset_side_to_move=1
    )
    assert target == pytest.approx(stm_to_target_prob(v))


def test_json_row_unchanged_when_turns_match():
    for stm in (0, 1):
        v = 0.25
        assert json_row_target_prob(v) == stm_to_target_prob(v)
    assert json_row_target_prob(1.0) == 1.0
    assert json_row_target_prob(-1.0) == 0.0


def test_outcome_p0_from_dataset_stm():
    assert dataset_stm_to_outcome_p0(0.6, 0) == pytest.approx(0.6)
    assert dataset_stm_to_outcome_p0(0.6, 1) == pytest.approx(-0.6)
    assert packed_row_outcome_p0(value_dataset_stm=0.6, engine_turn=1, dataset_side_to_move=0) == 0.6


def test_player_swap_complements_target():
    v = 0.6
    t0 = packed_row_target_prob(value_dataset_stm=v, engine_turn=1, dataset_side_to_move=0)
    t1 = packed_row_target_prob(value_dataset_stm=-v, engine_turn=0, dataset_side_to_move=1)
    assert t0 + t1 == pytest.approx(1.0)


def test_board_mirror_preserves_target():
    v = 0.35
    t = packed_row_target_prob(value_dataset_stm=v, engine_turn=1, dataset_side_to_move=0)
    assert t == packed_row_target_prob(value_dataset_stm=v, engine_turn=1, dataset_side_to_move=0)


@pytest.mark.skipif(not ENGINE_BIN.is_file(), reason="engine binary missing")
def test_packed_featurize_uses_engine_turn_features():
    labels_db = TRAINING / "data" / "canonical" / "labels.db"
    if not labels_db.is_file():
        pytest.skip("labels.db missing")
    import sqlite3

    con = sqlite3.connect(str(labels_db))
    row = con.execute(
        """
        SELECT p.position_key, p.packed_state, p.side_to_move, l.value_i16
        FROM teacher_positions p
        JOIN teacher_labels l ON l.position_key = p.position_key
        WHERE l.value_i16 IS NOT NULL
        LIMIT 1
        """
    ).fetchone()
    con.close()
    if not row:
        pytest.skip("no teacher rows")
    position_key, packed_state, dataset_stm, value_i16 = row
    value_dataset = value_i16_to_dataset_stm(int(value_i16))
    labeled = LabeledPosition(
        position_id=f"teacher:{bytes(position_key).hex()}",
        packed_state=bytes(packed_state),
        value_target=0.0,
        sample_weight=1.0,
        storage_kind="packed",
        dataset_side_to_move=int(dataset_stm),
        value_dataset_stm=value_dataset,
    )
    ids, features, targets, *_ = _featurize_records([labeled])
    assert len(ids) == 1
    recs = eval_packed_batch_allow_errors([(0, bytes(packed_state))])
    assert recs[0].get("ok")
    expected = packed_row_target_prob(
        value_dataset_stm=value_dataset,
        engine_turn=int(recs[0]["turn"]),
        dataset_side_to_move=int(dataset_stm),
    )
    assert targets[0] == pytest.approx(expected)
    assert int(recs[0]["turn"]) in (0, 1)


@pytest.mark.skipif(not ENGINE_BIN.is_file(), reason="engine binary missing")
def test_packed_label_target_matches_loader_featurize():
    labels_db = TRAINING / "data" / "canonical" / "labels.db"
    if not labels_db.is_file():
        pytest.skip("labels.db missing")
    import sqlite3

    con = sqlite3.connect(str(labels_db))
    rows = con.execute(
        """
        SELECT p.packed_state, p.side_to_move, l.value_i16
        FROM teacher_positions p
        JOIN teacher_labels l ON l.position_key = p.position_key
        WHERE l.value_i16 IS NOT NULL
        LIMIT 8
        """
    ).fetchall()
    con.close()
    if not rows:
        pytest.skip("no teacher rows")
    for packed_state, dataset_stm, value_i16 in rows:
        value_dataset = value_i16_to_dataset_stm(int(value_i16))
        labeled = LabeledPosition(
            position_id="teacher:test",
            packed_state=bytes(packed_state),
            value_target=0.0,
            sample_weight=1.0,
            storage_kind="packed",
            dataset_side_to_move=int(dataset_stm),
            value_dataset_stm=value_dataset,
        )
        _ids, _features, targets, *_ = _featurize_records([labeled])
        if not _ids:
            continue
        rec = eval_packed_batch_allow_errors([(0, bytes(packed_state))])[0]
        assert rec and rec.get("ok")
        expected = stm_to_target_prob(value_dataset)
        assert targets[0] == pytest.approx(expected)
        if int(rec["turn"]) != int(dataset_stm):
            assert expected == pytest.approx(stm_to_target_prob(value_dataset))
        break
