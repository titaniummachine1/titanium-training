"""Tests for checkpoint promotion flywheel metadata and prep guards."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_TRAINING = Path(__file__).resolve().parents[1]
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

from diversity.promotion_record import (
    PROMOTION_RECORD_VERSION,
    build_promotion_record,
    build_synthetic_validation_report,
    fixture_promotion_record,
)
from streaming_checkpoint_chain import accept_checkpoint, quarantine_checkpoint


def test_promotion_record_has_required_fields():
    rec = fixture_promotion_record(passed=True)
    assert rec.promotion_record_version == PROMOTION_RECORD_VERSION
    assert rec.epoch_id == 1
    assert rec.parent_accepted_epoch == 0
    assert len(rec.candidate_weights_sha256) == 64
    assert rec.code_promotion_separate is True
    assert rec.sign_test_result is not None
    assert not rec.validate()


def test_synthetic_validation_report_fixture():
    rep = build_synthetic_validation_report(passed=False)
    assert rep["passed"] is False
    assert rep["synthetic_only"] is True


def test_accept_checkpoint_refuses_under_prep(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINING_PREP_ONLY", "1")
    weights = tmp_path / "w.bin"
    weights.write_bytes(b"x" * 64)
    with pytest.raises(SystemExit) as exc:
        accept_checkpoint(
            weights_path=weights,
            epoch=99,
            validation=build_synthetic_validation_report(),
        )
    assert exc.value.code == 2


def test_accept_checkpoint_fixture_mode_records_promotion(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINING_PREP_ONLY", "1")
    chain = tmp_path / "chain.json"
    monkeypatch.setattr("streaming_checkpoint_chain.CHAIN_PATH", chain)
    monkeypatch.setattr("streaming_checkpoint_chain.ACCEPTED_DIR", tmp_path / "accepted")
    monkeypatch.setattr("streaming_checkpoint_chain.BEST_WEIGHTS", tmp_path / "best.bin")
    monkeypatch.setattr("streaming_checkpoint_chain.RUN_DIR", tmp_path)
    weights = tmp_path / "w.bin"
    weights.write_bytes(b"y" * 64)
    rec = fixture_promotion_record().to_dict()
    entry = accept_checkpoint(
        weights_path=weights,
        epoch=1,
        validation=build_synthetic_validation_report(),
        promotion_record=rec,
        fixture_mode=True,
    )
    assert entry["promotion_record"]["epoch_id"] == 1
    data = json.loads(chain.read_text(encoding="utf-8"))
    assert data["epochs"][0]["promotion_record"]["decision"] == "accepted"


def test_quarantine_checkpoint_refuses_under_prep(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINING_PREP_ONLY", "1")
    weights = tmp_path / "w.bin"
    weights.write_bytes(b"z" * 64)
    with pytest.raises(SystemExit) as exc:
        quarantine_checkpoint(weights_path=weights, reason="test")
    assert exc.value.code == 2


def test_build_promotion_record_from_validation():
    validation = build_synthetic_validation_report(passed=True)
    rec = build_promotion_record(
        epoch_id=5,
        candidate_weights_sha256="a" * 64,
        validation=validation,
        parent_accepted_epoch=4,
    )
    assert rec.match_results["match_vs_previous"]["passed"] is True
