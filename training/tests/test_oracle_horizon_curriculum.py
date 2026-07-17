from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from training.oracle_horizon.bands import active_bands_for_pilot, assign_band
from training.oracle_horizon.cycle1_audit import audit_rows, classify_resolution, race_proof
from training.oracle_horizon.config import PilotConfig
from training.oracle_horizon.flywheel_pilot import write_dry_run
from training.oracle_horizon.label_classes import (
    LabelClass, assert_not_fake_exact, can_train_primary, sample_weight,
)
from training.oracle_horizon.needs_learning import needs_learning
from training.oracle_horizon.safety import should_pause
from training.oracle_horizon.trajectory_miner import reject_book_assisted


def test_label_classes_and_fake_exact():
    assert sample_weight(LabelClass.EXACT_ORACLE) == 1.0
    assert sample_weight("ORACLE_SUPPORTED_PARTIAL") == 0.25
    assert can_train_primary("ORACLE_BACKED_MINIMAX")
    assert not can_train_primary("SEARCH_ONLY")
    with pytest.raises(ValueError):
        assert_not_fake_exact({"label_class": "SEARCH_ONLY", "exact": True})
    with pytest.raises(ValueError):
        assert_not_fake_exact({"exact": True, "nnue_score": 9999})


def test_bands_and_pilot_scope():
    assert [assign_band(value) for value in (0, 1, 2, 3, 5, 9, 17)] == [0, 1, 1, 2, 3, 4, 5]
    assert active_bands_for_pilot() == {0, 1, 2, 3}


def test_needs_learning_triggers():
    needed, reasons = needs_learning(
        {"wdl": "W", "best_move": "a", "nodes_ratio": 20, "eval": 0, "move_flip": True},
        {"wdl": "L", "best_move": "b", "decisive": True, "only_defense": True},
    )
    assert needed
    assert {"wrong_wdl_sign", "wrong_move", "missed_only_defense", "nodes_ratio",
            "near_zero_eval_on_decisive", "move_flip"} <= set(reasons)


def test_safety_fail_closed():
    assert should_pause({"unknown_critical_flags": ["future_flag"]})[0]
    assert should_pause({"loss": float("nan")})[0]


def test_book_assisted_rejected():
    with pytest.raises(ValueError):
        reject_book_assisted({"book_move_used": True})


def test_dry_run_manifest_keys():
    with tempfile.TemporaryDirectory(prefix="oracle_horizon_") as temp_dir:
        out_dir = Path(temp_dir)
        result = write_dry_run(out_dir)
        manifest = json.loads((out_dir / "DESIGN_MANIFEST.json").read_text())
        cycle = json.loads((out_dir / "CYCLE0_DRY_RUN.json").read_text())
    assert {"loop", "label_classes", "bands", "search_ladder", "mix", "safety", "budgets"} <= manifest.keys()
    assert manifest["loop"] == {
        "A": "Generate",
        "B": "Mine",
        "C": "Relabel",
        "D": "Train",
        "E": "Cheap screen",
        "F": "Full gate",
        "G": "Accept or quarantine",
        "H": "Staleness audit",
    }
    assert cycle["unattended"] is False
    assert cycle["book_off"] is True
    assert cycle["unresolved_cannot_be_exact"] is True
    assert result["started_training"] is False


def test_unattended_default_false():
    assert PilotConfig().unattended_repeat is False


def test_race_band_is_not_exact_without_proof():
    assert race_proof(31_000, False)
    assert not race_proof(31_000, True)
    assert classify_resolution(31_999, False) == "SEARCH_ONLY"
    assert classify_resolution(31_999, True) == "EXACT_ORACLE"


def test_cycle1_audit_rejects_mock_score_out_race_claim():
    row = {
        "weights_sha256": "w",
        "engine_sha256": "e",
        "game_id": "g0",
        "lineage_id": "l0",
        "packed_state_hex": "00",
        "book_move_used": False,
        "evaluation_only": False,
        "oracle_proven": False,
        "score": 31_500,
        "label_class": "EXACT_ORACLE",
        "primary": True,
        "oracle_wdl": "W",
    }
    report = audit_rows([row], parent_weights_sha256="w", parent_engine_sha256="e", min_primary=0)
    assert report["status"] == "FAIL"
    assert any("race-band" in failure for failure in report["failures"])


def test_cycle1_audit_accepts_proven_mock_score_out():
    row = {
        "weights_sha256": "w",
        "engine_sha256": "e",
        "game_id": "g0",
        "lineage_id": "l0",
        "packed_state_hex": "01",
        "book_move_used": False,
        "evaluation_only": False,
        "oracle_proven": True,
        "score": 31_500,
        "label_class": "EXACT_ORACLE",
        "primary": True,
        "oracle_wdl": "W",
    }
    report = audit_rows([row], parent_weights_sha256="w", parent_engine_sha256="e", min_primary=0)
    assert report["status"] == "PASS"
