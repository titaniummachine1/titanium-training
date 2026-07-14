"""Freeze hardening, semantics contract, dry-run determinism, canonical dedup."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

_TRAINING = Path(__file__).resolve().parents[1]
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

from diversity.canonical import CanonicalStateRow, deduplicate_finalized_rows
from diversity.certificate import (
    CertificateMeasurementContext,
    CertificateStatus,
    CorpusSampleRow,
    validate_full_certificate,
)
from diversity.lanes import DiversityLane
from diversity.planner import allocation_content_hash, default_prep_inputs
from engine_semantic_contract import (
    CompatibilityClass,
    EngineSemanticsContract,
    classify_compatibility,
    prep_placeholder_contract,
)
from game_opening_gate import (
    DEPLOY_COLLAPSE_OPENING,
    TEMPORARY_GARBAGE_FILTER_NOT_DIVERSITY_COMPLIANCE,
    contributes_to_n_eff_two_floor,
)
from launch_gate import validate_launch_gate
from prep_guard import DRY_RUN_LOG_DIR, validate_dry_run_output_path
from tools.audit_training_freeze import audit_freeze


def _row(**kwargs) -> CorpusSampleRow:
    defaults = dict(
        metadata_lane=DiversityLane.CLOSED_LOOP_POPULATION,
        source_game_id="g-001",
        fork_lineage_id=None,
        side_to_move=0,
        canonical=CanonicalStateRow("p", "h", "v", "10,10", 0),
    )
    defaults.update(kwargs)
    return CorpusSampleRow(**defaults)


def test_temporary_filter_not_n_eff_claim():
    assert TEMPORARY_GARBAGE_FILTER_NOT_DIVERSITY_COMPLIANCE is True
    assert contributes_to_n_eff_two_floor() is False


def test_deploy_trunk_not_training_requirement():
    assert DEPLOY_COLLAPSE_OPENING == ("e2", "e8", "e3", "e7")


def test_certificate_invalid_without_metadata():
    cert = validate_full_certificate([_row()])
    assert cert.status == CertificateStatus.INVALID


def test_certificate_invalid_never_pass_on_missing_prefix_trust():
    ctx = CertificateMeasurementContext(
        diversity_spec_version="DIVERSITY_SPEC_V1",
        engine_semantics_hash="abc123",
        corpus_generation_id="c-1",
        prefixes_trusted=False,
    )
    cert = validate_full_certificate([_row()], prefixes=[("e2", "e8")], measurement_context=ctx)
    assert cert.status == CertificateStatus.INVALID


def test_canonical_key_reflects_stm_and_stocks():
    a = CanonicalStateRow("p", "h", "v", "10,10", 0)
    b = CanonicalStateRow("p", "h", "v", "10,10", 1)
    assert a.canonical_key() != b.canonical_key()


def test_canonical_dedup_cross_lane():
    rows = [
        CanonicalStateRow("p", "h1", "v1", "10,10", 0),
        CanonicalStateRow("p", "h1", "v1", "10,10", 0),
    ]
    unique, dupes = deduplicate_finalized_rows(rows)
    assert dupes == 1 and len(unique) == 1


def test_planner_deterministic_hash():
    a = default_prep_inputs(corpus_generation_id="c", row_count=1000, planner_seed=7)
    b = default_prep_inputs(corpus_generation_id="c", row_count=1000, planner_seed=7)
    assert allocation_content_hash(a) == allocation_content_hash(b)
    c = default_prep_inputs(corpus_generation_id="c", row_count=1001, planner_seed=7)
    assert allocation_content_hash(a) != allocation_content_hash(c)


def test_semantic_compatibility_classes():
    left = prep_placeholder_contract()
    right = prep_placeholder_contract()
    assert classify_compatibility(left, right) == CompatibilityClass.INVALID
    good = EngineSemanticsContract(
        engine_semantic_version="v1",
        game_rules_version="r1",
        canonical_state_version="c1",
        move_encoding_version="m1",
        nnue_feature_schema_version="n1",
        evaluation_semantics_version="e1",
        score_band_version="s1",
        oracle_semantics_version="o1",
        search_label_semantics_version="l1",
        zobrist_version="z1",
        binary_sha256="a" * 64,
        source_commit="commit",
        generated_at="2026-01-01T00:00:00Z",
    )
    assert classify_compatibility(good, good) == CompatibilityClass.COMPATIBLE
    from dataclasses import replace

    left_prefix = replace(good, prefix_metric_version="prefix-metric-v1")
    right_prefix = replace(good, prefix_metric_version="prefix-metric-v2")
    assert classify_compatibility(left_prefix, right_prefix) == CompatibilityClass.RELABEL_REQUIRED


def test_launch_gate_rejects_stale_approval():
    from prep_guard import DRY_RUN_LOG_DIR

    approval = DRY_RUN_LOG_DIR / "_test_approval_stale.json"
    approval.write_text(
        json.dumps(
            {
                "corpus_generation_id": "old",
                "engine_semantics_hash": "dead",
                "generation_config_hash": "dead",
                "label_config_hash": "dead",
                "diversity_spec_version": "DIVERSITY_SPEC_V1",
                "prefix_metric_version": "stale",
                "eval_denylist_hash": "dead",
            }
        ),
        encoding="utf-8",
    )
    try:
        result = validate_launch_gate(
            corpus_generation_id="new",
            manifest=None,
            approval_path=approval,
        )
        assert not result.allowed
    finally:
        if approval.is_file():
            approval.unlink()


def test_dry_run_output_path_guard():
    validate_dry_run_output_path(DRY_RUN_LOG_DIR / "ok.json")
    with pytest.raises(SystemExit):
        validate_dry_run_output_path(_TRAINING / "runs" / "bad.json")


def test_dry_run_no_subprocess_or_db(monkeypatch):
    monkeypatch.setenv("TRAINING_PREP_ONLY", "1")
    import prepare_diversity_plan as planner
    from prep_guard import DRY_RUN_LOG_DIR

    out = DRY_RUN_LOG_DIR / "_test_side_effects.json"
    connect = mock.Mock()
    try:
        with (
            mock.patch(
                "corpus_semantic_manifest.subprocess.check_output",
                return_value="deadbeef\n",
            ),
            mock.patch("sqlite3.connect", connect),
        ):
            rc = planner.main(
                [
                    "--rows",
                    "500",
                    "--dry-run",
                    "--corpus-id",
                    "side-effect-test",
                    "--out",
                    str(out),
                ]
            )
        assert rc == 0
        connect.assert_not_called()
        assert out.is_file()
    finally:
        if out.is_file():
            out.unlink()


def test_freeze_audit_returns_structured_report(monkeypatch):
    monkeypatch.setenv("TRAINING_PREP_ONLY", "1")
    report = audit_freeze()
    assert report["status"] in ("PASS", "BLOCK", "INVALID")
    assert "manual_stop_command" in report


_TRAINING_ROOT = Path(__file__).resolve().parents[1]

# Executable training paths that must not enforce the deploy-only four-ply trunk.
_FORBIDDEN_FOUR_PLY_TRUNK_PATHS = [
    _TRAINING_ROOT / "streaming_db_loader.py",
    _TRAINING_ROOT / "training_coordinator.py",
    _TRAINING_ROOT / "continuous_pool.py",
    _TRAINING_ROOT / "generation_matchup.py",
    _TRAINING_ROOT / "streaming_epoch_validation.py",
    _TRAINING_ROOT / "streaming_checkpoint_chain.py",
    _TRAINING_ROOT / "canonical_sampling.py",
    _TRAINING_ROOT / "training_sampler.py",
    _TRAINING_ROOT / "label_resolution.py",
    _TRAINING_ROOT / "db_import.py",
]

_FORBIDDEN_FOUR_PLY_MARKERS = (
    "OPENING_SANITY_PREFIX",
    '("e2", "e8", "e3", "e7")',
    "move_num BETWEEN 0 AND 3",
    "COUNT(DISTINCT move_num) = 4",
)


def test_training_code_must_not_use_four_ply_trunk_filter():
    violations: list[str] = []
    for path in _FORBIDDEN_FOUR_PLY_TRUNK_PATHS:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for marker in _FORBIDDEN_FOUR_PLY_MARKERS:
            if marker in text:
                violations.append(f"{path.name}: {marker}")
    assert not violations, "four-ply deploy trunk misused in training paths:\n" + "\n".join(violations)
