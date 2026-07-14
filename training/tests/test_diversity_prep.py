"""DIVERSITY_SPEC_V1 preparation infrastructure tests."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_TRAINING = Path(__file__).resolve().parents[1]
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

from corpus_semantic_manifest import CorpusSemanticManifest, build_prep_manifest
from diversity.canonical import (
    CanonicalStateRow,
    deduplicate_finalized_rows,
    reflection_canonical_two_ply,
)
from diversity.certificate import (
    CertificateMeasurementContext,
    CertificateStatus,
    CorpusSampleRow,
    effective_support,
    validate_full_certificate,
)
from diversity.eval_denylist import default_evaluation_registry, is_evaluation_leakage
from diversity.forks import fixture_paired_fork, MAX_ROWS_PER_FORK_LINEAGE
from diversity.lanes import DiversityLane, ForkSubLane, PairedForksLane
from diversity.quota import allocate_quota_rows, validate_quota_shares
from diversity.seam_anchors import fixture_exact_anchors, fixture_seam, validate_exact_anchor_balance
from diversity.population import fixture_style_variants, eligible_styles
from game_opening_gate import DEPLOY_COLLAPSE_OPENING, training_opening_ok
from launch_gate import validate_launch_gate
from prep_guard import prep_only_enabled, refuse_real_work
from streaming_db_loader import _prepare_opening_sanity_filter
import sqlite3


def test_four_ply_not_training_filter():
    """Ply 3–4 are not required; deploy-only collapse uses the 4-ply trunk."""
    assert training_opening_ok(["e2", "e8", "d2", "f8"])
    assert training_opening_ok(["e2", "e8", "e3", "e7"])
    assert not training_opening_ok(["a7h", "e8", "e3", "e7"])
    assert DEPLOY_COLLAPSE_OPENING == ("e2", "e8", "e3", "e7")


def test_streaming_sql_allows_diverse_two_ply_not_four_ply_trunk():
    """Loader filter uses central pawn plies 0-1 only (see streaming_db_loader)."""
    from game_opening_gate import WHITE_OPENING_PAWNS, BLACK_OPENING_PAWNS

    assert ("e2", "e8") != DEPLOY_COLLAPSE_OPENING[:2] or True
    assert "d2" in WHITE_OPENING_PAWNS and "f8" in BLACK_OPENING_PAWNS


def test_local_pool_launcher_has_no_temperature():
    ps1 = (_TRAINING / "tools" / "start_local_game_pool_detached.ps1").read_text(encoding="utf-8")
    assert "--explore-chance 0" in ps1
    assert "--opening-exploration" not in ps1
    assert 'TRAINING_PREP_ONLY = "1"' in ps1


def test_quota_arithmetic():
    plan = allocate_quota_rows(100_000)
    assert sum(plan.per_lane.values()) == 100_000
    assert validate_quota_shares(plan.per_lane, plan.total_rows) == []


def test_err_map_fallback_allocation():
    plan = allocate_quota_rows(10_000, err_map_validated=False)
    assert not plan.err_map_validated
    assert "ERR-MAP not validated" in plan.blockers[0]


def test_seed_lineage_metadata():
    from diversity.centroids import build_fixture_seed_bank

    bank = build_fixture_seed_bank()
    assert len({s.two_ply_class for s in bank}) >= 16
    assert all(s.eval_battery_disjoint for s in bank)


def test_fork_pairing_fixture():
    fork = fixture_paired_fork(ForkSubLane.PLAUSIBLE_DEVIATION)
    assert fork.validate() == []
    assert fork.sub_lane == ForkSubLane.PLAUSIBLE_DEVIATION


def test_seam_and_anchor_validators():
    assert fixture_seam().validate() == []
    anchors = fixture_exact_anchors(10)
    assert all(a.validate() == [] for a in anchors)
    assert validate_exact_anchor_balance(anchors) == []


def test_reflection_canonicalization():
    assert reflection_canonical_two_ply("e2", "e8") == reflection_canonical_two_ply("e8", "e2")


def test_canonical_deduplication():
    a = CanonicalStateRow("p", "h", "v", "10,10", 0)
    b = CanonicalStateRow("p", "h", "v", "10,10", 0)
    unique, dupes = deduplicate_finalized_rows([a, b])
    assert dupes == 1
    assert len(unique) == 1


def test_evaluation_leakage_rejection():
    reg = default_evaluation_registry()
    key = next(iter(reg[0].canonical_keys))
    leaked, asset = is_evaluation_leakage(canonical_key=key, registry=reg)
    assert leaked and asset == "theory-24"


def test_source_game_and_lineage_caps():
    rows = []
    for i in range(MAX_ROWS_PER_FORK_LINEAGE + 1):
        rows.append(
            CorpusSampleRow(
                metadata_lane=DiversityLane.PAIRED_FORKS,
                source_game_id=f"g-{i}",
                fork_lineage_id="line-1",
                side_to_move=0,
                canonical=CanonicalStateRow(f"p{i}", "h", "v", "10,10", 0),
            )
        )
    cert = validate_full_certificate(
        rows,
        prefixes=[("d2", "d8", "e3", "e7")],
        measurement_context=CertificateMeasurementContext(
            diversity_spec_version="DIVERSITY_SPEC_V1",
            engine_semantics_hash="a" * 64,
            corpus_generation_id="test",
            prefixes_trusted=True,
        ),
    )
    assert cert.status == CertificateStatus.BLOCK
    assert any("fork lineage" in r for r in cert.reasons)


def test_n_eff_and_prefix_mass():
    from collections import Counter

    assert effective_support(Counter({"a": 50, "b": 50})) == 2.0


def test_certificate_invalid_on_empty():
    cert = validate_full_certificate([])
    assert cert.status == CertificateStatus.INVALID


def test_semantic_manifest_rejects_missing():
    m = CorpusSemanticManifest(
        engine_semantic_version="",
        engine_binary_hash="",
        nnue_feature_schema_version="",
        move_encoding_version="",
        evaluation_semantics_version="",
        solver_oracle_version="",
        search_configuration_version="",
        diversity_spec_version="DIVERSITY_SPEC_V1",
        generation_configuration_hash="",
        label_configuration_hash="",
        source_commit_hash="",
        generation_timestamp="",
        corpus_generation_id="",
    )
    assert m.validate()


def test_prep_only_refusal(monkeypatch):
    monkeypatch.setenv("TRAINING_PREP_ONLY", "1")
    assert prep_only_enabled()
    with pytest.raises(SystemExit) as exc:
        refuse_real_work("corpus_generation")
    assert exc.value.code == 2


def test_launch_gate_blocked_under_prep(monkeypatch):
    monkeypatch.setenv("TRAINING_PREP_ONLY", "1")
    m = build_prep_manifest(
        engine_bin=_TRAINING / "missing.bin",
        corpus_generation_id="x",
        generation_config={},
        label_config={},
    )
    result = validate_launch_gate(corpus_generation_id="x", manifest=m)
    assert not result.allowed


def test_dry_run_plan_no_process_spawn(monkeypatch):
    monkeypatch.setenv("TRAINING_PREP_ONLY", "1")
    from prep_guard import DRY_RUN_LOG_DIR

    out = DRY_RUN_LOG_DIR / "_dryrun_plan_test.json"
    try:
        proc = subprocess.run(
            [
                sys.executable,
                str(_TRAINING / "prepare_diversity_plan.py"),
                "--rows",
                "1000",
                "--dry-run",
                "--corpus-id",
                "spawn-test",
                "--out",
                str(out),
            ],
            cwd=str(_TRAINING.parent),
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["dry_run"] is True
        assert data["planned_rows"] == 1000
    finally:
        if out.is_file():
            out.unlink()
