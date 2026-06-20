"""Tests for immutable teacher gate evidence canonicalization and envelopes."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from teacher_dataset.evidence_canonical import (
    GATE_DEFINITIONS,
    audit_payload_sha256,
    build_canonical_audit_payload,
    canonical_json_dumps,
    compute_gate_aggregates,
    enrich_gate_record,
)
from teacher_dataset.evidence_envelope import (
    build_final_evidence_envelope,
    finalize_evidence_envelope,
    validate_final_envelope,
    write_sha256_sidecar,
)
from teacher_dataset.promotion_gates import sha256_file


@pytest.fixture
def legacy_bundle() -> dict:
    from titanium_training.store.config import ROOT

    path = (
        ROOT
        / "training/data/position_store_reports/"
        "gate_evidence_bundle_teacher_dataset_candidate_v9_20260620T101843Z.json"
    )
    if not path.is_file():
        pytest.skip("legacy audit bundle not present")
    return json.loads(path.read_text(encoding="utf-8"))


def test_canonical_payload_hash_is_order_independent(legacy_bundle: dict) -> None:
    payload = build_canonical_audit_payload(legacy_bundle)
    copy = json.loads(canonical_json_dumps(payload))
    assert audit_payload_sha256(copy) == payload["audit_payload_sha256"]


def test_envelope_whitespace_does_not_change_audit_payload_sha256(legacy_bundle: dict) -> None:
    payload = build_canonical_audit_payload(legacy_bundle)
    h1 = payload["audit_payload_sha256"]
    wrapped = {"audit_payload": payload, "pretty": True}
    h2 = audit_payload_sha256(payload)
    assert h1 == h2
    assert json.dumps(wrapped, indent=4) != canonical_json_dumps(wrapped)


def test_changing_gate_result_changes_audit_payload_sha256(legacy_bundle: dict) -> None:
    payload = build_canonical_audit_payload(legacy_bundle)
    mutated = json.loads(canonical_json_dumps(payload))
    mutated["gates"]["value_loader_smoke"]["status"] = "fail"
    assert audit_payload_sha256(mutated) != payload["audit_payload_sha256"]


def test_changing_artifact_hash_changes_audit_payload_sha256(legacy_bundle: dict) -> None:
    payload = build_canonical_audit_payload(legacy_bundle)
    mutated = json.loads(canonical_json_dumps(payload))
    first_key = next(iter(mutated["artifact_hashes"]))
    mutated["artifact_hashes"][first_key] = "0" * 64
    assert audit_payload_sha256(mutated) != payload["audit_payload_sha256"]


def test_post_audit_metadata_not_in_canonical_payload(legacy_bundle: dict) -> None:
    payload = build_canonical_audit_payload(legacy_bundle)
    assert "bundle_sha256" not in payload
    assert "bundle_path" not in payload
    envelope = build_final_evidence_envelope(
        legacy_bundle,
        audit_payload=payload,
        post_audit={"note": "added later"},
    )
    assert envelope["post_audit"]["note"] == "added later"
    assert audit_payload_sha256(payload) == payload["audit_payload_sha256"]


def test_nonblocking_failed_gate_keeps_required_aggregate_true(legacy_bundle: dict) -> None:
    gates = {name: enrich_gate_record(name, legacy_bundle["promotion_gates"][name]) for name in legacy_bundle["promotion_gates"]}
    agg = compute_gate_aggregates(gates)
    assert agg["all_required_teacher_gates_passed"] is True
    assert agg["all_executed_checks_passed"] is False
    assert "engine_move_gen_parity" in agg["nonblocking_failures"]


def test_required_failed_gate_makes_required_aggregate_false(legacy_bundle: dict) -> None:
    gates = {name: enrich_gate_record(name, legacy_bundle["promotion_gates"][name]) for name in legacy_bundle["promotion_gates"]}
    gates["value_loader_smoke"]["status"] = "fail"
    agg = compute_gate_aggregates(gates)
    assert agg["all_required_teacher_gates_passed"] is False


def test_missing_gate_definition_fails_closed() -> None:
    with pytest.raises(ValueError, match="unknown gate"):
        enrich_gate_record("not_a_gate", {"passed": True})


def test_promotion_allowed_false_in_canonical_payload(legacy_bundle: dict) -> None:
    payload = build_canonical_audit_payload(legacy_bundle)
    assert payload["promotion_allowed"] is False
    assert payload["aggregates"]["promotion_allowed"] is False


def test_finalize_envelope_matches_sidecar(legacy_bundle: dict, tmp_path: Path) -> None:
    legacy_path = tmp_path / "legacy.json"
    legacy_path.write_text(json.dumps(legacy_bundle, indent=2), encoding="utf-8")
    final_path = tmp_path / "bundle.final.json"
    result = finalize_evidence_envelope(legacy_path, final_path=final_path, reports_dir=tmp_path)
    sidecar = tmp_path / "bundle.final.json.sha256"
    assert sidecar.is_file()
    assert validate_final_envelope(final_path)["final_bundle_file_sha256"] == result["final_bundle_file_sha256"]


def test_validator_detects_post_finalization_mutation(legacy_bundle: dict, tmp_path: Path) -> None:
    legacy_path = tmp_path / "legacy.json"
    legacy_path.write_text(json.dumps(legacy_bundle), encoding="utf-8")
    final_path = tmp_path / "bundle.final.json"
    finalize_evidence_envelope(legacy_path, final_path=final_path, reports_dir=tmp_path)
    data = json.loads(final_path.read_text(encoding="utf-8"))
    data["post_audit"] = {"tampered": True}
    final_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="sidecar hash mismatch"):
        validate_final_envelope(final_path)


def test_atomic_sidecar_format(tmp_path: Path) -> None:
    target = tmp_path / "sample.json"
    target.write_text("{}", encoding="utf-8")
    sidecar = write_sha256_sidecar(target)
    line = sidecar.read_text(encoding="utf-8").strip()
    digest, name = line.split(maxsplit=1)
    assert name == target.name
    assert digest == sha256_file(target)


def test_all_executed_gates_have_definitions(legacy_bundle: dict) -> None:
    for name in legacy_bundle["promotion_gates"]:
        assert name in GATE_DEFINITIONS, f"missing definition for {name}"
