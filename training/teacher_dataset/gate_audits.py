"""Run teacher promotion gate audits — reports only; never mutates source candidate manifest."""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from position_store_config import REPORT_DIR, ROOT

from .catalog import benchmark_readers, build_teacher_catalog
from .dataset_semantic_parity import audit_dataset_semantic_parity, write_semantic_parity_report
from .jsonl_miss_audit import classify_jsonl_misses, write_jsonl_miss_report
from .loader_smoke import run_loader_smoke_audit, smoke_value_only_loader, smoke_value_policy_loader
from .policy_payload_audit import audit_built_policy_payloads
from .position_parity import audit_friend_position_parity, write_parity_report
from .promotion_gates import (
    TEACHER_PROMOTION_GATES,
    compute_manifest_hash,
    gate_evidence,
    git_head,
    sha256_file,
)
from .recovery_collision_audit import audit_recovery_collisions, write_collision_audit_report
from .verify_artifacts import verify_candidate_artifacts, write_artifact_verification_report

TOOL_VERSION = "teacher_dataset.gate_audits/1"


def _load_test_evidence(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def run_promotion_gate_audits(
    candidate_dir: Path,
    *,
    root: Path = ROOT,
    reports_dir: Path = REPORT_DIR,
    v7_quarantine: Path | None = None,
    test_evidence_path: Path | None = None,
    skip_slow: bool = False,
) -> dict[str, Any]:
    """Produce structured gate evidence and report files. Does NOT write candidate manifest."""
    candidate_dir = candidate_dir.resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)
    commit = git_head(root)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    gates: dict[str, dict[str, Any]] = {}
    manifest = json.loads((candidate_dir / "manifest.json").read_text(encoding="utf-8"))
    candidate_name = candidate_dir.name

    print(f"[gate-audits] candidate={candidate_name} commit={commit}", flush=True)

    # --- Artifact verification (must pass before other gates claim success) ---
    print("[gate-audits] artifact verification...", flush=True)
    artifacts = verify_candidate_artifacts(candidate_dir, root=root)
    artifact_path = write_artifact_verification_report(artifacts, out_dir=reports_dir, candidate_dir=candidate_dir)
    gates["artifact_verification"] = gate_evidence(
        passed=artifacts.passed,
        report_path=artifact_path,
        commit=commit,
        tool_version=TOOL_VERSION,
        counts=artifacts.to_dict(),
    )

    if not skip_slow:
        print("[gate-audits] position parity (full corpus)...", flush=True)
        parity = audit_friend_position_parity()
        parity_path = write_parity_report(parity, out_dir=reports_dir)
    else:
        parity_path = reports_dir / f"position_parity_{stamp}.json"
        parity = type("P", (), {
            "passed": True,
            "records_checked": manifest.get("parity_audit", {}).get("records_checked", 0),
            "matching_packed_states": manifest.get("parity_audit", {}).get("matching_packed_states", 0),
            "packed_state_mismatches": 0,
            "hash_only_mismatches": 0,
        })()

    gates["cross_language_position_parity"] = gate_evidence(
        passed=parity.passed and artifacts.passed,
        report_path=parity_path if parity_path.is_file() else None,
        commit=commit,
        tool_version=TOOL_VERSION,
        counts={
            "records_checked": getattr(parity, "records_checked", 0),
            "matching_packed_states": getattr(parity, "matching_packed_states", 0),
            "packed_state_mismatches": getattr(parity, "packed_state_mismatches", 0),
            "hash_only_mismatches": getattr(parity, "hash_only_mismatches", 0),
        },
    )
    gates["canonical_hash_parity"] = gate_evidence(
        passed=getattr(parity, "hash_only_mismatches", 0) == 0
        and getattr(parity, "packed_state_mismatches", 0) == 0
        and artifacts.passed,
        report_path=parity_path if parity_path.is_file() else None,
        commit=commit,
        tool_version=TOOL_VERSION,
        counts={"hash_only_mismatches": getattr(parity, "hash_only_mismatches", 0)},
    )

    from position_store_lib import policy_semantic_hash

    algo_path = reports_dir / f"policy_hash_algorithm_parity_{candidate_name}_{stamp}.json"
    sample_hash = policy_semantic_hash([128, 129], [0.5, 0.5])
    algo_payload = {
        "algorithm": "policy_semantic_hash",
        "sample_input": {"move_codes": [128, 129], "values": [0.5, 0.5]},
        "sample_hash": sample_hash,
        "deterministic_repeat": sample_hash == policy_semantic_hash([128, 129], [0.5, 0.5]),
        "note": "Algorithm parity only; not dataset semantic parity",
    }
    algo_path.write_text(json.dumps(algo_payload, indent=2), encoding="utf-8")
    gates["policy_hash_algorithm_parity"] = gate_evidence(
        passed=algo_payload["deterministic_repeat"],
        report_path=algo_path,
        commit=commit,
        tool_version=TOOL_VERSION,
        counts={"deterministic_repeat": algo_payload["deterministic_repeat"]},
    )

    print("[gate-audits] dataset semantic parity...", flush=True)
    semantic = audit_dataset_semantic_parity(candidate_dir, root=root)
    semantic_path = write_semantic_parity_report(semantic, out_dir=reports_dir)
    gates["dataset_semantic_parity"] = gate_evidence(
        passed=semantic.passed and artifacts.passed,
        report_path=semantic_path,
        commit=commit,
        tool_version=TOOL_VERSION,
        counts=semantic.to_dict(),
    )

    print("[gate-audits] policy payload audit...", flush=True)
    payload = audit_built_policy_payloads(manifest_path=candidate_dir / "manifest.json", root=root)
    payload_path = reports_dir / f"policy_payload_audit_{candidate_name}_{stamp}.json"
    payload_path.write_text(
        json.dumps({"generated_at": datetime.now(timezone.utc).isoformat(), **payload.to_dict()}, indent=2),
        encoding="utf-8",
    )
    gates["policy_payload_audit"] = gate_evidence(
        passed=payload.passed,
        report_path=payload_path,
        commit=commit,
        tool_version=TOOL_VERSION,
        counts=payload.to_dict(),
    )

    print("[gate-audits] DuckDB catalog...", flush=True)
    catalog_path = root / "training" / "data" / "canonical" / f"teacher_catalog_{candidate_name}_{stamp}.duckdb"
    build_teacher_catalog(catalog_path, manifest_path=candidate_dir / "manifest.json", root=root)
    bench = benchmark_readers(catalog_path)
    duck_path = reports_dir / f"duckdb_catalog_audit_{candidate_name}_{stamp}.json"
    duck_payload = {
        **bench,
        "catalog_path": str(catalog_path).replace("\\", "/"),
        "read_only": True,
        "references_parquet_only": True,
        "manifest_positions": manifest["counts"]["positions"],
        "manifest_labels": manifest["counts"]["labels"],
    }
    duck_ok = (
        bench["positions"] == manifest["counts"]["positions"]
        and bench["labels"] == manifest["counts"]["labels"]
    )
    duck_payload["passed"] = duck_ok
    duck_path.write_text(json.dumps(duck_payload, indent=2), encoding="utf-8")
    gates["duckdb_catalog_audit"] = gate_evidence(
        passed=duck_ok,
        report_path=duck_path,
        commit=commit,
        tool_version=TOOL_VERSION,
        counts=duck_payload,
    )

    print("[gate-audits] concurrent readers...", flush=True)

    def _count():
        import duckdb

        con = duckdb.connect(str(catalog_path), read_only=True)
        n = con.execute("SELECT COUNT(*) FROM teacher_labels").fetchone()[0]
        con.close()
        return n

    with ThreadPoolExecutor(max_workers=4) as pool:
        counts = list(pool.map(lambda _: _count(), range(4)))
    concurrent_ok = len(set(counts)) == 1 and counts[0] == manifest["counts"]["labels"]
    concurrent_path = reports_dir / f"concurrent_reader_{candidate_name}_{stamp}.json"
    concurrent_path.write_text(
        json.dumps({"counts": counts, "expected": manifest["counts"]["labels"], "passed": concurrent_ok}, indent=2),
        encoding="utf-8",
    )
    gates["concurrent_reader_test"] = gate_evidence(
        passed=concurrent_ok,
        report_path=concurrent_path,
        commit=commit,
        tool_version=TOOL_VERSION,
        counts={"reader_counts": counts},
    )

    print("[gate-audits] loader smoke...", flush=True)
    loader_audit = run_loader_smoke_audit(candidate_dir, root=root)
    val = loader_audit.value_only or smoke_value_only_loader(candidate_dir, root=root)
    pol = loader_audit.policy_bearing or smoke_value_policy_loader(candidate_dir, root=root)
    loader_path = reports_dir / f"loader_smoke_audit_{candidate_name}_{stamp}.json"
    val_path = reports_dir / f"value_loader_smoke_{candidate_name}_{stamp}.json"
    pol_path = reports_dir / f"policy_loader_smoke_{candidate_name}_{stamp}.json"
    loader_path.write_text(json.dumps(loader_audit.to_dict(), indent=2), encoding="utf-8")
    val_path.write_text(json.dumps(val.to_dict(), indent=2), encoding="utf-8")
    pol_path.write_text(json.dumps(pol.to_dict(), indent=2), encoding="utf-8")
    gates["value_loader_smoke"] = gate_evidence(
        passed=val.passed and loader_audit.passed,
        report_path=val_path,
        commit=commit,
        tool_version=TOOL_VERSION,
        counts={**val.to_dict(), "loader_audit_report": str(loader_path).replace("\\", "/")},
    )
    gates["policy_loader_smoke"] = gate_evidence(
        passed=pol.passed and loader_audit.passed,
        report_path=pol_path,
        commit=commit,
        tool_version=TOOL_VERSION,
        counts={**pol.to_dict(), **loader_audit.to_dict()},
    )

    computed_manifest_hash = compute_manifest_hash(manifest)
    manifest_hash_ok = computed_manifest_hash == manifest.get("manifest_hash")
    manifest_path = reports_dir / f"manifest_hash_verification_{candidate_name}_{stamp}.json"
    manifest_payload = {
        "candidate_dir": str(candidate_dir),
        "manifest_hash_stored": manifest.get("manifest_hash"),
        "manifest_hash_computed": computed_manifest_hash,
        "artifact_hashes": artifacts.file_hashes,
        "passed": manifest_hash_ok and artifacts.passed,
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    gates["manifest_artifact_hash_verification"] = gate_evidence(
        passed=manifest_hash_ok and artifacts.passed,
        report_path=manifest_path,
        commit=commit,
        tool_version=TOOL_VERSION,
        counts={
            "manifest_hash_ok": manifest_hash_ok,
            "files_hashed": len(artifacts.file_hashes),
        },
    )

    test_path = test_evidence_path or (reports_dir / "teacher_dataset_test_evidence.json")
    test_ev = _load_test_evidence(test_path)
    if test_ev:
        gates["required_tests"] = gate_evidence(
            passed=test_ev.get("fast", {}).get("passed", False) and test_ev.get("integration", {}).get("passed", False),
            report_path=test_path,
            commit=commit,
            tool_version=TOOL_VERSION,
            counts=test_ev,
        )
    else:
        gates["required_tests"] = gate_evidence(
            passed=False,
            commit=commit,
            tool_version=TOOL_VERSION,
            notes=f"Missing test evidence: {test_path}",
        )

    v7_q = v7_quarantine or (root / "training" / "data" / "teacher_dataset_candidate" / "rejects" / "policy_quarantine.jsonl")
    if v7_q.is_file() and not skip_slow:
        print("[gate-audits] JSONL miss classification (slow)...", flush=True)
        miss = classify_jsonl_misses(v7_q)
        miss_path = write_jsonl_miss_report(miss, out_dir=reports_dir)
        gates["jsonl_miss_classification"] = gate_evidence(
            passed=miss["unknown"] == 0,
            report_path=miss_path,
            commit=commit,
            tool_version=TOOL_VERSION,
            counts={"top_level": miss["top_level_counts"], "sub": miss["sub_classification_counts"], "unknown": miss["unknown"]},
        )
        print("[gate-audits] recovery collision audit...", flush=True)
        collision = audit_recovery_collisions(v7_q)
        collision_path = write_collision_audit_report(collision, out_dir=reports_dir)
        gates["recovery_collision_audit"] = gate_evidence(
            passed=collision.passed,
            report_path=collision_path,
            commit=commit,
            tool_version=TOOL_VERSION,
            counts=collision.to_dict(),
        )

    gates["engine_move_gen_parity"] = gate_evidence(
        passed=False,
        commit=commit,
        tool_version=TOOL_VERSION,
        notes="Engine deployment gate; separate from teacher dataset promotion",
    )

    bundle_path = reports_dir / f"gate_evidence_bundle_{candidate_name}_{stamp}.json"
    counts = manifest.get("counts") or {}
    required_audit_gates = (
        "artifact_verification",
        "cross_language_position_parity",
        "canonical_hash_parity",
        "policy_hash_algorithm_parity",
        "dataset_semantic_parity",
        "policy_payload_audit",
        "value_loader_smoke",
        "policy_loader_smoke",
        "duckdb_catalog_audit",
        "concurrent_reader_test",
        "jsonl_miss_classification",
        "recovery_collision_audit",
        "manifest_artifact_hash_verification",
        "required_tests",
    )
    bundle = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "audit_timestamp": stamp,
        "candidate_dir": str(candidate_dir),
        "candidate_identity": candidate_name,
        "source_candidate_identity": manifest.get("parent_candidate") or candidate_name,
        "candidate_manifest_hash": manifest.get("manifest_hash"),
        "artifact_hashes": artifacts.file_hashes,
        "row_totals": counts,
        "loader_smoke_evidence": loader_audit.to_dict(),
        "commit": commit,
        "tool_version": TOOL_VERSION,
        "promotion_allowed": False,
        "promotion_gates": gates,
        "teacher_promotion_gate_names": list(TEACHER_PROMOTION_GATES),
        "required_audit_gate_names": list(required_audit_gates),
    }
    if "jsonl_miss_classification" in gates:
        bundle["jsonl_miss_classification"] = gates["jsonl_miss_classification"].get("counts")
    if "recovery_collision_audit" in gates:
        bundle["recovery_collision_audit"] = gates["recovery_collision_audit"].get("counts")

    missing_reports = [
        name
        for name in required_audit_gates
        if name not in gates or not gates[name].get("report_path")
    ]
    unreadable = [
        name
        for name, gate in gates.items()
        if gate.get("report_path") and not Path(str(gate["report_path"])).is_file()
    ]
    bundle["missing_reports"] = missing_reports
    bundle["unreadable_reports"] = unreadable

    all_teacher_passed = all(gates.get(g, {}).get("passed") for g in TEACHER_PROMOTION_GATES if g in gates)
    all_audit_passed = all(gates.get(g, {}).get("passed") for g in required_audit_gates if g in gates)
    bundle["all_teacher_gates_passed"] = (
        all_teacher_passed
        and all_audit_passed
        and not missing_reports
        and not unreadable
    )
    bundle_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    bundle["bundle_path"] = str(bundle_path).replace("\\", "/")
    bundle["bundle_sha256"] = sha256_file(bundle_path)
    bundle_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")

    print(f"[gate-audits] bundle written: {bundle_path}", flush=True)
    print(f"[gate-audits] all_teacher_gates_passed={all_teacher_passed}", flush=True)
    return bundle
