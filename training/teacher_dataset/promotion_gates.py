"""Structured teacher-dataset promotion gate definitions and derived promotion_allowed."""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Gates required before promoting teacher_dataset_candidate -> teacher_dataset
TEACHER_PROMOTION_GATES: tuple[str, ...] = (
    "cross_language_position_parity",
    "canonical_hash_parity",
    "policy_hash_algorithm_parity",
    "dataset_semantic_parity",
    "policy_payload_audit",
    "duckdb_catalog_audit",
    "concurrent_reader_test",
    "value_loader_smoke",
    "policy_loader_smoke",
    "required_tests",
)

# Engine deployment only — never blocks teacher dataset promotion.
ENGINE_DEPLOYMENT_GATES: tuple[str, ...] = (
    "engine_move_gen_parity",
)

LEGACY_GATE_ALIASES: dict[str, str] = {
    "semantic_parity_passed": "policy_hash_algorithm_parity",
    "policy_payload_audit_passed": "policy_payload_audit",
    "duckdb_catalog_audit_passed": "duckdb_catalog_audit",
    "concurrent_reader_test_passed": "concurrent_reader_test",
    "value_loader_smoke_passed": "value_loader_smoke",
    "policy_loader_smoke_passed": "policy_loader_smoke",
    "all_required_tests_passed": "required_tests",
    "engine_move_gen_parity_verified": "engine_move_gen_parity",
}


def git_head(root: Path | None = None) -> str | None:
    try:
        root = root or Path(__file__).resolve().parents[2]
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_manifest_hash(manifest: dict[str, Any]) -> str:
    payload = {k: v for k, v in manifest.items() if k != "manifest_hash"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def gate_evidence(
    *,
    passed: bool,
    report_path: Path | str | None = None,
    tool_version: str = "teacher_dataset.promotion_gates/1",
    timestamp: str | None = None,
    commit: str | None = None,
    counts: dict[str, Any] | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "passed": passed,
        "tool_version": tool_version,
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
    }
    if commit:
        out["commit"] = commit
    if report_path is not None:
        rp = Path(report_path)
        out["report_path"] = str(rp).replace("\\", "/")
        if rp.is_file():
            out["report_sha256"] = sha256_file(rp)
    if counts:
        out["counts"] = counts
    if notes:
        out["notes"] = notes
    return out


def normalize_promotion_gates(raw: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Convert legacy bool gates to structured gate dicts."""
    if not raw:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for key, val in raw.items():
        canonical = LEGACY_GATE_ALIASES.get(key, key)
        if isinstance(val, dict):
            out[canonical] = val
        elif isinstance(val, bool):
            out[canonical] = {"passed": val}
        else:
            out[canonical] = {"passed": bool(val)}
    return out


def gate_passed(gates: dict[str, dict[str, Any]], name: str) -> bool:
    g = gates.get(name)
    if g is None:
        return False
    if isinstance(g, bool):
        return g
    return bool(g.get("passed"))


def derive_promotion_allowed(manifest: dict[str, Any]) -> bool:
    counts = manifest.get("counts") or {}
    if int(counts.get("policy_quarantined", 0)) != 0:
        return False
    resolution = manifest.get("policy_resolution") or {}
    unresolved = int(
        resolution.get("unresolved", 0) or resolution.get("v8_still_unresolved", 0)
    )
    if unresolved != 0:
        return False
    gates = normalize_promotion_gates(manifest.get("promotion_gates"))
    return all(gate_passed(gates, g) for g in TEACHER_PROMOTION_GATES)


def apply_promotion_allowed(manifest: dict[str, Any]) -> dict[str, Any]:
    manifest = dict(manifest)
    manifest["promotion_allowed"] = derive_promotion_allowed(manifest)
    manifest["manifest_hash"] = compute_manifest_hash(manifest)
    return manifest
