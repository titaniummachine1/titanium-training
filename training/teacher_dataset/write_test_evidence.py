"""Atomically write teacher_dataset_test_evidence.json after a successful pytest run."""
from __future__ import annotations

import json
import os
import platform
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from position_store_config import REPORT_DIR, ROOT

from .promotion_gates import compute_manifest_hash, git_head, sha256_file


def write_test_evidence(
    *,
    commands: list[dict],
    suites: list[dict],
    total_passed: int,
    total_failed: int,
    total_skipped: int,
    duration_seconds: float,
    exit_codes: list[int],
    start_timestamp: str,
    end_timestamp: str,
    audit_timestamp: str = "20260620T101843Z",
    gate_bundle_rel: str = (
        "training/data/position_store_reports/"
        "gate_evidence_bundle_teacher_dataset_candidate_v9_20260620T101843Z.json"
    ),
    candidate: str = "teacher_dataset_candidate_v10",
) -> Path:
    if any(code != 0 for code in exit_codes) or total_failed:
        raise RuntimeError("refusing to write pass evidence for failed test run")

    bundle_path = ROOT / gate_bundle_rel
    bundle_sha_on_disk = sha256_file(bundle_path) if bundle_path.is_file() else None
    bundle_sha_audit = bundle_sha_on_disk
    if bundle_path.is_file():
        embedded = json.loads(bundle_path.read_text(encoding="utf-8")).get("bundle_sha256")
        if embedded:
            bundle_sha_audit = embedded
    manifest_path = ROOT / "training" / "data" / candidate / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    candidate_manifest_sha = manifest.get("manifest_hash")

    payload = {
        "status": "pass",
        "timestamp": end_timestamp,
        "start_timestamp": start_timestamp,
        "end_timestamp": end_timestamp,
        "commands": commands,
        "suites": suites,
        "total_passed": total_passed,
        "total_failed": total_failed,
        "total_skipped": total_skipped,
        "duration_seconds": duration_seconds,
        "exit_codes": exit_codes,
        "python_version": sys.version,
        "pytest_version": pytest.__version__,
        "platform": platform.platform(),
        "training_commit_before_final_commits": git_head(ROOT),
        "audit_timestamp": audit_timestamp,
        "gate_evidence_bundle": gate_bundle_rel.replace("\\", "/"),
        "gate_evidence_bundle_sha256": bundle_sha_audit,
        "gate_evidence_bundle_sha256_on_disk": bundle_sha_on_disk,
        "candidate": candidate,
        "candidate_manifest_sha256": candidate_manifest_sha,
        "promotion_allowed": False,
    }

    out_dir = REPORT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / "teacher_dataset_test_evidence.json"
    fd, tmp_name = tempfile.mkstemp(dir=out_dir, prefix="teacher_dataset_test_evidence.", suffix=".tmp")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp_path.replace(final_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
    return final_path
