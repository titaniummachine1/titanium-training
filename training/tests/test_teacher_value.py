"""Tests for teacher-value dataset loading and bundle size rules."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "scripts" / "oracle" / "build_upload_bundle.py"


def test_bundle_excludes_repository_inventory_json(tmp_path: Path) -> None:
    out = tmp_path / "bundle"
    rc = subprocess.run(
        [sys.executable, str(BUILD), "--output", str(out), "--code-only"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert rc.returncode == 0, rc.stderr or rc.stdout
    manifest = json.loads((out / "transfer-manifest.json").read_text(encoding="utf-8"))
    paths = [e["path"].replace("\\", "/") for e in manifest["files"]]
    assert not any(p == "docs/maintenance/repository_inventory.json" for p in paths)
    assert not any(".pytest_cache" in p for p in paths)
    total = sum(e["size_bytes"] for e in manifest["files"])
    assert total < 3_000_000, f"unexpected code bundle size {total}"


def test_teacher_manifest_binding() -> None:
    from titanium_training.data.teacher_value import load_manifest, verify_manifest_identity
    from titanium_training.paths import ACTIVE_TEACHER_DATASET

    if not ACTIVE_TEACHER_DATASET.is_file():
        pytest.skip("active dataset not present")
    manifest = load_manifest(ACTIVE_TEACHER_DATASET)
    verify_manifest_identity(manifest)


def test_teacher_value_target_range() -> None:
    from titanium_training.data.teacher_value import teacher_value_target

    assert teacher_value_target(-100) == pytest.approx(0.0)
    assert teacher_value_target(0) == pytest.approx(0.5)
    assert teacher_value_target(100) == pytest.approx(1.0)


@pytest.mark.integration
def test_teacher_featurization_nonzero() -> None:
    from titanium_training.data.teacher_value import load_teacher_value_training_records
    from titanium_training.paths import ACTIVE_TEACHER_DATASET, ENGINE_BIN

    if not (ACTIVE_TEACHER_DATASET / "manifest.json").is_file():
        pytest.skip("active dataset not present")
    if not ENGINE_BIN.is_file():
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
    assert meta["dataset_manifest_sha256"] == "31a422f25a8c701ebfa72410f59fab9dff52c2717e30985a3f8e6929be007d02"
    for rec in records:
        assert "legal_wall_count" in rec
        assert -1.0 <= rec["outcome"] <= 1.0
