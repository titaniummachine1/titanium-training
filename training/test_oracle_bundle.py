"""Tests for Oracle bundle builder and repository doctor."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "scripts" / "oracle" / "build_upload_bundle.py"
VERIFY = ROOT / "scripts" / "oracle" / "verify_upload_bundle.py"
DOCTOR = ROOT / "scripts" / "maintenance" / "repository_doctor.py"


@pytest.fixture
def code_bundle(tmp_path: Path) -> Path:
    out = tmp_path / "oracle_upload"
    rc = subprocess.run(
        [sys.executable, str(BUILD), "--output", str(out), "--code-only"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert rc.returncode == 0, rc.stderr or rc.stdout
    return out


def test_build_code_only_bundle(code_bundle: Path) -> None:
    manifest = code_bundle / "transfer-manifest.json"
    assert manifest.is_file()
    doc = json.loads(manifest.read_text(encoding="utf-8"))
    assert doc.get("code_only") is True
    assert doc.get("include_dataset") is False
    assert (code_bundle / "README_FIRST.md").is_file()
    assert (code_bundle / "training" / "nnue_cli.py").is_file()


def test_verify_code_only_bundle(code_bundle: Path) -> None:
    rc = subprocess.run(
        [sys.executable, str(VERIFY), str(code_bundle)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert rc.returncode == 0, rc.stderr or rc.stdout


def test_repository_doctor_smoke() -> None:
    rc = subprocess.run(
        [sys.executable, str(DOCTOR), "--json"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert rc.returncode == 0, rc.stdout + rc.stderr


def test_active_manifest_hash_unchanged() -> None:
    manifest_path = ROOT / "training" / "data" / "teacher_dataset" / "manifest.json"
    if not manifest_path.is_file():
        pytest.skip("active dataset not present locally")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest.get("manifest_hash") == "31a422f25a8c701ebfa72410f59fab9dff52c2717e30985a3f8e6929be007d02"
