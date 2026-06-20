"""Tests for Oracle bundle builder and repository doctor."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
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
    assert (code_bundle / "training" / "titanium_training" / "cli.py").is_file()
    assert not (code_bundle / "training" / "train.py").exists()
    assert not (code_bundle / "training" / "experiments").exists()


def test_verify_code_only_bundle(code_bundle: Path) -> None:
    rc = subprocess.run(
        [sys.executable, str(VERIFY), str(code_bundle)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert rc.returncode == 0, rc.stderr or rc.stdout


def test_bundle_excludes_output_directory_recursion() -> None:
    """Regression: bundle output inside the repo must not package itself recursively."""
    out = ROOT / "dist" / "oracle_recursion_guard"
    rc = subprocess.run(
        [sys.executable, str(BUILD), "--output", str(out), "--code-only"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=180,
    )
    try:
        assert rc.returncode == 0, rc.stderr or rc.stdout
        manifest = json.loads((out / "transfer-manifest.json").read_text(encoding="utf-8"))
        paths = [entry["path"].replace("\\", "/") for entry in manifest.get("files") or []]
        assert all(not p.startswith("dist/oracle_recursion_guard/") for p in paths)
        assert len(paths) < 500, f"unexpected file explosion: {len(paths)}"
        assert (out / "dist" / "oracle_recursion_guard" / "transfer-manifest.json").exists() is False
    finally:
        import shutil

        if out.is_dir():
            shutil.rmtree(out, ignore_errors=True)


def test_bundle_excludes_pytest_temp_artifacts() -> None:
    """Regression: pytest basetemp under training/ must never enter the bundle."""
    temp_root = ROOT / "training" / ".pytest-temp" / "bundle_guard_marker"
    temp_root.mkdir(parents=True, exist_ok=True)
    marker = temp_root / "must_not_bundle.txt"
    marker.write_text("marker", encoding="utf-8")
    out = ROOT / "dist" / "oracle_pytest_temp_guard"
    try:
        rc = subprocess.run(
            [sys.executable, str(BUILD), "--output", str(out), "--code-only"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert rc.returncode == 0, rc.stderr or rc.stdout
        manifest = json.loads((out / "transfer-manifest.json").read_text(encoding="utf-8"))
        paths = [entry["path"].replace("\\", "/") for entry in manifest.get("files") or []]
        assert not any("must_not_bundle.txt" in p for p in paths)
        assert not any(".pytest-temp/" in p for p in paths)
    finally:
        import shutil

        if out.is_dir():
            shutil.rmtree(out, ignore_errors=True)
        if temp_root.is_dir():
            shutil.rmtree(temp_root, ignore_errors=True)
        (ROOT / "training" / ".pytest-temp").mkdir(parents=True, exist_ok=True)


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
