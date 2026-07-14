"""Tests for synthetic diversity rehearsal dry-run."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_TRAINING = Path(__file__).resolve().parents[1]
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

from prep_guard import DRY_RUN_LOG_DIR

import prepare_diversity_rehearsal as rehearsal


def test_rehearsal_writes_only_under_overnight_logs(monkeypatch):
    monkeypatch.setenv("TRAINING_PREP_ONLY", "1")
    out = DRY_RUN_LOG_DIR / "_pytest_rehearsal"
    out.mkdir(parents=True, exist_ok=True)
    try:
        rc = rehearsal.main(
            [
                "--rows",
                "1000",
                "--dry-run",
                "--corpus-id",
                "rehearsal-test",
                "--out-dir",
                str(out),
            ]
        )
        assert rc == 0
        plan = json.loads((out / "diversity_rehearsal_plan.json").read_text(encoding="utf-8"))
        assert plan[rehearsal.BANNER] is True
        assert (out / "diversity_rehearsal_certificate.json").is_file()
        assert (out / "diversity_rehearsal_provenance_report.json").is_file()
    finally:
        for name in (
            "diversity_rehearsal_plan.json",
            "diversity_rehearsal_certificate.json",
            "diversity_rehearsal_manifest.json",
            "diversity_rehearsal_provenance_report.json",
            "diversity_rehearsal_blockers.json",
        ):
            (out / name).unlink(missing_ok=True)


def test_rehearsal_refuses_non_overnight_output(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINING_PREP_ONLY", "1")
    with pytest.raises(SystemExit):
        rehearsal.main(
            [
                "--rows",
                "100",
                "--dry-run",
                "--out-dir",
                str(tmp_path / "bad"),
            ]
        )
