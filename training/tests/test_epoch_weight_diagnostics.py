from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))

from epoch_weight_diagnostics import EpochWeightDiagnostics


def test_weight_diagnostics_reports_sample_vs_loss_mass_share() -> None:
    diag = EpochWeightDiagnostics()
    diag.record_batch(
        tiers=["ishtar", "ishtar", "titanium_anchored", "titanium_anchored", "titanium_anchored"],
        phases=["opening", "midgame", "opening", "midgame", "endgame"],
        weights=[1.0, 1.0, 0.2, 0.2, 0.2],
    )
    report = diag.format_report()
    assert "ishtar" in report
    assert "titanium_anchored" in report
    assert "40.0% of rows" in report or "60.0% of rows" in report
    assert "% of weighted loss" in report
    assert "phases:" in report
