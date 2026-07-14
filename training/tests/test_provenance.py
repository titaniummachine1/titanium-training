"""Tests for external ingest provenance contract."""
from __future__ import annotations

import sys
from pathlib import Path

_TRAINING = Path(__file__).resolve().parents[1]
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

from diversity.provenance import (
    IngestDecision,
    ImportedRowProvenance,
    SourceCategory,
    fixture_provenance,
    validate_import_provenance,
)


def test_fixture_provenance_accepted_in_prep():
    result = validate_import_provenance(fixture_provenance(), prep_only=True)
    assert result.decision == IngestDecision.REJECT_NOT_GENERATION_APPROVED


def test_missing_provenance_invalid():
    result = validate_import_provenance(None)
    assert result.decision == IngestDecision.REJECT_MISSING_PROVENANCE


def test_legacy_untrusted_invalid():
    prov = fixture_provenance()
    prov = ImportedRowProvenance(**{**prov.to_dict(), "legacy_untrusted": True, "source_kind": SourceCategory.KA_TEACHER_IMPORT})
    result = validate_import_provenance(prov, prep_only=False, generation_approved=True)
    assert result.decision == IngestDecision.INVALID_LEGACY_PROVENANCE


def test_ka_teacher_category_requires_leakage_check():
    prov = fixture_provenance(source_kind=SourceCategory.KA_TEACHER_IMPORT)
    prov = ImportedRowProvenance(**{**prov.to_dict(), "passed_eval_leakage_check": False})
    result = validate_import_provenance(prov, prep_only=False, generation_approved=True)
    assert result.decision == IngestDecision.REJECT_EVALUATION_LEAKAGE


def test_ka_teacher_files_identified():
    ka_dir = _TRAINING / "tools" / "ka_teacher"
    assert (ka_dir / "ka_nn_collect_labels.py").is_file()
    assert (ka_dir / "ka_ab_collect_labels.py").is_file()
    text = (ka_dir / "ka_nn_collect_labels.py").read_text(encoding="utf-8")
    assert "INSERT INTO labels" in text
