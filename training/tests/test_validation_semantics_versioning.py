#!/usr/bin/env python3
"""Checkpoint best_val must not compare across validation-semantics changes."""
from __future__ import annotations

import sys
from pathlib import Path

_TRAINING = Path(__file__).resolve().parents[1]
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

from titanium_training.training.trainer import (
    TRAINING_SCHEMA,
    current_validation_semantics,
    validation_semantics_compatible,
)


def test_missing_stamps_are_incompatible():
    ckpt = {"schema": TRAINING_SCHEMA, "best_val": 0.38682}
    assert validation_semantics_compatible(ckpt) is False


def test_matching_stamps_are_compatible():
    sem = current_validation_semantics(validation_manifest_hash="abc")
    ckpt = {"schema": TRAINING_SCHEMA, "best_val": 0.5, **sem}
    assert validation_semantics_compatible(ckpt, sem) is True


def test_phase_classifier_change_breaks_compatibility():
    sem = current_validation_semantics(validation_manifest_hash="abc")
    ckpt = {"schema": TRAINING_SCHEMA, "best_val": 0.5, **sem}
    ckpt["phase_classifier_version"] = "old-walls-v0"
    assert validation_semantics_compatible(ckpt, sem) is False


def test_manifest_mismatch_breaks_compatibility():
    sem = current_validation_semantics(validation_manifest_hash="new")
    ckpt = {
        "schema": TRAINING_SCHEMA,
        "best_val": 0.5,
        **current_validation_semantics(validation_manifest_hash="old"),
    }
    assert validation_semantics_compatible(ckpt, sem) is False


if __name__ == "__main__":
    test_missing_stamps_are_incompatible()
    test_matching_stamps_are_compatible()
    test_phase_classifier_change_breaks_compatibility()
    test_manifest_mismatch_breaks_compatibility()
    print("ok")
