"""Tests for DIVERSITY_SPEC_V1 training opening gate."""
from __future__ import annotations

from game_opening_gate import (
    DEPLOY_COLLAPSE_OPENING,
    TRAINING_OPENING_MIN_PREFIX,
    opening_sanity_ok,
    training_opening_ok,
)


def test_training_gate_accepts_diverse_four_ply():
    assert training_opening_ok(["e2", "e8", "d2", "f8"])
    assert training_opening_ok(["d2", "d8", "e3", "e7"])
    assert opening_sanity_ok(["f2", "f8", "e3", "d7"])


def test_training_gate_rejects_wall_first():
    assert not training_opening_ok(["a7h", "e8", "e3", "e7"])
    assert not training_opening_ok(["e2", "d8h"])
    assert not training_opening_ok(["e2", "a8"])


def test_training_gate_minimum_two_plies():
    assert not training_opening_ok(["e2"])
    assert training_opening_ok(["e2", "e8"])
    assert training_opening_ok(["d2", "f8"])


def test_deploy_collapse_signature_is_four_ply():
    assert DEPLOY_COLLAPSE_OPENING == ("e2", "e8", "e3", "e7")
    assert TRAINING_OPENING_MIN_PREFIX == ("e2", "e8")
