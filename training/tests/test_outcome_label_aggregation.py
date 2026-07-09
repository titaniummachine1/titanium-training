#!/usr/bin/env python3
"""Outcome label aggregation for duplicate position keys."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

TRAINING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING))

from db_import import aggregate_outcome_label


def test_single_sample_passthrough():
    assert aggregate_outcome_label([], 1.0) == pytest.approx(1.0)


def test_running_mean_two_wins():
    # prior mean 1.0 with n=2, new sample -1.0 -> (1*2 + -1) / 3
    assert aggregate_outcome_label([1.0, 1.0], -1.0) == pytest.approx(1 / 3)


def test_draw_mix_produces_fractional_label():
    # P0 draw (0) then P0 win (+1) at same board key
    assert aggregate_outcome_label([0.0], 1.0) == pytest.approx(0.5)


def test_three_way_duplicate_not_terminal_game_outcome():
    labels = aggregate_outcome_label([1.0, -1.0], 0.0)
    assert labels == pytest.approx(0.0)
    assert labels != 1.0
    assert labels != -1.0


def test_aggregation_is_intentional_not_sign_error():
    stored = aggregate_outcome_label([-1.0, -1.0, 1.0], -1.0)
    expected_terminal = -1.0
    assert abs(stored - expected_terminal) > 0.15
    assert stored == pytest.approx((-1.0 * 3 + 1.0) / 4)
