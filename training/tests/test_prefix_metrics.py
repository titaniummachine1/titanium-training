"""Tests for versioned seeded prefix metrics."""
from __future__ import annotations

import sys
from pathlib import Path

_TRAINING = Path(__file__).resolve().parents[1]
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

from diversity.canonical import CanonicalStateRow
from diversity.prefix_metrics import (
    PREFIX_METRIC_VERSION,
    fixture_prefix_context,
    prefix2_key,
    prefix4_key,
    standard_start_state,
)
from game_opening_gate import contributes_to_n_eff_two_floor


def test_standard_start_identical_prefixes_group():
    ctx = fixture_prefix_context(root_seed_id="std")
    k1 = prefix2_key(ctx, ("e2", "e8"))
    k2 = prefix2_key(ctx, ("e2", "e8"))
    assert k1 == k2


def test_different_seeds_same_moves_do_not_collide():
    ctx_a = fixture_prefix_context(
        root_seed_id="seed-a",
        start_state=CanonicalStateRow("e2,e8", "", "", "10,10", 0),
    )
    ctx_b = fixture_prefix_context(
        root_seed_id="seed-b",
        start_state=CanonicalStateRow("d2,d8", "", "", "10,10", 0),
    )
    k_a = prefix2_key(ctx_a, ("e2", "e8"))
    k_b = prefix2_key(ctx_b, ("e2", "e8"))
    assert k_a != k_b


def test_missing_seed_metadata_invalid():
    ctx = fixture_prefix_context(root_seed_id="")
    assert prefix2_key(ctx, ("e2", "e8")) is None


def test_prefix4_deterministic():
    ctx = fixture_prefix_context(root_seed_id="s1", start_state=standard_start_state())
    moves = ("e2", "e8", "e3", "e7")
    assert prefix4_key(ctx, moves) == prefix4_key(ctx, moves)


def test_garbage_filter_not_n_eff_support():
    assert contributes_to_n_eff_two_floor() is False
    assert PREFIX_METRIC_VERSION == "prefix-metric-v1"
