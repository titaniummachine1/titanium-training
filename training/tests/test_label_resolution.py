#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import pytest

TRAINING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING))

from label_resolution import (
    is_unanimous_pool_outcome,
    merge_outcome_sample,
    resolve_pool_selfplay_target,
    resolve_position_label_bundle,
    resolve_position_labels,
    stm_from_eval_cp,
)
from label_weights import (
    effective_frequency_weight,
    pool_mixed_source_confidence,
    pool_unanimous_source_confidence,
    position_frequency_weight,
)


def test_ishtar_beats_zeroink_and_pool():
    v = resolve_position_labels(
        [
            ("ishtar_engine", 0.7),
            ("zeroink_nn", 0.4),
            ("pool_selfplay_outcome", -1.0, 3),
        ]
    )
    assert v == pytest.approx(0.7)


def test_zeroink_nn_beats_pool():
    v = resolve_position_labels(
        [
            ("zeroink_nn", 0.62),
            ("pool_selfplay_outcome", -1.0, 2),
            ("oracle_mixed_outcome", 1.0),
        ]
    )
    assert v == pytest.approx(0.62)


def test_zeroink_outcome_beats_ka_and_pool():
    v = resolve_position_labels(
        [
            ("zeroink_outcome", 1.0),
            ("pool_vs_ka_outcome", -1.0),
            ("pool_selfplay_outcome", -1.0, 1),
        ]
    )
    assert v == 1.0


def test_ka_beats_pool_selfplay():
    v = resolve_position_labels(
        [
            ("pool_vs_ka_outcome", 1.0),
            ("pool_selfplay_outcome", -1.0, 1),
        ]
    )
    assert v == 1.0


def test_unanimous_pool_outcome_uses_hard_sign():
    assert resolve_pool_selfplay_target(1.0, anchor=None) == 1.0
    assert resolve_pool_selfplay_target(-1.0, anchor=0.3) == -1.0
    assert is_unanimous_pool_outcome(0.999) is True
    assert is_unanimous_pool_outcome(0.5) is False


def test_mixed_pool_outcome_uses_soft_anchor():
    v = resolve_pool_selfplay_target(0.0, anchor=0.30)
    assert v == pytest.approx(0.30)


def test_mixed_pool_outcome_without_anchor_discarded():
    assert resolve_pool_selfplay_target(0.0, anchor=None) is None
    assert resolve_pool_selfplay_target(0.2, anchor=0.04) is None


def test_mixed_pool_outcome_zero_is_not_draw():
    bundle = resolve_position_label_bundle(
        [
            ("pool_selfplay_outcome", 0.0, 2),
            ("pool_selfplay_engine", 0.25, 1),
        ],
        engine_eval_stm=0.25,
        game_phase="midgame",
    )
    assert bundle is not None
    assert bundle.target == pytest.approx(0.25)
    assert bundle.source_tier == "titanium_anchored"
    assert bundle.source_n_samples == 2
    assert bundle.position_occurrence_count == 2


def test_unanimous_pool_bundle_scales_source_confidence_with_n_samples():
    low = resolve_position_label_bundle(
        [("pool_selfplay_outcome", 1.0, 1)],
        engine_eval_stm=0.2,
        game_phase="midgame",
    )
    high = resolve_position_label_bundle(
        [("pool_selfplay_outcome", 1.0, 100_000)],
        engine_eval_stm=0.2,
        game_phase="midgame",
    )
    assert low is not None and high is not None
    assert low.target == 1.0
    assert high.target == 1.0
    assert low.source_confidence == pytest.approx(pool_unanimous_source_confidence(1))
    assert high.source_confidence == pytest.approx(pool_unanimous_source_confidence(100_000))
    assert high.source_confidence > low.source_confidence
    assert high.frequency_weight > low.frequency_weight


def test_mixed_near_fifty_fifty_downweights_weak_anchor():
    conf = pool_mixed_source_confidence(anchor_abs=0.06, outcome_mean=0.0)
    assert conf == pytest.approx(0.35 * 0.75, rel=0.01)


def test_position_and_source_counts_are_separate():
    bundle = resolve_position_label_bundle(
        [
            ("pool_selfplay_outcome", 0.0, 50_000),
            ("pool_generation_selfplay_outcome", 0.0, 50_000),
            ("pool_selfplay_engine", 0.06, 1),
        ],
        engine_eval_stm=0.06,
        game_phase="midgame",
    )
    assert bundle is not None
    assert bundle.position_occurrence_count == 50_000
    assert bundle.source_n_samples == 50_000
    assert bundle.source_confidence < pool_mixed_source_confidence(anchor_abs=0.30, outcome_mean=0.0)


def test_weak_mixed_opening_not_nearly_full_weight():
    bundle = resolve_position_label_bundle(
        [
            ("pool_selfplay_outcome", 0.0, 100_000),
            ("pool_selfplay_engine", 0.06, 1),
        ],
        engine_eval_stm=0.06,
        game_phase="opening",
    )
    assert bundle is not None
    conf = bundle.source_confidence
    freq = position_frequency_weight(100_000)
    eff = effective_frequency_weight(freq, conf)
    assert bundle.loss_weight == pytest.approx(min(eff * conf * 1.25, 0.60), rel=0.05)
    assert bundle.loss_weight < 0.55


def test_excludes_mixed_measurement_sources():
    v = resolve_position_labels(
        [
            ("oracle_mixed_outcome", 1.0),
            ("pool_prior_epoch_outcome", -1.0),
            ("pool_selfplay_outcome", -1.0, 1),
        ]
    )
    assert v == -1.0


def test_merge_outcome_sample_running_mean():
    assert merge_outcome_sample(1.0, 1, -1.0) == pytest.approx(0.0)
    assert merge_outcome_sample(0.0, 2, 1.0) == pytest.approx(1 / 3)


def test_win_rate_from_mean():
  # p(+1 win) = (mean + 1) / 2
    assert (0.0 + 1) / 2 == 0.5
    assert (0.8 + 1) / 2 == pytest.approx(0.9)


def test_titanium_conflicting_low_anchor_discarded_in_bundle():
    bundle = resolve_position_label_bundle(
        [
            ("pool_selfplay_outcome", 0.0, 2),
            ("pool_selfplay_engine", 0.04, 1),
        ],
        engine_eval_stm=0.04,
        game_phase="midgame",
    )
    assert bundle is None


def test_stm_from_eval_cp():
    assert stm_from_eval_cp(400) == pytest.approx(0.761, abs=0.01)
    assert stm_from_eval_cp(-400) == pytest.approx(-0.761, abs=0.01)
