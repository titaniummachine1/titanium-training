"""Tests for deterministic Oracle matchup selection."""
from __future__ import annotations

from oracle_game_factory.matchup import (
    GENERATION_MIXED,
    GENERATION_SELFPLAY,
    choose_matchup,
)


def test_choose_matchup_selfplay_when_no_prior():
    choice = choose_matchup("game-1", "aaa", None)
    assert choice.kind == GENERATION_SELFPLAY
    assert choice.p0_hash == choice.p1_hash == "aaa"
    assert choice.opening_exploration is True


def test_choose_matchup_deterministic():
    a = choose_matchup("fixed-id", "current", "previous")
    b = choose_matchup("fixed-id", "current", "previous")
    assert a == b


def test_choose_matchup_mixed_ratio_over_many_ids():
    current, previous = "c" * 64, "p" * 64
    kinds = [choose_matchup(f"g-{i}", current, previous).kind for i in range(1000)]
    selfplay = kinds.count(GENERATION_SELFPLAY)
    mixed = kinds.count(GENERATION_MIXED)
    assert 650 <= selfplay <= 750, f"selfplay={selfplay}"
    assert 250 <= mixed <= 350, f"mixed={mixed}"


def test_mixed_games_balance_sides():
    current, previous = "c" * 64, "p" * 64
    mixed = []
    for i in range(2000):
        c = choose_matchup(f"m-{i}", current, previous)
        if c.kind == GENERATION_MIXED:
            mixed.append(c)
    p0_current = sum(1 for c in mixed if c.p0_hash == current)
    assert len(mixed) > 100
    assert 0.40 * len(mixed) <= p0_current <= 0.60 * len(mixed)
