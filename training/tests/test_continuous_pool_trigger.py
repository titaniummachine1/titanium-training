"""Epoch trigger tests for continuous pool."""
from __future__ import annotations

from continuous_pool import DEFAULT_EPOCH_GAMES, PoolConfig, PoolState, ContinuousPool


def test_position_trigger_when_enabled():
    cfg = PoolConfig(batch_games=9999, train_after_new_positions=100, use_position_trigger=True)
    pool = ContinuousPool(cfg)
    pool._state = PoolState(games_since_epoch=1, positions_since_epoch=100)
    assert pool._epoch_ready() is True
    assert pool._trigger_reason() == "position_threshold"


def test_game_count_at_1024():
    cfg = PoolConfig(batch_games=DEFAULT_EPOCH_GAMES, use_position_trigger=False)
    pool = ContinuousPool(cfg)
    pool._state = PoolState(games_since_epoch=DEFAULT_EPOCH_GAMES)
    assert pool._epoch_ready() is True
    assert pool._trigger_reason() == "game_count"
