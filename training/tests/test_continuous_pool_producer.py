"""Tests for continuous-producer pool behavior."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from continuous_pool import (
    DEFAULT_EPOCH_GAMES,
    ContinuousPool,
    PersistOutcome,
    PoolConfig,
    PoolState,
)
from pool_state_io import load_json, save_json_atomic


def test_epoch_ready_at_1024_games():
    cfg = PoolConfig(batch_games=1024, use_position_trigger=False)
    pool = ContinuousPool(cfg)
    pool._state = PoolState(games_since_epoch=1023)
    assert pool._epoch_ready() is False
    pool._state.games_since_epoch = 1024
    assert pool._epoch_ready() is True
    assert pool._trigger_reason() == "game_count"


def test_failed_teacher_persist_not_counted():
    cfg = PoolConfig(threads=1, batch_games=9999)
    pool = ContinuousPool(cfg)
    pool._state = PoolState(games_since_epoch=5, pool_games_total=10)

    r = {"game_id": "pool_test_1", "moves": ["e2e3"], "mixed": False, "plies": 1, "outcome_p0": 1}

    with patch("sync_overnight_to_teacher.load_synced_ids", return_value=set()), patch(
        "continuous_pool.write_batch"
    ), patch("continuous_pool.open_db", return_value=MagicMock()), patch(
        "continuous_pool.sync_single_game",
        side_effect=RuntimeError("teacher write failed"),
    ):
        with pytest.raises(RuntimeError):
            pool._persist_game(r)

    assert pool._state.games_since_epoch == 5
    assert pool._state.pool_games_total == 10


def test_duplicate_sync_skipped_not_counted():
    cfg = PoolConfig()
    pool = ContinuousPool(cfg)
    pool._state = PoolState(games_since_epoch=3)

    with patch("sync_overnight_to_teacher.load_synced_ids", return_value={"pool_dup"}):
        out = pool._persist_game({"game_id": "pool_dup", "moves": ["a"], "mixed": False})
    assert out == PersistOutcome(0, False)
    pool._record_game({"mixed": False}, out)
    assert pool._state.games_since_epoch == 3


def test_successful_persist_increments_counter():
    cfg = PoolConfig()
    pool = ContinuousPool(cfg)
    pool._state = PoolState()

    with patch("sync_overnight_to_teacher.load_synced_ids", return_value=set()), patch(
        "continuous_pool.write_batch"
    ), patch("continuous_pool.open_db", return_value=MagicMock()), patch(
        "continuous_pool.sync_single_game",
        return_value={"new_positions": 12, "counted": True},
    ):
        out = pool._persist_game(
            {"game_id": "g1", "moves": ["a"], "mixed": False, "outcome_p0": 1}
        )

    assert out.counted is True
    pool._record_game({"mixed": False}, out)
    assert pool._state.games_since_epoch == 1
    assert pool._state.positions_since_epoch == 12


def test_restart_recovers_games_since_epoch(tmp_path: Path, monkeypatch):
    state_path = tmp_path / "continuous_pool_state.json"
    monkeypatch.setattr("continuous_pool.STATE_PATH", state_path)
    save_json_atomic(
        state_path,
        {
            "epoch": 2,
            "pool_games_total": 1500,
            "games_since_epoch": 42,
            "positions_since_epoch": 900,
            "cache_generation": 1405885,
        },
    )
    pool = ContinuousPool(PoolConfig())
    assert pool._state.epoch == 2
    assert pool._state.games_since_epoch == 42
    assert pool._state.pool_games_total == 1500
    assert pool._state.cache_generation == 1405885


def test_atomic_state_roundtrip(tmp_path: Path):
    path = tmp_path / "state.json"
    save_json_atomic(path, {"games_since_epoch": 7, "positions_since_epoch": 100})
    data = load_json(path)
    assert data["games_since_epoch"] == 7
    assert "updated_at" in data


def test_epoch_transition_drains_inflight():
    cfg = PoolConfig(threads=4, batch_games=2, use_position_trigger=False)
    pool = ContinuousPool(cfg)
    pool._state = PoolState(games_since_epoch=2)
    pool._begin_inflight()
    pool._begin_inflight()

    def release():
        time.sleep(0.3)
        pool._end_inflight()
        pool._end_inflight()

    threading.Thread(target=release, daemon=True).start()
    assert pool._drain_inflight(timeout_sec=2.0) is True
    assert pool._inflight == 0


def test_concurrent_persist_serializes_on_teacher_lock():
    cfg = PoolConfig()
    pool = ContinuousPool(cfg)
    order: list[str] = []
    lock_held = threading.Event()

    def fake_sync(game_id, **kwargs):
        order.append(f"start-{game_id}")
        time.sleep(0.05)
        order.append(f"end-{game_id}")
        return {"new_positions": 1, "counted": True}

    with patch("sync_overnight_to_teacher.load_synced_ids", return_value=set()), patch(
        "continuous_pool.write_batch"
    ), patch("continuous_pool.open_db", return_value=MagicMock()), patch(
        "continuous_pool.sync_single_game", side_effect=fake_sync
    ):

        def run_one(gid: str):
            pool._persist_game(
                {"game_id": gid, "moves": ["a"], "mixed": False, "outcome_p0": 1}
            )

        t1 = threading.Thread(target=run_one, args=("gA",))
        t2 = threading.Thread(target=run_one, args=("gB",))
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

    assert order.index("end-gA") < order.index("start-gB") or order.index("end-gB") < order.index(
        "start-gA"
    )


def test_try_epoch_clears_accept_games_during_run():
    cfg = PoolConfig(batch_games=1, use_position_trigger=False)
    pool = ContinuousPool(cfg)
    pool._state = PoolState(games_since_epoch=1)

    seen_blocked = threading.Event()

    def fake_epoch():
        if not pool._accept_games.is_set():
            seen_blocked.set()
        return False

    pool._run_epoch = fake_epoch  # type: ignore[method-assign]
    pool._try_epoch()
    assert seen_blocked.is_set()
    assert pool._accept_games.is_set()
