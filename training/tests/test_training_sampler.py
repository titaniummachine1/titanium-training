"""Tests for recent replay training sampler."""
from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import numpy as np

from training_sampler import mix_train_indices


def _mk_db(path: Path) -> None:
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE games (game_id TEXT PRIMARY KEY, source TEXT, outcome_p0 INT, move_count INT, imported_at TEXT);
        CREATE TABLE game_moves (game_id TEXT, move_num INT, pos_key TEXT, move_alg TEXT, PRIMARY KEY(game_id, move_num));
        """
    )
    con.execute(
        "INSERT INTO games VALUES ('recent1','overnight_selfplay',1,1,'2026-06-24T12:00:00Z')"
    )
    con.execute("INSERT INTO game_moves VALUES ('recent1',0,'r1','e2')")
    con.execute(
        "INSERT INTO games VALUES ('old1','wallz',1,1,'2020-01-01T00:00:00Z')"
    )
    con.execute("INSERT INTO game_moves VALUES ('old1',0,'o1','e2')")
    con.commit()
    con.close()


def test_recent_replay_fraction_zero_is_unchanged_size():
    train = np.arange(100, dtype=np.int32)
    row_keys = [f"k{i}" for i in range(100)]
    with tempfile.TemporaryDirectory() as td:
        db = Path(td) / "g.db"
        _mk_db(db)
        out = mix_train_indices(train, row_keys, db, recent_fraction=0.0)
    assert len(out) == len(train)


def test_recent_replay_mix_keeps_epoch_size():
    # rows 0,1 are recent keys; rest historical
    train = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int32)
    row_keys = ["r1", "r1", "o1", "o2", "o3", "o4", "o5", "o6", "o7", "o8"]
    with tempfile.TemporaryDirectory() as td:
        db = Path(td) / "g.db"
        _mk_db(db)
        out = mix_train_indices(
            train, row_keys, db, recent_fraction=0.3, recent_window_games=8, seed=0
        )
    assert len(out) == len(train)
