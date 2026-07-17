"""Tests for game-stable cache validation split."""
from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pytest

from cache_val_split import (
    assign_train_val_row_indices,
    build_val_position_keys,
    load_val_manifest,
    select_val_game_ids,
)
from titanium_training.data.split import _split_bucket


def _key_in_bucket(*, seed: int, val_fraction: float, selected: bool) -> str:
    for i in range(1, 100_000):
        key = f"{i:032x}"
        if (_split_bucket(bytes.fromhex(key), seed) < val_fraction) == selected:
            return key
    raise AssertionError("could not find a deterministic split key")


def test_select_val_game_ids_whole_game_stable():
    games = [f"g{i:04d}" for i in range(200)]
    a = select_val_game_ids(games, val_fraction=0.1, seed=42)
    b = select_val_game_ids(games, val_fraction=0.1, seed=42)
    assert a == b
    assert 0 < len(a) < len(games)


def test_no_position_in_both_splits():
    keys = [f"{i:032x}" for i in range(100)]
    val = set(keys[:10])
    train_idx, val_idx = assign_train_val_row_indices(keys, val)
    assert len(set(train_idx) & set(val_idx)) == 0
    assert len(val_idx) == 10


def test_val_manifest_persists(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    keys = [bytes([i] * 16) for i in range(50)]
    val_keys, manifest = build_val_position_keys(
        cache_dir=cache_dir,
        position_keys_in_order=keys,
        games_db=tmp_path / "missing.db",
        val_fraction=0.1,
        seed=7,
    )
    assert load_val_manifest(cache_dir) is not None
    val2, manifest2 = build_val_position_keys(
        cache_dir=cache_dir,
        position_keys_in_order=keys,
        games_db=tmp_path / "missing.db",
        val_fraction=0.1,
        seed=7,
    )
    assert val_keys == val2
    assert manifest["val_position_keys_hex"] == manifest2["val_position_keys_hex"]


def test_existing_manifest_hashes_new_canonical_positions(tmp_path: Path):
    """New rows must not be excluded merely because an old manifest exists."""
    cache_dir = tmp_path / "cache"
    seed = 7
    fraction = 0.2
    old_key = _key_in_bucket(seed=seed, val_fraction=fraction, selected=False)
    new_val_key = _key_in_bucket(seed=seed, val_fraction=fraction, selected=True)

    first_val, _ = build_val_position_keys(
        cache_dir=cache_dir,
        position_keys_in_order=[old_key],
        games_db=tmp_path / "missing.db",
        val_fraction=fraction,
        seed=seed,
    )
    # The first one is a persisted small-cohort fallback, not a hash hit.
    assert old_key in first_val

    expanded_val, manifest = build_val_position_keys(
        cache_dir=cache_dir,
        position_keys_in_order=[old_key, new_val_key],
        games_db=tmp_path / "missing.db",
        val_fraction=fraction,
        seed=seed,
    )
    assert old_key in expanded_val
    assert new_val_key in expanded_val
    assert new_val_key in manifest["val_position_keys_hex"]


def test_game_linked_positions_stay_together(tmp_path: Path):
    gdb = tmp_path / "games.db"
    con = sqlite3.connect(gdb)
    con.executescript(
        """
        CREATE TABLE games (game_id TEXT PRIMARY KEY, source TEXT, outcome_p0 INT, move_count INT, imported_at TEXT);
        CREATE TABLE positions (pos_key TEXT PRIMARY KEY, position_data BLOB, side_to_move INT);
        CREATE TABLE game_moves (game_id TEXT, move_num INT, pos_key TEXT, move_alg TEXT, PRIMARY KEY(game_id, move_num));
        """
    )
    con.execute("INSERT INTO games VALUES ('g1','selfplay_train',1,2,'t')")
    con.execute("INSERT INTO positions VALUES ('aa','{}',0)")
    con.execute("INSERT INTO positions VALUES ('bb','{}',1)")
    con.execute("INSERT INTO game_moves VALUES ('g1',0,'aa','e2')")
    con.execute("INSERT INTO game_moves VALUES ('g1',1,'bb','e8')")
    con.commit()
    con.close()

    cache_dir = tmp_path / "cache"
    keys = ["aa", "bb", "cc", "dd"]
    val_keys, _ = build_val_position_keys(
        cache_dir=cache_dir,
        position_keys_in_order=keys,
        games_db=gdb,
        val_fraction=0.5,
        seed=1,
    )
    # If g1 is in val, both aa and bb must be val together.
    if "aa" in val_keys or "bb" in val_keys:
        assert "aa" in val_keys and "bb" in val_keys
