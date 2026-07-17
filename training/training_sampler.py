#!/usr/bin/env python3
"""Epoch training index sampler (recent replay mix)."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from cache_val_split import position_keys_for_games


def recent_game_ids(
    games_db: Path,
    *,
    window_games: int,
    sources: tuple[str, ...] = (
        "overnight_selfplay",
        "overnight_mixed",
        "selfplay_train",
        "selfplay_verify",
    ),
) -> set[str]:
    if not games_db.is_file() or window_games <= 0:
        return set()
    con = sqlite3.connect(str(games_db), timeout=30)
    try:
        ph = ",".join("?" * len(sources))
        rows = con.execute(
            f"SELECT game_id FROM games WHERE source IN ({ph}) ORDER BY imported_at DESC LIMIT ?",
            (*sources, window_games),
        ).fetchall()
        return {str(r[0]) for r in rows}
    finally:
        con.close()


def row_keys_hex(row_position_keys: list[Any]) -> list[str]:
    from cache_val_split import _pos_key_hex

    return [_pos_key_hex(pk) for pk in row_position_keys]


def mix_train_indices(
    train_indices: np.ndarray,
    row_position_keys: list[Any],
    games_db: Path,
    *,
    recent_fraction: float = 0.0,
    recent_window_games: int = 128,
    seed: int = 42,
) -> np.ndarray:
    """Return train indices for one epoch: mix recent + uniform historical.

    Total length equals len(train_indices). recent_fraction=0 returns a shuffled
  copy of train_indices (baseline behavior).
    """
    n = len(train_indices)
    if n == 0 or recent_fraction <= 0:
        return train_indices.copy()

    recent_games = recent_game_ids(games_db, window_games=recent_window_games)
    recent_keys = position_keys_for_games(games_db, recent_games)
    if not recent_keys:
        return train_indices.copy()

    keys_hex = row_keys_hex(row_position_keys)
    recent_rows = np.array(
        [int(r) for r in train_indices if keys_hex[int(r)] in recent_keys],
        dtype=np.int32,
    )
    hist_rows = np.array(
        [int(r) for r in train_indices if keys_hex[int(r)] not in recent_keys],
        dtype=np.int32,
    )
    n_recent = min(len(recent_rows), max(0, int(round(n * recent_fraction))))
    n_hist = n - n_recent
    rng = np.random.default_rng(seed)
    if n_recent > 0 and len(recent_rows) > 0:
        pick_recent = rng.choice(recent_rows, size=n_recent, replace=len(recent_rows) < n_recent)
    else:
        pick_recent = np.array([], dtype=np.int32)
    if n_hist > 0 and len(hist_rows) > 0:
        pick_hist = rng.choice(hist_rows, size=n_hist, replace=len(hist_rows) < n_hist)
    else:
        pick_hist = np.array([], dtype=np.int32)
    if len(pick_recent) + len(pick_hist) < n:
        # Pad from full train pool if buckets thin.
        pool = train_indices.copy()
        rng.shuffle(pool)
        out = np.concatenate([pick_recent, pick_hist, pool])[:n]
    else:
        out = np.concatenate([pick_recent, pick_hist])
    rng.shuffle(out)
    return out.astype(np.int32)
