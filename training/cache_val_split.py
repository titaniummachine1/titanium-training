#!/usr/bin/env python3
"""Stable train/validation splits for feature-cache and streaming training.

Game-linked positions (from games.db) are assigned by whole game_id.  Other
positions are assigned directly from their canonical position-key hash.  The
manifest is an audit record plus a small set of forced fallback assignments;
it is deliberately *not* the authority for all future positions.  Otherwise a
long-running streaming job would keep evaluating only the finite set that was
present when the manifest was first created.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from titanium_training.data.split import _split_bucket

VAL_MANIFEST = "val_manifest.json"
VAL_FRAC_DEFAULT = 0.05
SPLIT_SEED_DEFAULT = 42
SPLIT_ALGORITHM_V2 = "game_id_whole_game_plus_hash_position_key_v2"


def _pos_key_hex(key: Any) -> str:
    if isinstance(key, bytes):
        return key.hex()
    if isinstance(key, str):
        return key if all(c in "0123456789abcdef" for c in key.lower()) else key.encode().hex()
    return bytes(key).hex()


def _pos_key_bytes(key: Any) -> bytes:
    if isinstance(key, bytes):
        return key
    if isinstance(key, str):
        try:
            return bytes.fromhex(key)
        except ValueError:
            return key.encode()
    return bytes(key)


def _game_bucket(game_id: str, seed: int) -> float:
    return _split_bucket(game_id.encode("utf-8"), seed)


def _hash_selected_game_ids(
    game_ids: list[str], *, val_fraction: float, seed: int
) -> set[str]:
    """Select only the stateless hash bucket; fallback handling is separate."""
    if val_fraction <= 0:
        return set()
    return {gid for gid in game_ids if _game_bucket(gid, seed) < val_fraction}


def _hash_selected_position_keys(
    position_keys: list[Any], *, val_fraction: float, seed: int
) -> set[str]:
    """Return canonical keys assigned to validation without cohort dependence."""
    if val_fraction <= 0:
        return set()
    return {
        _pos_key_hex(key)
        for key in position_keys
        if _split_bucket(_pos_key_bytes(key), seed) < val_fraction
    }


def select_val_game_ids(
    game_ids: list[str],
    *,
    val_fraction: float = VAL_FRAC_DEFAULT,
    seed: int = SPLIT_SEED_DEFAULT,
) -> set[str]:
    """Deterministic whole-game validation selection."""
    if not game_ids or val_fraction <= 0:
        return set()
    chosen = {gid for gid in game_ids if _game_bucket(gid, seed) < val_fraction}
    if not chosen and game_ids:
        chosen = {min(game_ids, key=lambda g: _game_bucket(g, seed ^ 0xA5A5))}
    return chosen


def load_game_ids(games_db: Path) -> list[str]:
    if not games_db.is_file():
        return []
    con = sqlite3.connect(str(games_db), timeout=30)
    try:
        rows = con.execute("SELECT game_id FROM games ORDER BY game_id").fetchall()
        return [str(r[0]) for r in rows]
    finally:
        con.close()


def position_keys_for_games(games_db: Path, game_ids: set[str]) -> set[str]:
    """All position keys (hex) appearing in the given games."""
    if not game_ids or not games_db.is_file():
        return set()
    con = sqlite3.connect(str(games_db), timeout=60)
    try:
        keys: set[str] = set()
        for gid in game_ids:
            rows = con.execute(
                "SELECT DISTINCT pos_key FROM game_moves WHERE game_id=?",
                (gid,),
            ).fetchall()
            keys.update(str(r[0]) for r in rows)
        return keys
    finally:
        con.close()


def load_val_manifest(cache_dir: Path) -> dict[str, Any] | None:
    path = cache_dir / VAL_MANIFEST
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_val_manifest(cache_dir: Path, manifest: dict[str, Any]) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / VAL_MANIFEST
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def build_val_position_keys(
    *,
    cache_dir: Path,
    position_keys_in_order: list[Any],
    games_db: Path,
    val_fraction: float = VAL_FRAC_DEFAULT,
    seed: int = SPLIT_SEED_DEFAULT,
    recent_selfplay_sources: tuple[str, ...] = (
        "overnight_selfplay",
        "overnight_mixed",
        "selfplay_train",
        "selfplay_verify",
    ),
) -> tuple[set[str], dict[str, Any]]:
    """Return validation keys + an audit manifest.

    The normal assignment is stateless and therefore gives every newly arriving
    canonical position the same answer regardless of which streaming epoch first
    contains it.  Forced keys exist solely to guarantee a non-empty validation
    set in very small cohorts and to preserve assignments from old manifests.
    """
    existing = load_val_manifest(cache_dir)
    all_game_ids = load_game_ids(games_db)
    is_v2 = bool(existing and existing.get("split_algorithm") == SPLIT_ALGORITHM_V2)

    # A pre-v2 manifest was the split authority.  Pin its selected rows once so
    # an upgrade cannot move a previously validated position back to training.
    # New manifests keep only genuine fallback pins here; normal membership is
    # always recomputed from the stable hash.
    forced_val_keys = set((existing or {}).get("forced_val_position_keys_hex") or [])
    forced_val_games = set((existing or {}).get("forced_val_game_ids") or [])
    if existing and not is_v2:
        forced_val_keys.update(existing.get("val_position_keys_hex") or [])
        forced_val_games.update(existing.get("val_game_ids") or [])

    hash_val_games = _hash_selected_game_ids(
        all_game_ids, val_fraction=val_fraction, seed=seed
    )
    if not hash_val_games and not forced_val_games and all_game_ids and val_fraction > 0:
        forced_val_games.add(
            min(all_game_ids, key=lambda g: _game_bucket(g, seed ^ 0xA5A5))
        )
    val_games = hash_val_games | forced_val_games
    val_keys = position_keys_for_games(games_db, val_games)

    recent_val_games: set[str] = set()
    if games_db.is_file():
        con = sqlite3.connect(str(games_db), timeout=30)
        try:
            ph = ",".join("?" * len(recent_selfplay_sources))
            rows = con.execute(
                f"SELECT game_id FROM games WHERE source IN ({ph}) ORDER BY imported_at DESC LIMIT 64",
                recent_selfplay_sources,
            ).fetchall()
            recent_candidates = [str(r[0]) for r in rows]
            # This is intentionally hash-only.  A rolling "recent" window must
            # not create a different fallback game whenever its membership
            # changes, which would make the split cohort-dependent.
            recent_val_games = _hash_selected_game_ids(
                recent_candidates,
                val_fraction=max(val_fraction, 0.1),
                seed=seed ^ 0xBEEF,
            )
            val_keys |= position_keys_for_games(games_db, recent_val_games)
        finally:
            con.close()

    # Position keys in game_moves can use a different key format (for example an
    # 8-byte fast hash) than the teacher dataset's canonical 16-byte key.  Hash
    # assignment therefore also covers canonical positions that cannot be joined
    # to a game row.  Unlike the old rank-based fill, this remains stable as new
    # positions arrive.
    cache_key_hexes = {_pos_key_hex(pk) for pk in position_keys_in_order}
    val_keys |= _hash_selected_position_keys(
        position_keys_in_order, val_fraction=val_fraction, seed=seed
    )
    val_keys |= forced_val_keys

    # A tiny cohort can legitimately miss its hash bucket.  Pin a deterministic
    # fallback key in the manifest so later epochs never silently substitute the
    # first row they happen to receive.
    if val_fraction > 0 and cache_key_hexes and not (val_keys & cache_key_hexes):
        fallback = min(
            position_keys_in_order,
            key=lambda key: _split_bucket(_pos_key_bytes(key), seed ^ 0xA5A5),
        )
        fallback_hex = _pos_key_hex(fallback)
        forced_val_keys.add(fallback_hex)
        val_keys.add(fallback_hex)

    manifest = {
        "split_algorithm": SPLIT_ALGORITHM_V2,
        "split_seed": seed,
        "val_fraction": val_fraction,
        "val_game_ids": sorted(val_games),
        "recent_val_game_ids": sorted(recent_val_games),
        "forced_val_game_ids": sorted(forced_val_games),
        "forced_val_position_keys_hex": sorted(forced_val_keys),
        "val_position_keys_hex": sorted(val_keys),
        "n_val_games": len(val_games),
        "n_recent_val_games": len(recent_val_games),
        "n_val_position_keys": len(val_keys),
    }
    save_val_manifest(cache_dir, manifest)
    return val_keys, manifest


def assign_train_val_row_indices(
    row_position_keys: list[Any],
    val_position_keys: set[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Map cache rows to train/val index arrays (whole positions only)."""
    val_rows: list[int] = []
    train_rows: list[int] = []
    val_set = set(val_position_keys)
    for i, pk in enumerate(row_position_keys):
        hx = _pos_key_hex(pk)
        if hx in val_set:
            val_rows.append(i)
        else:
            train_rows.append(i)
    if not val_rows and train_rows:
        val_rows = [train_rows.pop(0)]
    rng = np.random.default_rng(SPLIT_SEED_DEFAULT)
    train_arr = rng.permutation(np.array(train_rows, dtype=np.int32))
    val_arr = np.array(val_rows, dtype=np.int32)
    return train_arr, val_arr


def recent_val_row_indices(
    row_position_keys: list[Any],
    games_db: Path,
    recent_val_game_ids: set[str],
) -> np.ndarray:
    """Rows belonging to recent self-play validation games."""
    if not recent_val_game_ids:
        return np.array([], dtype=np.int32)
    recent_keys = position_keys_for_games(games_db, recent_val_game_ids)
    if not recent_keys:
        return np.array([], dtype=np.int32)
    rows = [i for i, pk in enumerate(row_position_keys) if _pos_key_hex(pk) in recent_keys]
    return np.array(rows, dtype=np.int32)
