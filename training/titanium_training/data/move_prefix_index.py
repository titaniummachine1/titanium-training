"""Build packed_state -> move-prefix index from canonical game store replays."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from titanium_training.store.lib import connect_db
from titanium_training.store.state import replay_game


def build_game_store_prefix_index(db_path: Path) -> dict[bytes, tuple[str, ...]]:
    """Map packed_state bytes to algebraic move prefix reachable from game starts."""
    if not db_path.is_file():
        raise FileNotFoundError(f"game store missing: {db_path}")

    conn = connect_db(db_path)
    has_paths = bool(
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='game_paths'"
        ).fetchone()
    )
    if not has_paths:
        conn.close()
        return {}

    index: dict[bytes, tuple[str, ...]] = {}
    rows = conn.execute(
        "SELECT gp.packed_u8_move_sequence FROM games g "
        "JOIN game_paths gp ON gp.game_id=g.game_id "
        "WHERE g.result IS NOT NULL"
    ).fetchall()
    conn.close()

    from titanium_training.store.state import moves_from_u8_blob

    for (blob,) in rows:
        if not blob:
            continue
        moves = moves_from_u8_blob(bytes(blob))
        for ply, state in enumerate(replay_game(moves)):
            index[state.packed_state()] = tuple(moves[:ply])
    return index
