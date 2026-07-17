"""Canonical website-finished-game ingest helpers.

This module is intentionally tiny: website/public games contribute only
move-sequence + verdict to the canonical replayable game store.
"""
from __future__ import annotations

from dataclasses import dataclass

from titanium_training.store.config import GAME_STORE_DB
from titanium_training.store.lib import connect_db, insert_game


@dataclass(frozen=True)
class FinishedWebsiteGame:
    moves: tuple[str, ...]
    result: int
    source: str = "website_finished_game"

    def validate(self) -> None:
        if not self.moves:
            raise ValueError("moves must be non-empty")
        if self.result not in (-1, 0, 1):
            raise ValueError("result must be one of -1, 0, 1")
        if not self.source.strip():
            raise ValueError("source must be non-empty")


def parse_text_wire(text: str) -> FinishedWebsiteGame:
    raw = str(text or "").strip()
    if not raw:
        raise ValueError("empty website game payload")
    lines = raw.splitlines()
    if lines[0].strip() != "TI-GAME-1":
        raise ValueError("unknown website game wire format")
    values: dict[str, str] = {}
    for line in lines[1:]:
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    moves = tuple(token for token in values.get("moves", "").split() if token)
    result = int(values["result"])
    source = values.get("source") or "website_finished_game"
    game = FinishedWebsiteGame(moves=moves, result=result, source=source)
    game.validate()
    return game


def insert_finished_website_game(game: FinishedWebsiteGame) -> int:
    game.validate()
    conn = connect_db(GAME_STORE_DB)
    try:
        conn.execute("PRAGMA busy_timeout=750")
        game_id = insert_game(
            conn,
            list(game.moves),
            game.result,
            source=game.source,
            source_cohort="website_finished_game",
            metadata=None,
        )
        conn.commit()
        return game_id
    finally:
        conn.close()
