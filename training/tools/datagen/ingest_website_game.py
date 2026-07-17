#!/usr/bin/env python3
"""Insert one website-finished game into the canonical game store."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from titanium_training.store.config import GAME_STORE_DB  # noqa: E402
from titanium_training.store.website_games import FinishedWebsiteGame, insert_finished_website_game  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True, choices=("1", "-1", "0"))
    ap.add_argument("--source", default="website_finished_game")
    ap.add_argument("--moves", required=True, help="space-separated move list")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    moves = [token for token in args.moves.split() if token]
    if not moves:
        raise SystemExit("no moves provided")
    game_id = insert_finished_website_game(
        FinishedWebsiteGame(moves=tuple(moves), result=int(args.result), source=args.source)
    )
    print(f"OK {game_id} {GAME_STORE_DB}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
