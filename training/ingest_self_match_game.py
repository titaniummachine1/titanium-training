#!/usr/bin/env python3
"""Insert one self-match game into all_games.db (no coordinator required)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))

from datagen import DB_PATH, insert_single_game  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--moves", required=True, help="space-separated moves")
    ap.add_argument("--result", required=True, choices=("W", "B"))
    ap.add_argument("--tag", default="self-match")
    args = ap.parse_args()
    moves = args.moves.split()
    outcome = 1 if args.result == "W" else -1
    gid = insert_single_game(moves, outcome, DB_PATH, args.tag)
    print(gid)


if __name__ == "__main__":
    main()
