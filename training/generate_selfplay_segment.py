#!/usr/bin/env python3
"""Generate an isolated Titanium self-play segment with an exact position budget.

The segment writes to its own games.db/labels.db by default.  Positions are the
training rows produced by db_import.write_batch: one pre-move position per move.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

_TRAINING = Path(__file__).resolve().parent
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

from db_import import GAMES_SCHEMA, LABELS_SCHEMA, open_db, write_batch
from self_play_overnight import DEFAULT_CURRENT, DEFAULT_PREVIOUS, play_one_game


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--positions", type=int, default=1034)
    ap.add_argument("--time", type=float, default=0.05)
    ap.add_argument("--current", type=Path, default=DEFAULT_CURRENT)
    ap.add_argument("--previous", type=Path, default=DEFAULT_PREVIOUS)
    ap.add_argument("--same-net-pct", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=20260625)
    ap.add_argument("--games-db", type=Path, default=_TRAINING / "data" / "selfplay_isolated" / "games.db")
    ap.add_argument("--labels-db", type=Path, default=_TRAINING / "data" / "selfplay_isolated" / "labels.db")
    ap.add_argument("--out", type=Path, default=_TRAINING / "data" / "selfplay_isolated" / "last_segment.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    target = max(1, int(args.positions))
    total_positions = 0
    games_written = 0
    batch: list[tuple] = []
    reports: list[dict] = []
    started = int(time.time())

    while total_positions < target:
        remaining = target - total_positions
        mixed = rng.random() >= args.same_net_pct
        if mixed:
            current_is_p0 = rng.random() < 0.5
            if current_is_p0:
                w_p0, w_p1 = args.current, args.previous
            else:
                w_p0, w_p1 = args.previous, args.current
        else:
            current_is_p0 = True
            w_p0 = w_p1 = args.current

        gid = f"isolated_selfplay_{started}_{games_written:05d}"
        result = play_one_game(gid, args.time, w_p0, w_p1, mixed, current_is_p0)
        moves = list(result.get("moves") or [])
        if not moves:
            continue
        if len(moves) > remaining:
            moves = moves[:remaining]
            outcome_p0 = 0
        else:
            outcome_p0 = int(result.get("outcome_p0", 0))

        batch.append((gid, moves, outcome_p0, None, "isolated_selfplay"))
        reports.append({
            "game_id": gid,
            "positions": len(moves),
            "mixed": mixed,
            "truncated": len(moves) != int(result.get("plies", len(moves))),
        })
        total_positions += len(moves)
        games_written += 1
        print(f"  segment {games_written}: positions={total_positions}/{target}", flush=True)

    games_db = open_db(args.games_db, GAMES_SCHEMA)
    labels_db = open_db(args.labels_db, LABELS_SCHEMA)
    try:
        n_games, n_pos, n_labels = write_batch(games_db, labels_db, batch, chunk_size=512, workers=1)
    finally:
        games_db.close()
        labels_db.close()

    report = {
        "requested_positions": target,
        "generated_positions": total_positions,
        "games_written": n_games,
        "db_positions": n_pos,
        "labels": n_labels,
        "games_db": str(args.games_db),
        "labels_db": str(args.labels_db),
        "time_sec_per_move": args.time,
        "source": "isolated_selfplay",
        "games": reports,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
