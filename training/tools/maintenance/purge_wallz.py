#!/usr/bin/env python3
"""Remove wallz (human) games from canonical games.db / labels.db."""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
GDB = _REPO / "training" / "data" / "canonical" / "games.db"
LDB = _REPO / "training" / "data" / "canonical" / "labels.db"


def counts(con: sqlite3.Connection, table: str, where: str = "") -> int:
    q = f"SELECT COUNT(*) FROM {table}"
    if where:
        q += f" WHERE {where}"
    return int(con.execute(q).fetchone()[0])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--vacuum", action="store_true", help="compact DB after delete (slow on large labels.db)")
    args = ap.parse_args()

    if not GDB.is_file():
        print(f"missing {GDB}")
        return 1

    gcon = sqlite3.connect(GDB)
    lcon = sqlite3.connect(LDB, timeout=120) if LDB.is_file() else None

    wallz_games = counts(gcon, "games", "source='wallz'")
    total_games = counts(gcon, "games")
    selfplay = counts(gcon, "games", "source LIKE 'selfplay%' OR source LIKE 'overnight%'")

    wallz_labels = 0
    if lcon:
        wallz_labels = counts(lcon, "labels", "source='wallz_outcome'")

    print(f"games.db: {total_games:,} total, wallz={wallz_games:,}, titanium/selfplay={selfplay:,}")
    if lcon:
        print(f"labels.db: wallz_outcome={wallz_labels:,}")

    if args.dry_run:
        print("dry-run — no changes")
        gcon.close()
        if lcon:
            lcon.close()
        return 0

    if wallz_games == 0 and wallz_labels == 0:
        print("nothing to remove")
        gcon.close()
        if lcon:
            lcon.close()
        return 0

    print("Removing wallz data...")
    gcon.execute("BEGIN")
    gcon.execute(
        "DELETE FROM game_moves WHERE game_id IN (SELECT game_id FROM games WHERE source='wallz')"
    )
    gcon.execute("DELETE FROM games WHERE source='wallz'")
    gcon.commit()

    if lcon and wallz_labels:
        lcon.execute("BEGIN")
        lcon.execute("DELETE FROM labels WHERE source='wallz_outcome'")
        lcon.commit()
        if args.vacuum:
            lcon.execute("VACUUM")
        lcon.close()

    if args.vacuum:
        gcon.execute("VACUUM")
    gcon.close()

    print(f"Done. Remaining games: {selfplay:,} (selfplay/overnight only in canonical DB)")
    print("Note: active teacher parquet / feature_cache were already wallz-free.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
