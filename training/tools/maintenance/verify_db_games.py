#!/usr/bin/env python3
"""Quick integrity check for all_games.db storage."""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

from tools.datagen.datagen import DB_PATH, validate_game, count_pool_games
from titanium_training.store.move_codec import moves_from_row, unpack_moves

def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    n = conn.execute("SELECT COUNT(*) FROM games").fetchone()[0]
    max_id = conn.execute("SELECT MAX(id) FROM games").fetchone()[0]
    no_bin = conn.execute(
        "SELECT COUNT(*) FROM games WHERE moves_bin IS NULL OR length(moves_bin)=0"
    ).fetchone()[0]
    bad_out = conn.execute(
        "SELECT COUNT(*) FROM games WHERE outcome NOT IN (1, -1)"
    ).fetchone()[0]
    short = 0
    for row in conn.execute("SELECT moves, moves_bin FROM games"):
        if len(moves_from_row(row["moves"], row["moves_bin"])) < 8:
            short += 1

    manifest = json.loads((ROOT / "training/data/manifest.json").read_text())
    pool_run = manifest.get("tournament", {}).get("games", 0)

    print("STORAGE VERIFY", DB_PATH.name)
    print(f"  rows={n}  max_id={max_id}  id_gap={'NONE' if n == max_id else 'YES (ok after pruning bad rows)'}")
    print(f"  pool-tagged={count_pool_games()}  manifest_pool_run={pool_run}")
    print(f"  missing_moves_bin={no_bin}  bad_outcome={bad_out}  short_games={short}")

    ka_pool = conn.execute(
        "SELECT s.name, COUNT(*) c FROM games g JOIN sources s ON s.id=g.src_id "
        "WHERE s.name LIKE 'pool-v15-vs-ka-%' OR s.name LIKE 'pool-titanium-v15-vs-ka-%' "
        "GROUP BY s.name"
    ).fetchall()
    print("  ka pool sources:", {r["name"]: r["c"] for r in ka_pool} or "(none)")

    errors = 0
    sample_ids = [max_id, max_id - 1, max_id - 10, 1]
    for gid in sample_ids:
        if gid < 1:
            continue
        row = conn.execute(
            "SELECT g.moves, g.moves_bin, g.outcome, s.name FROM games g "
            "JOIN sources s ON s.id=g.src_id WHERE g.id=?",
            (gid,),
        ).fetchone()
        if not row:
            print(f"  sample id={gid}: MISSING")
            errors += 1
            continue
        moves = moves_from_row(row["moves"], row["moves_bin"])
        err = validate_game(moves, row["outcome"])
        text_ok = bool(row["moves"] and row["moves"].split())
        bin_ok = bool(row["moves_bin"]) and len(row["moves_bin"]) >= 2
        roundtrip = unpack_moves(row["moves_bin"]) if bin_ok else []
        rt_ok = roundtrip == moves if bin_ok else True
        storage_ok = bin_ok or text_ok
        status = "OK" if not err and storage_ok and rt_ok else "BAD"
        if status == "BAD":
            errors += 1
        print(
            f"  sample id={gid} [{row['name']}] plies={len(moves)} outcome={row['outcome']} "
            f"text={'Y' if text_ok else 'N'} bin={'Y' if bin_ok else 'N'} rt={'Y' if rt_ok else 'N'} {status}"
        )
        if err:
            print(f"    validate: {err}")

    conn.close()
    print(f"  result: {'PASS' if errors == 0 and no_bin == 0 and bad_out == 0 and short == 0 else 'ISSUES'}")
    if pool_run != count_pool_games():
        print(
            f"  note: manifest pool_run ({pool_run}) != pool-tagged rows ({count_pool_games()}) — "
            "pool_run is session counter (resets on housekeeping); DB is cumulative."
        )


if __name__ == "__main__":
    main()
