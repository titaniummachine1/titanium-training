#!/usr/bin/env python3
"""Quick games.db / labels.db inspection."""
from __future__ import annotations

import sqlite3
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
GDB = _REPO / "training" / "data" / "canonical" / "games.db"
LDB = _REPO / "training" / "data" / "canonical" / "labels.db"


def main() -> None:
    print("=== games.db ===")
    con = sqlite3.connect(GDB)
    con.row_factory = sqlite3.Row
    total = con.execute("SELECT COUNT(*) FROM games").fetchone()[0]
    print(f"total games: {total:,}")
    print("by source:")
    for r in con.execute("SELECT source, COUNT(*) c FROM games GROUP BY source ORDER BY c DESC"):
        print(f"  {r[0]}: {r[1]:,}")

    print("\nimported_at present on all rows:", con.execute(
        "SELECT COUNT(*) FROM games WHERE imported_at IS NULL OR imported_at=''"
    ).fetchone()[0] == 0)

    print("\nlatest 5 games:")
    for r in con.execute(
        "SELECT game_id, source, imported_at, move_count, outcome_p0 "
        "FROM games ORDER BY imported_at DESC LIMIT 5"
    ):
        print(dict(r))

    overnight = con.execute(
        """
        SELECT game_id, source, imported_at, move_count, outcome_p0
        FROM games
        WHERE source IN ('overnight_selfplay', 'overnight_mixed', 'titanium-selfplay')
           OR source LIKE 'pool%'
           OR game_id LIKE 'pool_%'
        ORDER BY imported_at DESC
        LIMIT 300
        """
    ).fetchall()
    print(f"\ntitanium self-play (latest up to 300): {len(overnight)}")
    if overnight:
        print("  sources:", dict(Counter(r["source"] for r in overnight)))
        print("  newest:", dict(overnight[0]))
        print("  oldest in window:", dict(overnight[-1]))
    con.close()

    print("\n=== labels.db ===")
    con2 = sqlite3.connect(LDB)
    print("positions:", f"{con2.execute('SELECT COUNT(*) FROM positions').fetchone()[0]:,}")
    print("labels:", f"{con2.execute('SELECT COUNT(*) FROM labels').fetchone()[0]:,}")
    print("label sources:")
    for r in con2.execute("SELECT source, COUNT(*) c FROM labels GROUP BY source ORDER BY c DESC"):
        print(f"  {r[0]}: {r[1]:,}")
    con2.close()


if __name__ == "__main__":
    main()
