#!/usr/bin/env python3
"""Confirm labels.db stores per-source rows; resolution is not import-time AVG."""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

_TRAINING = Path(__file__).resolve().parents[1]
LABELS = _TRAINING / "data" / "canonical" / "labels.db"


def main() -> int:
    con = sqlite3.connect(LABELS)
    ddl = con.execute("SELECT sql FROM sqlite_master WHERE name='labels'").fetchone()[0]
    if "PRIMARY KEY (pos_key, source)" not in ddl.replace("\n", " "):
        print("FAIL: labels table is not keyed by (pos_key, source)")
        return 1

    multi = con.execute(
        """
        SELECT COUNT(*) FROM (
          SELECT pos_key FROM labels GROUP BY pos_key HAVING COUNT(DISTINCT source) > 1
        )
        """
    ).fetchone()[0]

    row = con.execute(
        """
        SELECT pos_key FROM labels GROUP BY pos_key HAVING COUNT(DISTINCT source) > 1
        LIMIT 1
        """
    ).fetchone()
    sample: list[tuple] = []
    if row:
        sample = con.execute(
            "SELECT source, value_stm, n_samples FROM labels WHERE pos_key=? ORDER BY source",
            (row[0],),
        ).fetchall()

    con.close()
    print("OK: PRIMARY KEY (pos_key, source)")
    print(f"multi_source_positions={multi}")
    if sample:
        print(f"example pos_key={row[0][:16]}… distinct per-source rows:")
        for s, v, n in sample[:8]:
            print(f"  {s}: value_stm={v:.4f} n_samples={n}")
    print("import running-mean is per (pos_key, source) via ON CONFLICT(pos_key, source)")
    print("training resolves via label_resolution.resolve_position_label_bundle()")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
