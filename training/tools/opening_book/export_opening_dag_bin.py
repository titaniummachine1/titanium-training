#!/usr/bin/env python3
"""Export non_titanium_opening_dag.db → compact binary for WASM embed (QBKG v1)."""
from __future__ import annotations

import argparse
import sqlite3
import struct
from pathlib import Path

MAGIC = b"QBKG"
VERSION = 1
HEADER = struct.Struct("<4sB3xHH")  # magic, version, pad, n_pos, n_edges
POS_HEAD = struct.Struct("<24sH")  # packed_state, edge_count
EDGE = struct.Struct("<BIIII")  # code, visits, wins, losses, draws


def export(db_path: Path, out_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    positions = conn.execute(
        "SELECT position_id, packed_state FROM positions ORDER BY packed_state"
    ).fetchall()
    id_to_packed = {pid: packed for pid, packed in positions}
    edges_by_parent: dict[int, list[tuple]] = {}
    for row in conn.execute(
        "SELECT parent_position_id, move_code_u8, visit_count, wins_stm, losses_stm, draws "
        "FROM edges ORDER BY parent_position_id, move_code_u8"
    ):
        edges_by_parent.setdefault(row[0], []).append(row[1:])
    conn.close()

    body = bytearray()
    total_edges = 0
    for _pid, packed in positions:
        pid_edges = edges_by_parent.get(_pid, [])
        total_edges += len(pid_edges)
        body.extend(POS_HEAD.pack(packed, len(pid_edges)))
        for code, visits, wins, losses, draws in pid_edges:
            body.extend(
                EDGE.pack(int(code), int(visits), int(wins), int(losses), int(draws))
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = HEADER.pack(MAGIC, VERSION, len(positions), total_edges)
    out_path.write_bytes(header + body)
    print(f"wrote {out_path} ({len(header) + len(body)} bytes, {len(positions)} pos, {total_edges} edges)")


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    default_db = root / "training" / "data" / "opening_book" / "non_titanium_opening_dag.db"
    default_out = root / "engine" / "src" / "titanium" / "data" / "non_titanium_opening_dag.bin"
    p = argparse.ArgumentParser()
    p.add_argument("--db", type=Path, default=default_db)
    p.add_argument("--out", type=Path, default=default_out)
    args = p.parse_args()
    if not args.db.is_file():
        raise SystemExit(f"missing DAG db: {args.db}")
    export(args.db, args.out)


if __name__ == "__main__":
    main()
