#!/usr/bin/env python3
"""Batch high-quality NNUE value labels via bounded Ace Ka-AB search.

Samples position prefixes from canonical games.db, labels each with
ka_ab_teacher.mjs (node-bounded alpha-beta), and writes JSONL plus optional
labels.db rows (source ``ka_ab_engine``).

Usage:
  python training/tools/ka_teacher/ka_ab_collect_labels.py --limit 256
  python training/tools/ka_teacher/ka_ab_collect_labels.py --continuous --sync-labels-db
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_TRAINING = _REPO / "training"
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

from db_import import (  # noqa: E402
    GAMES_DB_PATH,
    LABELS_DB_PATH,
    LABELS_SCHEMA,
    make_pos_data,
    make_pos_key,
    open_db,
)
from extend_teacher_dataset import float_stm_to_value_i16  # noqa: E402

SCRIPT = Path(__file__).resolve().parent / "ka_ab_teacher.mjs"
DEFAULT_ACE = Path(os.environ.get("KA_ACE", r"C:\Users\Terminatort8000\Downloads\ace.html"))
DEFAULT_OUT = _TRAINING / "data" / "ka_teacher_quarantine" / "ka_ab_labels.jsonl"
LOG_DIR = _TRAINING / "data" / "overnight_logs"
STATE_PATH = LOG_DIR / "ka_ab_labeling_state.json"
SOURCE = "ka_ab_engine"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(msg: str) -> None:
    line = f"[{utc_now()}] {msg}"
    print(line, flush=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with (LOG_DIR / "ka_ab_labeling.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def save_state(payload: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(STATE_PATH)


def sample_prefixes(games_db: sqlite3.Connection, *, limit: int, seed: int) -> list[tuple[str, list[str]]]:
    """Return (game_id, official_move_prefix) samples."""
    rng = random.Random(seed)
    game_rows = games_db.execute(
        """
        SELECT game_id, MAX(move_num) AS max_ply
        FROM game_moves
        GROUP BY game_id
        """
    ).fetchall()
    if not game_rows:
        return []
    rng.shuffle(game_rows)
    out: list[tuple[str, list[str]]] = []
    seen: set[str] = set()
    for game_id, max_ply in game_rows:
        if len(out) >= limit:
            break
        cut = rng.randint(0, min(int(max_ply), 24))
        prefix_rows = games_db.execute(
            """
            SELECT move_alg FROM game_moves
            WHERE game_id=? AND move_num < ?
            ORDER BY move_num
            """,
            (game_id, cut),
        ).fetchall()
        moves = [str(r[0]) for r in prefix_rows]
        key = " ".join(moves)
        if key in seen:
            continue
        seen.add(key)
        out.append((str(game_id), moves))
    if len(out) < limit:
        fixed = [
            ("startpos", []),
            ("opening", ["e2", "e8"]),
            ("mid", ["e2", "e8", "e3", "e7", "e4", "e6"]),
        ]
        for _gid, moves in fixed:
            key = " ".join(moves)
            if key not in seen:
                seen.add(key)
                out.append((_gid, moves))
            if len(out) >= limit:
                break
    rng.shuffle(out)
    return out[:limit]


def node_budget_for_ply(base: int, ply: int) -> int:
    scaled = int(base * (1.0 + ply / 6.0))
    return min(max(base, scaled), 250_000)


def run_ka_ab(
    *,
    ace: Path,
    moves: list[str],
    nodes: int,
    backend: str,
    timeout_sec: float,
) -> dict | None:
    if not SCRIPT.is_file():
        raise FileNotFoundError(SCRIPT)
    if not ace.is_file():
        raise FileNotFoundError(ace)
    cmd = [
        "node",
        str(SCRIPT),
        "--ace",
        str(ace),
        "--nodes",
        str(nodes),
        "--time-ms",
        "0",
        "--backend",
        backend,
    ]
    if moves:
        cmd.append("--moves")
        cmd.extend(moves)
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(_REPO),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None
    if proc.returncode != 0:
        if "exceeded bounded budget" in (proc.stderr or "") and nodes < 250_000:
            return run_ka_ab(
                ace=ace,
                moves=moves,
                nodes=min(nodes * 4, 250_000),
                backend=backend,
                timeout_sec=timeout_sec,
            )
        return None
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None


def eval_prefix(moves: list[str]) -> dict | None:
    from db_import import eval_batch_chunk

    prefix = " ".join(moves)
    recs = eval_batch_chunk([prefix])
    return recs[0] if recs else None


def upsert_label(labels_db: sqlite3.Connection, rec: dict, value_stm: float) -> None:
    key = make_pos_key(rec)
    data = make_pos_data(rec)
    stm = int(rec.get("turn", 0))
    labels_db.execute(
        "INSERT OR IGNORE INTO positions (pos_key, position_data, side_to_move) VALUES (?,?,?)",
        (key, data, stm),
    )
    labels_db.execute(
        """
        INSERT INTO labels (pos_key, source, value_stm, n_samples)
        VALUES (?, ?, ?, 1)
        ON CONFLICT(pos_key, source) DO UPDATE SET
            value_stm = excluded.value_stm,
            n_samples = labels.n_samples + 1
        """,
        (key, SOURCE, float(value_stm)),
    )


def collect_once(
    *,
    games_db: sqlite3.Connection,
    labels_db: sqlite3.Connection | None,
    out_path: Path,
    ace: Path,
    limit: int,
    seed: int,
    base_nodes: int,
    backend: str,
    timeout_sec: float,
    sync_labels_db: bool,
) -> dict[str, int]:
    samples = sample_prefixes(games_db, limit=limit, seed=seed)
    written = skipped = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if out_path.is_file() else "w"
    with out_path.open(mode, encoding="utf-8") as handle:
        for game_id, moves in samples:
            ka = run_ka_ab(
                ace=ace,
                moves=moves,
                nodes=node_budget_for_ply(base_nodes, len(moves)),
                backend=backend,
                timeout_sec=timeout_sec,
            )
            if ka is None:
                skipped += 1
                continue
            board = eval_prefix(moves)
            if board is None:
                skipped += 1
                continue
            value_stm = float(ka["teacher"]["value_stm"])
            row = {
                "schema": "ka-ab-collect-v1",
                "created_at": utc_now(),
                "source": SOURCE,
                "game_id": game_id,
                "moves": moves,
                "ply": len(moves),
                "pos_key": make_pos_key(board),
                "side_to_move": int(board.get("turn", len(moves) % 2)),
                "value_stm": value_stm,
                "value_i16": float_stm_to_value_i16(value_stm),
                "node_budget": ka["budget"]["requested_evals"],
                "actual_evals": ka["budget"]["actual_evals"],
                "best_move": ka["teacher"]["best_move_official"],
                "proven": bool(ka["teacher"].get("proven")),
                "depth": ka["teacher"].get("depth"),
                "ka_ab": ka,
            }
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            written += 1
            if sync_labels_db and labels_db is not None:
                upsert_label(labels_db, board, value_stm)
        if sync_labels_db and labels_db is not None:
            labels_db.commit()
    return {"sampled": len(samples), "written": written, "skipped": skipped}


def main() -> int:
    from prep_guard import guard_real_work

    guard_real_work("labeling", detail="ka_ab_collect_labels.py")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=128, help="positions per batch (0 = use --batch-size only in loop)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--continuous", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--nodes", type=int, default=32_768, help="base Ka-AB eval budget (scaled up by ply)")
    ap.add_argument("--backend", default="auto", choices=["auto", "wasm", "js"])
    ap.add_argument("--timeout-sec", type=float, default=180.0)
    ap.add_argument("--ace", type=Path, default=DEFAULT_ACE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--games-db", type=Path, default=GAMES_DB_PATH)
    ap.add_argument("--sync-labels-db", action="store_true")
    args = ap.parse_args()

    if not args.games_db.is_file():
        log(f"ERROR: games.db missing: {args.games_db}")
        return 2
    if not args.ace.is_file():
        log(f"ERROR: Ace bundle missing: {args.ace}")
        return 2

    labels_db = open_db(LABELS_DB_PATH, LABELS_SCHEMA) if args.sync_labels_db else None
    games_db = sqlite3.connect(args.games_db)
    games_db.row_factory = sqlite3.Row

    total_written = 0
    batch_idx = 0
    try:
        while True:
            batch_idx += 1
            batch_limit = args.batch_size if args.continuous else args.limit
            if batch_limit <= 0:
                break
            seed = args.seed + batch_idx
            log(f"batch {batch_idx}: labeling up to {batch_limit} positions (nodes base={args.nodes})")
            stats = collect_once(
                games_db=games_db,
                labels_db=labels_db,
                out_path=args.out,
                ace=args.ace,
                limit=batch_limit,
                seed=seed,
                base_nodes=args.nodes,
                backend=args.backend,
                timeout_sec=args.timeout_sec,
                sync_labels_db=args.sync_labels_db,
            )
            total_written += stats["written"]
            save_state(
                {
                    "updated_at": utc_now(),
                    "batch_idx": batch_idx,
                    "total_written": total_written,
                    "last_batch": stats,
                    "out": str(args.out),
                    "sync_labels_db": args.sync_labels_db,
                    "nodes_base": args.nodes,
                }
            )
            log(
                f"batch {batch_idx} done: written={stats['written']} skipped={stats['skipped']} "
                f"total={total_written} -> {args.out}"
            )
            if not args.continuous:
                break
            time.sleep(1.0)
    finally:
        games_db.close()
        if labels_db is not None:
            labels_db.close()

    log(f"finished: {total_written} labels written")
    return 0 if total_written > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
