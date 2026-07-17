#!/usr/bin/env python3
"""Collect Titanium-vs-Titanium thinks under the regular match clock.

Clock rule matches tools/binary_match/parallel_engine_match.py:
  - 60s per side per game (default)
  - allotted think = remaining_ms / 20

Writes thinks.jsonl + games.jsonl for later per-think flamegraph replay.
Supports --workers N so multiple games run in parallel (1 engine thread each).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_TRAINING = _REPO / "training"
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

from engine_session import ENGINE_BIN, EngineSession  # noqa: E402


def check_winner(moves: list[str]) -> int | None:
    if not moves:
        return None
    last = moves[-1]
    if last[-1] in ("h", "v"):
        return None
    row = last[-1]
    mover = (len(moves) - 1) % 2
    if mover == 0 and row == "9":
        return 0
    if mover == 1 and row == "1":
        return 1
    return None


def play_game(
    *,
    game: int,
    engine: str,
    engine_bin: Path,
    clock_sec: float,
    max_plies: int,
    threads: int,
) -> tuple[dict, list[dict]]:
    """Return (game_row, think_records). No shared I/O — safe for thread pool."""
    sess_p0 = EngineSession(engine, None, threads=threads, engine_bin=engine_bin)
    sess_p1 = EngineSession(engine, None, threads=threads, engine_bin=engine_bin)
    moves: list[str] = []
    clock_ms = {0: clock_sec * 1000.0, 1: clock_sec * 1000.0}
    termination = "ply_cap"
    winner: int | None = None
    thinks: list[dict] = []

    try:
        for ply in range(max_plies):
            w = check_winner(moves)
            if w is not None:
                winner = w
                termination = "goal"
                break

            side = ply % 2
            sess = sess_p0 if side == 0 else sess_p1
            if clock_ms[side] <= 0.0:
                winner = 1 - side
                termination = "time"
                break
            if not sess.alive():
                winner = 1 - side
                termination = "engine_dead"
                break
            if not sess.sync(moves):
                winner = 1 - side
                termination = "sync_failed"
                break

            remaining_before = clock_ms[side]
            allotted_ms = max(1.0, remaining_before / 20.0)
            move_sec = allotted_ms / 1000.0
            t0 = time.perf_counter()
            mv = sess.go(move_sec)
            used_ms = (time.perf_counter() - t0) * 1000.0
            clock_ms[side] = max(0.0, remaining_before - used_ms)

            thinks.append(
                {
                    "game": game,
                    "ply": ply,
                    "side": side,
                    "moves": list(moves),
                    "allotted_ms": round(allotted_ms, 3),
                    "used_ms": round(used_ms, 3),
                    "move": mv,
                    "remaining_ms_before": round(remaining_before, 3),
                    "remaining_ms_after": round(clock_ms[side], 3),
                }
            )

            if used_ms > remaining_before:
                winner = 1 - side
                termination = "time"
                break
            if not mv:
                winner = 1 - side
                termination = "no_move"
                break
            moves.append(mv)
        else:
            w = check_winner(moves)
            if w is not None:
                winner = w
                termination = "goal"
    finally:
        sess_p0.close()
        sess_p1.close()

    row = {
        "game": game,
        "winner": winner,
        "termination": termination,
        "plies": len(moves),
        "thinks": len(thinks),
        "moves": moves,
        "clocks_sec": {str(k): v / 1000.0 for k, v in clock_ms.items()},
    }
    return row, thinks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--clock-sec", type=float, default=60.0)
    ap.add_argument("--max-plies", type=int, default=300)
    ap.add_argument("--engine", default="titanium-v15")
    ap.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Engine search threads per session (keep 1 when workers>1)",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel games (each game uses 2 sessions × --threads cores)",
    )
    ap.add_argument(
        "--engine-bin",
        type=Path,
        default=None,
        help="Defaults to training ENGINE_BIN / engine/target/release/titanium.exe",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
    )
    args = ap.parse_args()

    engine_bin = args.engine_bin or ENGINE_BIN
    if not engine_bin.is_file():
        print(f"ERROR: titanium binary missing: {engine_bin}", file=sys.stderr)
        print(
            "Build first: $env:RUSTFLAGS='-C target-cpu=native'; cargo build --release -p titanium",
            file=sys.stderr,
        )
        return 2

    workers = max(1, args.workers)
    # 4 parallel games × 2 sides × 1 thread ≈ 8 processes; warn if oversubscribed.
    approx_procs = workers * 2 * max(1, args.threads)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_dir = args.out_dir or (
        _REPO / "training" / "data" / "profiles" / f"tt_per_think_{stamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    thinks_path = out_dir / "thinks.jsonl"
    games_path = out_dir / "games.jsonl"
    meta_path = out_dir / "collect_meta.json"

    meta = {
        "games": args.games,
        "clock_sec": args.clock_sec,
        "engine": args.engine,
        "engine_bin": str(engine_bin.resolve()),
        "threads": args.threads,
        "workers": workers,
        "approx_engine_procs": approx_procs,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "allotment_rule": "remaining_ms / 20",
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(
        f"collect: games={args.games} workers={workers} threads/session={args.threads} "
        f"clock={args.clock_sec}s/side engine={args.engine} "
        f"(~{approx_procs} engine procs)",
        flush=True,
    )
    print(f"out: {out_dir}", flush=True)

    results: dict[int, tuple[dict, list[dict]]] = {}
    t_all = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {
            pool.submit(
                play_game,
                game=g,
                engine=args.engine,
                engine_bin=engine_bin,
                clock_sec=args.clock_sec,
                max_plies=args.max_plies,
                threads=args.threads,
            ): g
            for g in range(args.games)
        }
        for fut in as_completed(futs):
            g = futs[fut]
            row, thinks = fut.result()
            results[g] = (row, thinks)
            print(
                f"game {g}: winner={row['winner']} term={row['termination']} "
                f"plies={row['plies']} thinks={row['thinks']}",
                flush=True,
            )

    with thinks_path.open("w", encoding="utf-8") as thinks_fp, games_path.open(
        "w", encoding="utf-8"
    ) as games_fp:
        for g in range(args.games):
            row, thinks = results[g]
            games_fp.write(json.dumps(row, separators=(",", ":")) + "\n")
            for rec in thinks:
                thinks_fp.write(json.dumps(rec, separators=(",", ":")) + "\n")

    wall = time.perf_counter() - t_all
    meta["finished_utc"] = datetime.now(timezone.utc).isoformat()
    meta["wall_sec"] = round(wall, 2)
    meta["total_thinks"] = sum(len(t) for _, t in results.values())
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(
        f"DONE wall={wall:.1f}s thinks={meta['total_thinks']} -> {thinks_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
