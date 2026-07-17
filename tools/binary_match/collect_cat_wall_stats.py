#!/usr/bin/env python3
"""Offline CAT wall-participation instrumentation (no pruning).

Samples position prefixes and records per-legal-wall CAT/LMR metadata via the
engine `lmr` JSON snapshot.  Use this to see whether v16 tail-LMR already
concentrates search on hot walls before enabling any hard top-K deletion.

Usage:
  python tools/binary_match/collect_cat_wall_stats.py --limit 200 --depth 8 \\
    --out training/data/cat_wall_stats.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_TRAINING = _REPO / "training"
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

from titanium_training.paths import ENGINE_BIN, REPO_ROOT

GAMES_DB = _TRAINING / "data" / "all_games.db"


def sample_prefixes(limit: int, rng: random.Random) -> list[list[str]]:
    fixed = [
        [],
        ["e2", "e8"],
        ["e2", "e8", "e3", "e7"],
        ["e2", "e8", "e3", "e7", "e4", "e6"],
    ]
    out = list(fixed)
    if not GAMES_DB.is_file():
        return out[:limit]
    conn = sqlite3.connect(GAMES_DB)
    try:
        rows = conn.execute(
            "SELECT moves FROM games WHERE moves IS NOT NULL AND length(moves) > 2 ORDER BY RANDOM() LIMIT ?",
            (limit * 3,),
        ).fetchall()
    finally:
        conn.close()
    out: list[list[str]] = []
    for (moves_blob,) in rows:
        try:
            moves = json.loads(moves_blob)
        except json.JSONDecodeError:
            continue
        if not isinstance(moves, list) or len(moves) < 4:
            continue
        cut = rng.randint(4, min(len(moves), 24))
        out.append([str(m) for m in moves[:cut]])
        if len(out) >= limit:
            break
    return out or [[]]


def lmr_snapshot(moves: list[str], depth: int, time_sec: float) -> dict | None:
    cmd = [
        str(ENGINE_BIN),
        "lmr",
        *moves,
        "--depth",
        str(depth),
        "--time",
        str(time_sec),
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None
    if proc.returncode != 0:
        return None
    text = proc.stdout.strip()
    if not text:
        return None
    try:
        return json.loads(text.splitlines()[-1])
    except json.JSONDecodeError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Collect CAT wall LMR participation stats")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--time", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument(
        "--out",
        type=Path,
        default=_TRAINING / "data" / "cat_wall_stats.jsonl",
    )
    args = ap.parse_args()
    if not ENGINE_BIN.is_file():
        print(f"engine missing: {ENGINE_BIN}", file=sys.stderr)
        return 2

    rng = random.Random(args.seed)
    prefixes = sample_prefixes(args.limit, rng)
    if len(prefixes) < args.limit:
        prefixes.extend([[]] * (args.limit - len(prefixes)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.out.open("w", encoding="utf-8") as f:
        for moves in prefixes:
            snap = lmr_snapshot(moves, args.depth, args.time)
            if snap is None:
                continue
            walls = []
            for entry in snap.get("moves", []):
                if entry.get("kind") != "wall":
                    continue
                walls.append(
                    {
                        "move": entry.get("move"),
                        "heat_cm": entry.get("catCm"),
                        "attention_ratio": entry.get("attentionRatio"),
                        "child_depth": entry.get("childDepthUsed"),
                        "v16_override": entry.get("hardOverride"),
                        "rank": entry.get("order"),
                        "dead_tail": entry.get("deadTail"),
                        "cold": entry.get("cold"),
                    }
                )
            if not walls:
                continue
            row = {
                "moves": moves,
                "depth": args.depth,
                "wall_count": len(walls),
                "walls": walls,
                "dead_tail_count": sum(1 for w in walls if w.get("dead_tail")),
                "depth_one_count": sum(1 for w in walls if w.get("child_depth") == 1),
                "cold_wall_count": sum(1 for w in walls if w.get("cold")),
            }
            f.write(json.dumps(row) + "\n")
            written += 1
    print(f"wrote {written} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
