#!/usr/bin/env python3
"""Local Ace epoch15000 teacher label producer (quarantined, not main WDL).

Writes JSONL rows with ka_nn soft values and ka_policy_budget hints derived from
the JS-forward reference harness.  Labels stay in a quarantine path until
validate_teacher_sidecar.py passes held-out gates.

Never feeds raw Ka value directly into HalfPW WDL training.

Usage:
  python training/tools/ka_teacher/ka_local_teacher.py --limit 100 \\
    --out training/data/ka_teacher_quarantine/labels.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_TRAINING = _REPO / "training"
_EXTRACT = _TRAINING / "tools" / "ka_teacher" / "extract_ace_runtime.js"
_HARNESS = _TRAINING / "tools" / "ka_teacher" / "ace_harness.mjs"
GAMES_DB = _TRAINING / "data" / "all_games.db"
DEFAULT_OUT = _TRAINING / "data" / "ka_teacher_quarantine" / "labels.jsonl"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sample_prefixes(limit: int, seed: int) -> list[list[str]]:
    rng = random.Random(seed)
    fixed = [
        [],
        ["e2", "e8"],
        ["e2", "e8", "e3", "e7"],
        ["e2", "e8", "e3", "e7", "e4", "e6"],
    ]
    out = list(fixed)
    conn = sqlite3.connect(GAMES_DB)
    try:
        rows = conn.execute(
            "SELECT moves FROM games WHERE moves IS NOT NULL ORDER BY RANDOM() LIMIT ?",
            (limit * 4,),
        ).fetchall()
    finally:
        conn.close()
    out: list[list[str]] = []
    for (blob,) in rows:
        try:
            moves = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if not isinstance(moves, list) or len(moves) < 2:
            continue
        cut = rng.randint(2, min(len(moves), 20))
        out.append([str(m) for m in moves[:cut]])
        if len(out) >= limit:
            break
    return out or [[]]


def ace_forward(moves: list[str]) -> dict | None:
    cmd = ["node", str(_HARNESS), "--moves", *moves]
    try:
        proc = subprocess.run(cmd, cwd=str(_REPO), capture_output=True, text=True, timeout=120, check=False)
    except subprocess.TimeoutExpired:
        return None
    if proc.returncode != 0:
        return None
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None


def ka_value_stm(value_black: float, ply: int) -> float:
    # Ace value is fixed-black; stm alternates with ply count from startpos.
    stm_is_black = ply % 2 == 0
    v01 = 0.5 + 0.5 * float(value_black)
    stm = v01 if stm_is_black else 1.0 - v01
    return max(-1.0, min(1.0, 2.0 * stm - 1.0))


def policy_entropy(top: list[dict]) -> float:
    import math

    probs = [max(1e-12, float(x.get("prob", 0))) for x in top]
    s = sum(probs) or 1.0
    probs = [p / s for p in probs]
    return -sum(p * math.log(p) for p in probs)


def main() -> int:
    ap = argparse.ArgumentParser(description="Produce quarantined local Ace teacher labels")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    subprocess.run(["node", str(_EXTRACT)], cwd=str(_REPO), check=True)
    prefixes = sample_prefixes(args.limit, args.seed)
    written = 0
    with args.out.open("w", encoding="utf-8") as f:
        for moves in prefixes:
            doc = ace_forward(moves)
            if doc is None:
                continue
            top = doc.get("policy_top") or []
            row = {
                "schema": "ka-local-teacher-v1",
                "created_at": utc_now(),
                "moves": moves,
                "backend": doc.get("backend", "js"),
                "latency_ms": doc.get("latency_ms"),
                "ka_nn": {
                    "value_stm": ka_value_stm(doc.get("value_black", 0.0), int(doc.get("ply", 0))),
                    "confidence": 0.85,
                },
                "ka_policy_budget": {
                    "top_moves": top,
                    "entropy": policy_entropy(top),
                    "search_pressure": max(0.0, 1.0 - policy_entropy(top) / 2.5),
                },
            }
            f.write(json.dumps(row) + "\n")
            written += 1
    print(f"wrote {written} quarantined labels to {args.out}")
    return 0 if written > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
