#!/usr/bin/env python3
"""Paired opening plies 0-20 review for two Titanium weight files."""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sqlite3
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
TRAINING = ROOT / "training"
if str(TRAINING) not in sys.path:
    sys.path.insert(0, str(TRAINING))

from db_import import GAMES_DB_PATH  # noqa: E402
from streaming_checkpoint_chain import sha256_file  # noqa: E402
from titanium_training.paths import ENGINE_BIN, REPO_ROOT  # noqa: E402

BESTMOVE_RE = re.compile(r"\bbestmove\s+(\(none\)|[a-i][1-9](?:[hv])?)\b")


def _load_prefixes(games_db: Path, *, limit: int, max_ply: int, seed: int) -> list[list[str]]:
    con = sqlite3.connect(str(games_db), timeout=30)
    try:
        game_ids = [
            str(r[0])
            for r in con.execute(
                """
                SELECT game_id
                FROM games
                WHERE move_count >= 4
                ORDER BY imported_at DESC
                LIMIT 1000
                """
            ).fetchall()
        ]
        rng = random.Random(seed)
        rng.shuffle(game_ids)
        prefixes: list[list[str]] = [[]]
        seen = {""}
        for gid in game_ids:
            rows = con.execute(
                "SELECT move_num, move_alg FROM game_moves WHERE game_id=? ORDER BY move_num ASC",
                (gid,),
            ).fetchall()
            moves = [str(r[1]) for r in rows if r[1]]
            for ply in range(1, min(max_ply, len(moves)) + 1):
                prefix = moves[:ply]
                key = " ".join(prefix)
                if key not in seen:
                    seen.add(key)
                    prefixes.append(prefix)
                if len(prefixes) >= limit:
                    return prefixes
        return prefixes[:limit]
    finally:
        con.close()


def _run_genmove(weights: Path, moves: list[str], *, nodes: int, time_sec: float) -> dict[str, Any]:
    env = os.environ.copy()
    env["TITANIUM_NET_WEIGHTS_PATH"] = str(weights.resolve())
    cmd = [
        str(ENGINE_BIN),
        "genmove",
        "--engine",
        "titanium-v16",
        "--book",
        "off",
        "--log",
    ]
    if nodes > 0:
        cmd.extend(["--nodes", str(nodes)])
    else:
        cmd.extend(["--time", str(time_sec)])
    cmd.extend(moves)
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=env,
        timeout=max(30, int(time_sec + 10)),
    )
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    best = None
    m = BESTMOVE_RE.search(text)
    if m:
        best = None if m.group(1) == "(none)" else m.group(1)
    infos: list[dict[str, Any]] = []
    for line in text.splitlines():
        if "info json " not in line:
            continue
        payload = line.split("info json ", 1)[1]
        try:
            infos.append(json.loads(payload))
        except json.JSONDecodeError:
            pass
    pv = ""
    score = None
    nodes_seen = None
    depth = None
    for info in infos:
        if info.get("pv"):
            pv = str(info.get("pv") or pv)
        if "score" in info:
            score = info.get("score")
        if "rootScore" in info:
            score = info.get("rootScore")
        if "nodes" in info:
            nodes_seen = info.get("nodes")
        if "totalNodes" in info:
            nodes_seen = info.get("totalNodes")
        if "depth" in info:
            depth = info.get("depth")
        if "searchDepth" in info:
            depth = info.get("searchDepth")
    return {
        "returncode": proc.returncode,
        "bestmove": best,
        "score": score,
        "pv": pv,
        "nodes": nodes_seen,
        "depth": depth,
        "stderr_tail": "\n".join((proc.stderr or "").splitlines()[-8:]),
    }


def _classify(parent: dict[str, Any], candidate: dict[str, Any]) -> str:
    if parent.get("returncode") != 0 or candidate.get("returncode") != 0:
        return "unclear"
    if parent.get("bestmove") == candidate.get("bestmove"):
        return "same_move"
    ps = parent.get("score")
    cs = candidate.get("score")
    try:
        delta = float(cs) - float(ps)
    except (TypeError, ValueError):
        return "unclear"
    if delta <= -800:
        return "catastrophic_candidate_choice"
    if delta >= 800:
        return "catastrophic_parent_choice"
    if delta >= 150:
        return "candidate_better"
    if delta <= -150:
        return "parent_better"
    return "unclear"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parent", type=Path, required=True)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--games-db", type=Path, default=GAMES_DB_PATH)
    ap.add_argument("--positions", type=int, default=80)
    ap.add_argument("--max-ply", type=int, default=20)
    ap.add_argument("--nodes", type=int, default=5000)
    ap.add_argument("--time-sec", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=38)
    ap.add_argument("--out", type=Path, default=TRAINING / "runs" / "v16" / "opening_plies_0_20_review_epoch38.json")
    args = ap.parse_args()

    parent = args.parent if args.parent.is_absolute() else (ROOT / args.parent).resolve()
    candidate = args.candidate if args.candidate.is_absolute() else (ROOT / args.candidate).resolve()
    games_db = args.games_db if args.games_db.is_absolute() else (ROOT / args.games_db).resolve()
    prefixes = _load_prefixes(games_db, limit=args.positions, max_ply=args.max_ply, seed=args.seed)

    rows: list[dict[str, Any]] = []
    categories: Counter[str] = Counter()
    for idx, moves in enumerate(prefixes):
        p = _run_genmove(parent, moves, nodes=args.nodes, time_sec=args.time_sec)
        c = _run_genmove(candidate, moves, nodes=args.nodes, time_sec=args.time_sec)
        category = _classify(p, c)
        categories[category] += 1
        rows.append(
            {
                "index": idx,
                "ply": len(moves),
                "side_to_move": "p0" if len(moves) % 2 == 0 else "p1",
                "moves": moves,
                "parent": p,
                "candidate": c,
                "same_move": p.get("bestmove") == c.get("bestmove"),
                "category": category,
            }
        )

    suspicious = [
        r
        for r in rows
        if r["category"] in {"parent_better", "catastrophic_candidate_choice"}
        or (not r["same_move"] and str(r["candidate"].get("bestmove") or "").endswith(("h", "v")) and r["ply"] <= 8)
    ]
    report = {
        "title": "OPENING PLIES 0-20 REVIEW",
        "parent": {"path": str(parent), "sha256": sha256_file(parent)},
        "candidate": {"path": str(candidate), "sha256": sha256_file(candidate)},
        "settings": {
            "positions": len(prefixes),
            "max_ply": args.max_ply,
            "nodes": args.nodes,
            "time_sec": args.time_sec if args.nodes <= 0 else None,
            "budget_kind": "nodes" if args.nodes > 0 else "time",
            "book": "off",
            "engine": "titanium-v16",
            "same_positions": True,
        },
        "summary": dict(categories),
        "suspicious_count": len(suspicious),
        "suspicious": suspicious[:25],
        "positions": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("title", "settings", "summary", "suspicious_count")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
