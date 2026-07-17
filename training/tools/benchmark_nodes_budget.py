#!/usr/bin/env python3
"""Measure Titanium v15 nodes/move at a fixed wall-clock budget (self-play parity).

One fresh engine process per position. Parses `info json` from stderr (same as
production genmove without --log).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import sqlite3
import statistics
import subprocess
import sys
import time
from pathlib import Path

_TRAINING = Path(__file__).resolve().parents[1]
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

try:
    from titanium_training.paths import ENGINE_BIN as _ENGINE_BIN, REPO_ROOT
except ImportError:
    REPO_ROOT = _TRAINING.parent
    _ENGINE_BIN = REPO_ROOT / "engine" / "target" / "release" / (
        "titanium.exe" if os.name == "nt" else "titanium"
    )

GAMES_DB = _TRAINING / "data" / "canonical" / "games.db"
DEFAULT_WEIGHTS = _TRAINING / "runs" / "value_oracle" / "net_weights_best.bin"
ENGINE_BIN = _ENGINE_BIN
ENGINE_NAME = "titanium-v15"
INFO_JSON_RE = re.compile(r"^info json (\{.*\})\s*$")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hardware_id() -> str:
    return platform.platform()


def sample_positions(db_path: Path, *, count: int, seed: int) -> list[dict]:
    """Sample mid-game positions from recent canonical games."""
    import random

    rng = random.Random(seed)
    con = sqlite3.connect(str(db_path))
    try:
        games = [
            r[0]
            for r in con.execute(
                """
                SELECT game_id FROM games
                WHERE move_count >= 12 AND move_count <= 120
                ORDER BY imported_at DESC
                LIMIT 500
                """
            ).fetchall()
        ]
        if not games:
            raise RuntimeError(f"no suitable games in {db_path}")

        positions: list[dict] = []
        seen: set[tuple[str, int]] = set()
        attempts = 0
        while len(positions) < count and attempts < count * 50:
            attempts += 1
            gid = rng.choice(games)
            rows = con.execute(
                "SELECT move_num, move_alg FROM game_moves WHERE game_id=? ORDER BY move_num",
                (gid,),
            ).fetchall()
            if len(rows) < 8:
                continue
            # skip terminal plies (last 4 moves often race/terminal)
            max_ply = min(len(rows) - 4, 80)
            min_ply = 4
            if max_ply <= min_ply:
                continue
            ply = rng.randint(min_ply, max_ply)
            key = (gid, ply)
            if key in seen:
                continue
            seen.add(key)
            moves = [m for _, m in rows[:ply]]
            positions.append(
                {
                    "game_id": gid,
                    "ply": ply,
                    "move_count": len(moves),
                    "moves": moves,
                }
            )
        if len(positions) < count:
            raise RuntimeError(f"only sampled {len(positions)}/{count} positions")
        return positions
    finally:
        con.close()


def run_one(
    *,
    engine_bin: Path,
    weights: Path,
    moves: list[str],
    time_sec: float,
    timeout_sec: float,
) -> dict:
    env = os.environ.copy()
    env["TITANIUM_NET_WEIGHTS_PATH"] = str(weights.resolve())
    cmd = [str(engine_bin), "genmove", "--engine", ENGINE_NAME, *moves, "--time", str(time_sec)]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout_sec,
            cwd=str(engine_bin.resolve().parent.parent.parent),
        )
    except subprocess.TimeoutExpired as exc:
        wall = time.perf_counter() - t0
        return {
            "ok": False,
            "termination": "wall_timeout",
            "elapsed_sec": wall,
            "nodes": None,
            "search_depth": None,
            "stderr_tail": (exc.stderr or "")[-500:] if exc.stderr else "",
        }
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        return {
            "ok": False,
            "termination": "engine_error",
            "elapsed_sec": wall,
            "nodes": None,
            "search_depth": None,
            "stderr_tail": (proc.stderr or "")[-1000:],
            "stdout_tail": (proc.stdout or "")[-500:],
        }

    info: dict | None = None
    for line in (proc.stderr or "").splitlines():
        m = INFO_JSON_RE.match(line.strip())
        if m:
            info = json.loads(m.group(1))
            break

    bestmove = None
    for line in reversed((proc.stdout or "").splitlines()):
        line = line.strip()
        if line.startswith("bestmove "):
            bestmove = line.split()[1]
            break

    if not info:
        return {
            "ok": False,
            "termination": "missing_info_json",
            "elapsed_sec": wall,
            "nodes": None,
            "search_depth": None,
            "stderr_tail": (proc.stderr or "")[-1000:],
            "stdout_tail": (proc.stdout or "")[-500:],
        }

    nodes = int(info.get("nodes", 0))
    depth = int(info.get("searchDepth", 0))
    elapsed_ms = int(info.get("elapsedMs", int(wall * 1000)))
    nps = nodes / max(elapsed_ms / 1000.0, 1e-6)
    term = "time_budget"
    if elapsed_ms > time_sec * 1000 * 1.5:
        term = "slow_completion"
    return {
        "ok": True,
        "termination": term,
        "elapsed_sec": wall,
        "engine_elapsed_ms": elapsed_ms,
        "nodes": nodes,
        "search_depth": depth,
        "nps": nps,
        "root_score": info.get("rootScore"),
        "bestmove": bestmove,
    }


def percentile(vals: list[float], p: float) -> float:
    if not vals:
        return float("nan")
    xs = sorted(vals)
    k = (len(xs) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def summarize(results: list[dict]) -> dict:
    ok = [r for r in results if r.get("ok")]
    nodes = [float(r["nodes"]) for r in ok if r.get("nodes") is not None]
    nps = [float(r["nps"]) for r in ok if r.get("nps") is not None]
    failures = [r for r in results if not r.get("ok")]
    outliers = [
        r for r in ok
        if r.get("nodes") is not None and (
            r["nodes"] < percentile(nodes, 10) * 0.25 or r["nodes"] > percentile(nodes, 90) * 4
        )
    ]
    return {
        "positions_total": len(results),
        "positions_ok": len(ok),
        "positions_failed": len(failures),
        "nodes_median": statistics.median(nodes) if nodes else None,
        "nodes_mean": statistics.mean(nodes) if nodes else None,
        "nodes_p10": percentile(nodes, 10) if nodes else None,
        "nodes_p25": percentile(nodes, 25) if nodes else None,
        "nodes_p75": percentile(nodes, 75) if nodes else None,
        "nodes_p90": percentile(nodes, 90) if nodes else None,
        "nps_median": statistics.median(nps) if nps else None,
        "failures": failures[:20],
        "outliers": [
            {
                "game_id": r.get("game_id"),
                "ply": r.get("ply"),
                "nodes": r.get("nodes"),
                "termination": r.get("termination"),
            }
            for r in outliers[:20]
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", type=Path, default=ENGINE_BIN)
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--games-db", type=Path, default=GAMES_DB)
    ap.add_argument("--positions", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--time-sec", type=float, default=2.0)
    ap.add_argument("--timeout-sec", type=float, default=30.0)
    ap.add_argument("--positions-file", type=Path, help="reuse frozen position set (JSON)")
    ap.add_argument("--write-positions", type=Path, help="write sampled positions JSON")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--label", default="laptop")
    args = ap.parse_args()

    if not args.engine.is_file():
        print(f"engine missing: {args.engine}", file=sys.stderr)
        return 2
    if not args.weights.is_file():
        print(f"weights missing: {args.weights}", file=sys.stderr)
        return 2

    if args.positions_file and args.positions_file.is_file():
        positions = json.loads(args.positions_file.read_text(encoding="utf-8"))
    else:
        if not args.games_db.is_file():
            print(f"games.db missing: {args.games_db}", file=sys.stderr)
            return 2
        positions = sample_positions(args.games_db, count=args.positions, seed=args.seed)
        if args.write_positions:
            args.write_positions.parent.mkdir(parents=True, exist_ok=True)
            args.write_positions.write_text(json.dumps(positions, indent=2), encoding="utf-8")

    engine_hash = sha256_file(args.engine)
    weight_hash = sha256_file(args.weights)
    hw = hardware_id()

    results: list[dict] = []
    t_start = time.perf_counter()
    for i, pos in enumerate(positions):
        r = run_one(
            engine_bin=args.engine,
            weights=args.weights,
            moves=pos["moves"],
            time_sec=args.time_sec,
            timeout_sec=args.timeout_sec,
        )
        r.update(
            {
                "index": i,
                "game_id": pos.get("game_id"),
                "ply": pos.get("ply"),
                "move_count": pos.get("move_count"),
                "engine_hash": engine_hash,
                "weight_hash": weight_hash,
                "hardware_id": hw,
            }
        )
        results.append(r)
        if (i + 1) % 10 == 0:
            print(f"progress {i+1}/{len(positions)}", flush=True)

    total_sec = time.perf_counter() - t_start
    report = {
        "label": args.label,
        "time_sec_per_move": args.time_sec,
        "engine": str(args.engine),
        "engine_hash": engine_hash,
        "weights": str(args.weights),
        "weight_hash": weight_hash,
        "hardware_id": hw,
        "engine_name": ENGINE_NAME,
        "total_benchmark_sec": total_sec,
        "summary": summarize(results),
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    return 0 if report["summary"]["positions_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
