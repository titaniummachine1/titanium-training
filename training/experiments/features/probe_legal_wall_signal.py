#!/usr/bin/env python3
"""30-minute probe: is legal_wall_count orthogonal to BFS planes + corridor_width?

Reads positions from a .games file, runs eval-batch, reports correlations.
Requires titanium built with legal_wall_count in eval --json.

Usage:
    python training/probe_legal_wall_signal.py [path/to/file.games] [--max-positions N]
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "engine" / "target" / "release" / "titanium.exe"
DEFAULT_GAMES = ROOT / "training" / "data" / "self_match_games.games"


def parse_games(path: Path, max_positions: int, seed: int) -> list[str]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    seqs: list[str] = []
    for line in lines:
        line = line.strip()
        if not line.startswith("GAME "):
            continue
        moves = line[5:].strip()
        if moves:
            seqs.append(moves)
    if not seqs:
        return [""]
    rng = random.Random(seed)
    rng.shuffle(seqs)
    out: list[str] = []
    for moves in seqs:
        parts = moves.split()
        for ply in range(4, min(len(parts), 80), 3):
            out.append(" ".join(parts[:ply]))
            if len(out) >= max_positions:
                return out
    return out[:max_positions]


def eval_batch(seqs: list[str]) -> list[dict]:
    payload = "\n".join(seqs) + "\n"
    r = subprocess.run(
        [str(BIN), "eval-batch"],
        input=payload,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        timeout=120,
    )
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
        raise SystemExit(f"eval-batch failed: {r.returncode}")
    recs = []
    for line in r.stdout.splitlines():
        line = line.strip()
        if line:
            recs.append(json.loads(line))
    return recs


def field_sum(rec: dict, key: str) -> int:
    arr = rec.get(key, [])
    return sum(arr)


def field_nonzero_cells(rec: dict, key: str) -> int:
    arr = rec.get(key, [])
    return sum(1 for v in arr if v not in (0, 255))


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 3:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("games", nargs="?", default=str(DEFAULT_GAMES))
    ap.add_argument("--max-positions", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not BIN.exists():
        raise SystemExit(f"missing {BIN} — cargo build --release -p titanium")

    games = Path(args.games)
    if not games.is_file():
        raise SystemExit(f"no games file: {games}")

    seqs = parse_games(games, args.max_positions, args.seed)
    print(f"Probing {len(seqs)} positions from {games.name} …")
    recs = eval_batch(seqs)

    lw = [float(r["legal_wall_count"]) for r in recs]
    cw0 = [float(r.get("corridor_width0", 0)) for r in recs]
    cw1 = [float(r.get("corridor_width1", 0)) for r in recs]
    walls = [float(sum(c == "1" for c in r.get("hw", "")) + sum(c == "1" for c in r.get("vw", ""))) for r in recs]
    pc = [float(field_sum(r, "path_cross_p0_field") + field_sum(r, "path_cross_p1_field")) for r in recs]
    ch = [float(field_sum(r, "choke_p0_field") + field_sum(r, "choke_p1_field")) for r in recs]
    cd = [float(field_nonzero_cells(r, "corridor_delta_p0_field") + field_nonzero_cells(r, "corridor_delta_p1_field")) for r in recs]
    wl_sum = [float(r["wl0"] + r["wl1"]) for r in recs]

    pairs = [
        ("corridor_width0", cw0),
        ("corridor_width1", cw1),
        ("placed_walls", walls),
        ("path_cross_sum", pc),
        ("choke_sum", ch),
        ("corridor_delta_cells", cd),
        ("wl0+wl1", wl_sum),
    ]

    print("\n=== Pearson r vs legal_wall_count ===")
    for name, ys in pairs:
        r = pearson(lw, ys)
        print(f"  {name:22s}  r = {r:+.3f}")

    print("\n=== legal_wall_count distribution ===")
    print(f"  min={min(lw):.0f}  max={max(lw):.0f}  mean={sum(lw)/len(lw):.1f}")

    # Partial: is lw explained by (walls placed + corridor width)?
    print("\n=== Interpretation ===")
    r_w = pearson(lw, walls)
    r_cw = pearson(lw, [(a + b) / 2 for a, b in zip(cw0, cw1)])
    if abs(r_w) > 0.85 and abs(r_cw) < 0.5:
        print("  legal_wall_count tracks open slots (~inverse of placed walls) — partially redundant with w1c density.")
    elif abs(r_cw) > 0.7:
        print("  legal_wall_count correlates with corridor_width — ws[14] replacement may overlap semantics.")
    elif max(lw) - min(lw) < 8:
        print("  legal_wall_count barely varies in sample — probe more positions or wall-heavy lines.")
    else:
        print("  legal_wall_count varies independently enough to justify scalar ablation on ws[14].")

    missing = sum(1 for r in recs if "legal_wall_count" not in r)
    if missing:
        print(f"\n  WARNING: {missing} records missing legal_wall_count — rebuild titanium.")


if __name__ == "__main__":
    main()
