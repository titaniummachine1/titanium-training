#!/usr/bin/env python3
"""Print a clear A/B scoreboard from local + oracle status.json files."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def wilson(succ: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    p = succ / n
    z2 = z * z
    den = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / den
    margin = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n) / den
    return center - margin, center + margin


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", type=Path, required=True)
    ap.add_argument("--oracle", type=Path, required=True)
    ap.add_argument("--a-name", default="A")
    ap.add_argument("--b-name", default="B")
    ap.add_argument("--title", default="A/B MATCH")
    args = ap.parse_args()

    local = load(args.local)
    oracle = load(args.oracle)

    la, lb, ld = int(local["a_wins"]), int(local["b_wins"]), int(local["draws"])
    oa, ob, od = int(oracle["a_wins"]), int(oracle["b_wins"]), int(oracle["draws"])
    ln, on = la + lb + ld, oa + ob + od
    a, b, d = la + oa, lb + ob, ld + od
    n = a + b + d
    succ = a + 0.5 * d
    score = succ / n if n else 0.0
    wlb, wub = wilson(succ, n)
    if 0.0 < score < 1.0:
        elo = -400.0 * math.log10(1.0 / score - 1.0)
        elo_s = f"{elo:+.1f} Elo for A"
    elif score <= 0.0:
        elo_s = "A much weaker (near 0%)"
    else:
        elo_s = "A much stronger (near 100%)"

    margin = a - b
    if margin > 0:
        leader = f"A leading  (+{margin})"
    elif margin < 0:
        leader = f"B leading  (+{-margin})"
    else:
        leader = "TIED"

    ls = 100.0 * (la + 0.5 * ld) / ln if ln else 0.0
    os_ = 100.0 * (oa + 0.5 * od) / on if on else 0.0
    lstat = "RUN " if local.get("running") else "DONE"
    ostat = "RUN " if oracle.get("running") else "DONE"

    print()
    print("=" * 66)
    print(f"  {args.title}")
    print(f"  A = {args.a_name}")
    print(f"  B = {args.b_name}")
    print("=" * 66)
    print()
    print(
        f"  LOCAL    A {la:>3}   B {lb:<3}   D {ld}   |"
        f"  A {ls:5.1f}%   {ln:>2}/{local.get('shard_games', '?')}  {lstat}"
    )
    print(
        f"  ORACLE   A {oa:>3}   B {ob:<3}   D {od}   |"
        f"  A {os_:5.1f}%   {on:>2}/{oracle.get('shard_games', '?')}  {ostat}"
    )
    print("  " + "-" * 62)
    print(
        f"  TOTAL   A {a:>3}   B {b:<3}   D {d}   |"
        f"  A {100.0 * score:5.1f}%   {n:>2}/100"
    )
    print()
    print(f"  LEADER:       {leader}")
    print(f"  WIN MARGIN:   {abs(margin)} game(s)   (A minus B = {margin:+d})")
    print(f"  A WINRATE:    {100.0 * score:.2f}%   (score={score:.4f})")
    print(f"  APPROX ELO:   {elo_s}")
    print(f"  WILSON 95%:   [{100.0 * wlb:.1f}% , {100.0 * wub:.1f}%]")
    print(f"  GAMES LEFT:   {100 - n}")
    print("=" * 66)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
