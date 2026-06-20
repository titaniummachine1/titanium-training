#!/usr/bin/env python3
"""Eyeball engine JSON inputs on sample positions — sanity, not parity."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

from titanium_training.models.field_planes import (
    CHOKE_P0,
    CHOKE_P1,
    CONTESTED,
    CORRIDOR_DELTA_P0,
    CORRIDOR_DELTA_P1,
    GOAL_INV_P0,
    GOAL_INV_P1,
    PATH_CROSS_P0,
    PATH_CROSS_P1,
    PAWN_FWD_P0,
    PAWN_FWD_P1,
    rec_field,
)
from titanium_training.models.halfpw import Net, forward, legal_wall_norm, opponent_corridor_width

BIN = ROOT / "engine" / "target" / "release" / "titanium.exe"
WEIGHTS = ROOT / "engine" / "src" / "acev13" / "net_weights.bin"

POSITIONS = [
    ("startpos", []),
    ("mid_walls", ["e2", "e8", "e3", "e7", "d3h", "f5v"]),
    ("wall_heavy", ["e2", "e8", "e3", "e7", "e4", "e6", "c6h", "f3v", "b5h"]),
    ("asymmetric", ["e2", "e8", "d2", "f8", "c4h", "g5h"]),
]

PLANE_KEYS = [
    GOAL_INV_P0,
    GOAL_INV_P1,
    PAWN_FWD_P0,
    PAWN_FWD_P1,
    CORRIDOR_DELTA_P0,
    CORRIDOR_DELTA_P1,
    PATH_CROSS_P0,
    PATH_CROSS_P1,
    CHOKE_P0,
    CHOKE_P1,
    CONTESTED,
]

REQUIRED = (
    "turn", "d0", "d1", "wl0", "wl1", "pawn0", "pawn1", "hw", "vw",
    "legal_wall_count", "corridor_width0", "corridor_width1", "eval",
)


def plane_stats(rec: dict, key: str) -> tuple[int, int, int, int]:
    vals = rec_field(rec, key)
    if not vals:
        return 0, 0, 0, 0
    nz = sum(1 for v in vals if v != 0)
    return len(vals), nz, min(vals), max(vals)


def main() -> None:
    net = Net.load(WEIGHTS)
    issues = 0

    for tag, moves in POSITIONS:
        rec = json.loads(
            subprocess.run(
                [str(BIN), "eval", *moves, "--json"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        )
        missing = [k for k in REQUIRED if k not in rec]
        me = rec["turn"]
        d_me = rec["d0" if me == 0 else "d1"]
        d_opp = rec["d1" if me == 0 else "d0"]
        lw = rec["legal_wall_count"]
        lwn = legal_wall_norm(rec)
        w15 = opponent_corridor_width(rec, me, int(d_me), int(d_opp))
        py = forward(net, rec)
        eng = rec["eval"]
        match = py == eng

        print(f"\n=== {tag}  plies={len(moves)}  turn=P{me}  eval={eng}  parity={'OK' if match else f'BAD {py}!={eng}'} ===")
        if missing:
            print(f"  MISSING FIELDS: {missing}")
            issues += 1
        wl0, wl1 = rec["wl0"], rec["wl1"]
        print(
            f"  scalars: d_me={d_me} d_opp={d_opp} wl=({wl0},{wl1}) "
            f"legal_wall={lw}/128={lwn:.4f} ws15_opp_corridor={w15} "
            f"corridor_w=({rec['corridor_width0']},{rec['corridor_width1']})"
        )
        if not (0 <= lw <= 128):
            print(f"  *** legal_wall_count out of range: {lw}")
            issues += 1
        if len(rec["hw"]) != 64 or len(rec["vw"]) != 64:
            print("  *** hw/vw length wrong")
            issues += 1

        for k in PLANE_KEYS:
            n, nz, mn, mx = plane_stats(rec, k)
            if n == 0:
                print(f"  {k:28} MISSING")
                issues += 1
                continue
            note = ""
            if nz == 0 and k != CONTESTED:
                note = "  *** all zero (unexpected for BFS plane)"
                issues += 1
            elif k == CONTESTED and nz == 0:
                note = "  (ok: no contested cells)"
            print(f"  {k:28} nz={nz:2}/{n}  range=[{mn},{mx}]{note}")

        # ws[13] fragile-lead term sanity
        pd = d_opp - d_me
        fragile = pd * wl1 / 10.0 if me == 0 else pd * wl0 / 10.0
        print(f"  ws[13] fragile-lead input ~ pd*w_opp/10 = {fragile:.2f}")

    print(f"\n{'PASS' if issues == 0 else f'ISSUES: {issues}'} — eyeballed {len(POSITIONS)} positions")


if __name__ == "__main__":
    main()
