#!/usr/bin/env python3
"""Measure wall_ignore loss-cert hit rate with TITANIUM_WALL_IGNORE_LOSS_CERT=1.

Uses search_bench (not titanium.exe):
  $env:RUSTFLAGS='-C target-cpu=native'
  cargo build --release -p titanium --bin search_bench

Production default remains OFF; this script only sets the env for measurement.

Usage:
  py -3 measure_wall_ignore_stats.py <search_bench.exe> [ms=200] [runs=20]
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

KEYS = (
    "wall_ignore_calls",
    "wall_ignore_decisive",
    "wall_ignore_unknown",
    "wall_ignore_cut_fail_high",
    "wall_ignore_cut_fail_low",
)


def run_think(exe: Path, ms: int, moves: str | None, env: dict[str, str]) -> dict:
    args = [str(exe), "think", "--ms", str(ms), "--full", "--threads", "1"]
    if moves:
        args += ["--moves", moves]
    proc = subprocess.run(args, capture_output=True, text=True, env=env, check=False)
    stats: dict = {}
    for stream in (proc.stdout, proc.stderr):
        for line in stream.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if any(k in obj for k in KEYS) or "broke_calls" in obj:
                stats = obj
    return stats


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    exe = Path(sys.argv[1])
    ms = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    runs = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    if not exe.is_file():
        print(f"missing exe: {exe}", file=sys.stderr)
        return 1

    env = os.environ.copy()
    env.pop("TITANIUM_ALLOW_SUBOPTIMAL", None)
    env["TITANIUM_WALL_IGNORE_LOSS_CERT"] = "1"
    env["TITANIUM_BOOK_MODE"] = "off"

    positions = [None] + [
        "e2 e8",
        "e2 e8 c3h",
        "e2 e8 c3h f6h",
        "e2 e8 c3v f6v",
        "e2 e8 e3 e7 c3h f6h",
        "e2 e8 e3 e7 c3h f6h d3v g6v",
    ]

    totals = {k: 0 for k in KEYS}
    totals["thinks"] = 0
    rows = []

    for i in range(runs):
        moves = positions[i % len(positions)]
        stats = run_think(exe, ms, moves, env)
        totals["thinks"] += 1
        row = {"i": i, "moves": moves or "", "stats": {}}
        for k in KEYS:
            v = int(stats.get(k, 0) or 0)
            totals[k] += v
            row["stats"][k] = v
        rows.append(row)
        print(
            f"think {i}: moves={moves or '-'} "
            f"calls={row['stats']['wall_ignore_calls']} "
            f"dec={row['stats']['wall_ignore_decisive']} "
            f"unk={row['stats']['wall_ignore_unknown']} "
            f"cutH={row['stats']['wall_ignore_cut_fail_high']} "
            f"cutL={row['stats']['wall_ignore_cut_fail_low']}",
            flush=True,
        )

    if totals["wall_ignore_calls"]:
        totals["decisive_rate_pct"] = round(
            100.0 * totals["wall_ignore_decisive"] / totals["wall_ignore_calls"], 2
        )
        totals["cut_rate_pct"] = round(
            100.0
            * (totals["wall_ignore_cut_fail_high"] + totals["wall_ignore_cut_fail_low"])
            / totals["wall_ignore_calls"],
            2,
        )
    else:
        totals["decisive_rate_pct"] = None
        totals["cut_rate_pct"] = None
        print(
            "WARNING: zero wall_ignore_calls — use search_bench built from current sources.",
            flush=True,
        )

    out = {
        "summary": totals,
        "rows": rows,
        "note": "env TITANIUM_WALL_IGNORE_LOSS_CERT=1 for measure only; production default OFF",
        "exe": str(exe),
    }
    print(json.dumps(totals, indent=2))
    out_path = Path(__file__).resolve().parent / "wall_ignore_measure_last.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
