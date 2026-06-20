#!/usr/bin/env python3
"""Eye ASCII grids for NNUE distance fields — sanity-check training geometry.

Uses the same parallel flood as search eval (`acev13/fields_viz.rs`).

Examples:
    python training/visualize_fields.py
    python training/visualize_fields.py e2 e8 e3 e7 d3h f5v --check
    python training/visualize_fields.py --positions   # built-in mid-game set
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "engine" / "target" / "release" / "titanium.exe"

POSITIONS = [
    [],
    ["e2", "e8", "e3", "e7", "d3h", "f5v"],
    ["e2", "e8", "e3", "e7", "e4", "e6", "a3h", "d4v"],
    ["e2", "e8", "d2", "f8", "c4h", "g5h"],
]


def run(moves: list[str], check: bool) -> int:
    cmd = [str(BIN), "fields", *moves]
    if check:
        cmd.append("--check")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        sys.stderr.write(proc.stdout)
        return proc.returncode
    if proc.stdout:
        print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
    if proc.stderr:
        sys.stderr.write(proc.stderr)
    return 0


def main() -> None:
    args = sys.argv[1:]
    check = "--check" in args
    args = [a for a in args if a != "--check"]

    if not BIN.is_file():
        print(f"Build engine first: {BIN}", file=sys.stderr)
        sys.exit(1)

    if args == ["--positions"]:
        rc = 0
        for moves in POSITIONS:
            label = "startpos" if not moves else " ".join(moves)
            print("=" * 72)
            print(label)
            print("=" * 72)
            rc |= run(moves, check=True)
        sys.exit(rc)

    rc = run(args, check=check)
    sys.exit(rc)


if __name__ == "__main__":
    main()
