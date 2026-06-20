#!/usr/bin/env python3
"""Collect and train the leaf search-pressure sidecar.

This is deliberately separate from live NNUE WDL training. It builds a compact
label file from current engine searches, then trains the tiny sidecar head that
predicts whether a reached child node is saturated or deserves more budget.

Example:
    python training/run_search_pressure_experiment.py --labels 2000 --chunk 200 --time 2.0 --cpu
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "training" / "data" / "search_pressure.jsonl"


def count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            json.loads(line)
        except json.JSONDecodeError:
            continue
        n += 1
    return n


def run(cmd: list[str]) -> None:
    print(">> " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(DATA), help="Compact search-pressure JSONL output")
    ap.add_argument("--labels", type=int, default=1000, help="Target total label rows")
    ap.add_argument("--chunk", type=int, default=100, help="Labels to collect per subprocess")
    ap.add_argument("--engine", default="titanium-v15")
    ap.add_argument("--shallow-depth", type=int, default=2)
    ap.add_argument("--deep-depth", type=int, default=5)
    ap.add_argument("--time", type=float, default=2.0)
    ap.add_argument("--min-ply", type=int, default=6)
    ap.add_argument("--max-ply", type=int, default=90)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--collect-only", action="store_true")
    ap.add_argument("--train-only", action="store_true")
    args = ap.parse_args()

    out = Path(args.data)
    out.parent.mkdir(parents=True, exist_ok=True)

    if not args.train_only:
        while count_rows(out) < args.labels:
            have = count_rows(out)
            want = min(args.chunk, args.labels - have)
            run([
                sys.executable,
                str(ROOT / "training" / "collect_search_importance.py"),
                "--out", str(out),
                "--engine", args.engine,
                "--limit", str(want),
                "--shallow-depth", str(args.shallow_depth),
                "--deep-depth", str(args.deep_depth),
                "--time", str(args.time),
                "--min-ply", str(args.min_ply),
                "--max-ply", str(args.max_ply),
                "--seed", str(args.seed + have),
            ])
            after = count_rows(out)
            if after <= have:
                print("collector made no progress; stopping before a spin loop", file=sys.stderr)
                return 1
        print(f"labels ready: {count_rows(out)} rows -> {out}", flush=True)

    if args.collect_only:
        return 0

    run([
        sys.executable,
        str(ROOT / "training" / "train_search_importance.py"),
        "--data", str(out),
        "--epochs", str(args.epochs),
        "--batch", str(args.batch),
        "--lr", str(args.lr),
        "--seed", str(args.seed),
        *(["--cpu"] if args.cpu else []),
    ])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
