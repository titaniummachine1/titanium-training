"""Infinite strength benchmark: titanium-v15 vs ace-v13-ti-pure baseline.

Runs batches forever — each game is saved and ingested immediately so the
training DB grows without manual steps.  Progress is written to:
  training/data/STATUS.txt           (human-readable — check this first)
  training/data/manifest.json        (machine-readable)
  training/data/v15_vs_ti_pure.games (raw GAME/RESULT dump)

Usage:
    python training/run_infinite_benchmark.py
    python training/run_infinite_benchmark.py --batch-size 8 --time 5
"""

import argparse
import math
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SELF_MATCH = ROOT / "site" / "self_match.js"
BIN = ROOT / "engine" / "target" / "release" / "titanium.exe"
GAMES_FILE = ROOT / "training" / "data" / "v15_vs_ti_pure.games"

from tools.maintenance.manifest import (  # noqa: E402
    update_strength_tracker,
    save_manifest,
    load_manifest,
    PATHS,
    CURRENT_ENGINE,
    BASELINE_ENGINE,
)


def parse_match_summary(stderr: str) -> dict | None:
    for line in stderr.splitlines():
        m = re.search(
            r"MATCH_SUMMARY A=(\d+) B=(\d+) DRAWS=(\d+) SCORE=([\d.]+)/(\d+) ELO=([+-]?(?:Infinity|\d+(?:\.\d+)?))",
            line,
        )
        if m:
            a, b, d, score, n, elo = m.groups()
            return {
                "a_wins": int(a),
                "b_wins": int(b),
                "draws": int(d),
                "n": int(n),
                "elo": float(elo),
            }
    return None


def run_batch(engine_a, engine_b, batch_size, time_s, concurrency, ponder_time):
    cmd = [
        "node", str(SELF_MATCH),
        "--engine-a", engine_a,
        "--engine-b", engine_b,
        "--games", str(batch_size),
        "--time", str(time_s),
        "--ponder-time", str(ponder_time),
        "--concurrency", str(concurrency),
        "--save-games", str(GAMES_FILE),
        "--source-tag", "v15-vs-ti-pure",
    ]
    if not BIN.exists():
        print(f"ERROR: engine binary not found: {BIN}")
        print("Run in engine/: $env:RUSTFLAGS='-C target-cpu=native'; cargo build --release -p titanium")
        sys.exit(1)

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"self_match.js exited {result.returncode}")
    return parse_match_summary(result.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine-a", default=CURRENT_ENGINE,
                    help=f"Our engine (default {CURRENT_ENGINE})")
    ap.add_argument("--engine-b", default=BASELINE_ENGINE,
                    help=f"Baseline for Elo (default {BASELINE_ENGINE})")
    ap.add_argument("--batch-size", type=int, default=16,
                    help="Games per batch (default 16 — progress updates often)")
    ap.add_argument("--time", type=float, default=5.0)
    ap.add_argument("--ponder-time", type=float, default=None,
                    help="Ponder seconds (default = --time)")
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()
    ponder = args.ponder_time if args.ponder_time is not None else args.time

    GAMES_FILE.parent.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    totals = manifest.get("strength_tracker", {})
    cum_a = totals.get("a_wins", 0)
    cum_b = totals.get("b_wins", 0)
    batch_num = totals.get("batches_completed", 0)

    status_path = ROOT / "training" / "data" / "STATUS.txt"
    print("=" * 60)
    print(f"INFINITE BENCHMARK: {args.engine_a} vs {args.engine_b}")
    print(f"  ({CURRENT_ENGINE} = current Titanium; {BASELINE_ENGINE} = JS v13 baseline)")
    print(f"  batch={args.batch_size}  time={args.time}s  ponder={ponder}s  concurrency={args.concurrency}")
    print(f"  games file: {GAMES_FILE}")
    print(f"  training DB: {PATHS['training_db']}")
    print(f"  progress:    {status_path}")
    print(f"  starting score: {args.engine_a} {cum_a} - {cum_b} {args.engine_b}  ({cum_a + cum_b} games)")
    print("  Ctrl+C to stop")
    print("=" * 60)

    while True:
        batch_num += 1
        t0 = time.time()
        print(f"\n--- batch {batch_num} ({args.batch_size} games) ---")
        try:
            summary = run_batch(
                args.engine_a, args.engine_b,
                args.batch_size, args.time, args.concurrency, ponder,
            )
        except RuntimeError as e:
            print(f"BATCH FAILED: {e}  — retrying in 30s")
            time.sleep(30)
            continue

        if summary:
            # self_match.js already persists cumulative totals to manifest each game.
            cum_a = summary["a_wins"]
            cum_b = summary["b_wins"]
        n = cum_a + cum_b or 1
        p = cum_a / n
        elo = -400 * math.log10((1 - p) / p) if 0 < p < 1 else (9999 if p >= 1 else -9999)
        elapsed = time.time() - t0

        update_strength_tracker(cum_a, cum_b, batch=batch_num)
        print(
            f"CUMULATIVE: {args.engine_a} {cum_a} - {cum_b} {args.engine_b}  "
            f"/ {n} games  ~{elo:+.0f} Elo vs baseline  ({elapsed/60:.1f} min this batch)"
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
        save_manifest(load_manifest())
