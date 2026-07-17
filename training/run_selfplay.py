#!/usr/bin/env python3
"""
Launch N parallel self-play instances.

Each instance is a separate self_play_loop.py process, writing its own log.
Ctrl-C here kills all of them cleanly.

Usage:
  python training/run_selfplay.py --threads 4 --time 2.0 --verify-ratio 3
"""
from __future__ import annotations

import argparse
import signal
import subprocess
import sys
from pathlib import Path

_REPO     = Path(__file__).resolve().parent.parent
_TRAINING = _REPO / "training"
_DATA     = _TRAINING / "data"
_SCRIPT   = _TRAINING / "self_play_loop.py"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threads",      type=int,   default=4,
                    help="Number of parallel self-play instances (default 4)")
    ap.add_argument("--time",         type=float, default=2.0,
                    help="Seconds per engine move (default 2.0)")
    ap.add_argument("--verify-ratio", type=int,   default=3,
                    help="Training games per verify game (default 3 → 1 verify per 4 games)")
    ap.add_argument("--baseline-weights", type=str, default=None,
                    help="Path to frozen baseline weights for B side (default: baked-in engine weights)")
    args = ap.parse_args()

    _DATA.mkdir(parents=True, exist_ok=True)

    procs: list[subprocess.Popen] = []
    logs:  list[Path]             = []

    cmd_base = [
        sys.executable, str(_SCRIPT),
        "--time",         str(args.time),
        "--verify-ratio", str(args.verify_ratio),
    ]
    if args.baseline_weights:
        cmd_base += ["--baseline-weights", args.baseline_weights]

    print(f"Starting {args.threads} self-play instance(s)")
    print(f"  time={args.time}s/move  verify_ratio=1:{args.verify_ratio}")
    print()

    for i in range(1, args.threads + 1):
        log_path = _DATA / f"self_play_{i}.log"
        logs.append(log_path)
        log_fh = open(log_path, "a")
        p = subprocess.Popen(cmd_base, stdout=log_fh, stderr=log_fh, cwd=str(_REPO))
        procs.append(p)
        print(f"  Instance {i}  PID={p.pid}  log=training/data/self_play_{i}.log")

    print()
    print("All instances running. Press Ctrl-C to stop all.")
    print()

    def _stop_all(sig, frame):
        print("\nStopping all instances ...")
        for p in procs:
            try:
                p.terminate()
            except Exception:
                pass
        for p in procs:
            try:
                p.wait(timeout=10)
            except Exception:
                pass
        print("Done.")
        sys.exit(0)

    signal.signal(signal.SIGINT,  _stop_all)
    signal.signal(signal.SIGTERM, _stop_all)

    # Wait for all (they run forever unless killed)
    for p in procs:
        p.wait()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
