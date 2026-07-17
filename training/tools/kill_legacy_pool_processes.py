#!/usr/bin/env python3
"""Stop legacy coupled continuous_pool zombies before starting split services."""
from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

_TRAINING = Path(__file__).resolve().parents[1]
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

from pool_lock import find_legacy_pool_processes, _pid_alive


def _stop_pid(pid: int) -> bool:
    if not _pid_alive(pid):
        return False
    if sys.platform == "win32":
        import subprocess

        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            check=False,
            capture_output=True,
        )
    else:
        import os

        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            return False
    for _ in range(20):
        if not _pid_alive(pid):
            return True
        time.sleep(0.25)
    return not _pid_alive(pid)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keep-pid", type=int, default=0, help="Do not stop this PID")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    victims = [
        (pid, cmd)
        for pid, cmd in find_legacy_pool_processes()
        if pid != args.keep_pid
    ]
    if not victims:
        print("No legacy continuous_pool processes found")
        return 0
    for pid, cmd in victims:
        short = cmd[:120] + ("..." if len(cmd) > 120 else "")
        if args.dry_run:
            print(f"would stop pid={pid} cmd={short}")
            continue
        ok = _stop_pid(pid)
        print(f"stopped pid={pid} ok={ok} cmd={short}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
