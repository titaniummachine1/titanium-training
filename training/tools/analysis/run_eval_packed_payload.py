#!/usr/bin/env python3
"""Feed eval-packed-batch payload from disk (for flamegraph / timing)."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exe", type=Path, required=True)
    ap.add_argument("--payload", type=Path, required=True)
    args = ap.parse_args()
    payload = args.payload.read_bytes()
    proc = subprocess.run(
        [str(args.exe), "eval-packed-batch"],
        input=payload,
        capture_output=True,
    )
    sys.stdout.buffer.write(proc.stdout)
    sys.stderr.buffer.write(proc.stderr)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
