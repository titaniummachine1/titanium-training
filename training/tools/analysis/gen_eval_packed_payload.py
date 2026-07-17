#!/usr/bin/env python3
"""Build stdin payload for `titanium eval-packed-batch` flamegraph runs."""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

_TRAINING = Path(__file__).resolve().parents[2]
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

from titanium_training.data.teacher_value import iter_value_only_rows
from titanium_training.paths import ACTIVE_TEACHER_DATASET

PACKED_RECORD = struct.Struct("<I24s")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=8192, help="Number of positions")
    ap.add_argument("-o", "--output", type=Path, required=True)
    args = ap.parse_args()

    buf = bytearray()
    n = 0
    for row in iter_value_only_rows(ACTIVE_TEACHER_DATASET):
        ps = row.get("packed_state")
        if not ps or len(ps) != 24:
            continue
        if isinstance(ps, str):
            ps = bytes.fromhex(ps)
        buf.extend(PACKED_RECORD.pack(n, ps))
        n += 1
        if n >= args.limit:
            break

    if n == 0:
        print("ERROR: no packed positions found", file=sys.stderr)
        return 1
    args.output.write_bytes(buf)
    print(f"Wrote {n} records ({len(buf)} bytes) -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
