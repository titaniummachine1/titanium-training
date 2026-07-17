"""Extend engine net blobs from WSKIP_LEN=18 to WSKIP_LEN=20 (cat_best ws[18]/ws[19])."""
from __future__ import annotations

import struct
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OLD_WS = 18
NEW_WS = 20
FILES = [
    REPO / "engine" / "src" / "titanium" / "net_weights.bin",
    REPO / "engine" / "src" / "titanium" / "net_weights_frozen.bin",
    REPO / "engine" / "src" / "titanium" / "net_weights_medium.bin",
]


def extend(path: Path) -> None:
    if not path.is_file():
        print(f"  SKIP missing {path.name}")
        return
    data = path.read_bytes()
    head = OLD_WS * 8
    if len(data) < head:
        print(f"  ERROR {path.name}: too small", file=sys.stderr)
        sys.exit(1)
    if len(data) == NEW_WS * 8 + (len(data) - head - (NEW_WS - OLD_WS) * 8):
        pass
    if len(data) == (len(data) // 8) * 8 + (NEW_WS - OLD_WS) * 8 and len(data) > head + (NEW_WS - OLD_WS) * 8:
        # already extended if size matches ws20 layout heuristic
        expected_ws20 = len(data)
        if expected_ws20 == 340296:
            print(f"  {path.name}: already ws20 ({len(data)} bytes)")
            return
    if len(data) == 340296:
        print(f"  {path.name}: already ws20 ({len(data)} bytes)")
        return
    if len(data) != 340280:
        print(f"  WARN {path.name}: unexpected size {len(data)}", file=sys.stderr)
    extra = struct.pack(f"<{NEW_WS - OLD_WS}d", *([0.0] * (NEW_WS - OLD_WS)))
    out = data[:head] + extra + data[head:]
    path.write_bytes(out)
    print(f"  {path.name}: {len(data)} -> {len(out)} bytes")


def main() -> None:
    print(f"Extending ws {OLD_WS} -> {NEW_WS}:")
    for p in FILES:
        extend(p)
    print("Done — rebuild titanium with RUSTFLAGS=-C target-cpu=native")


if __name__ == "__main__":
    main()
