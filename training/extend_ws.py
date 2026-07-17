"""Extend net_weights.bin and net_weights_frozen.bin from WSKIP_LEN=16 to WSKIP_LEN=18.

Inserts two zero f64 values (ws[16], ws[17]) immediately after the existing 16 ws values.
Old layout: Wskip[16] B1[32] W2[32] W1C[36864] PO[2592] PX[2592] route_planes...
New layout: Wskip[18] B1[32] W2[32] W1C[36864] PO[2592] PX[2592] route_planes...
"""
import struct
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
WEIGHT_FILES = [
    REPO / "engine" / "src" / "titanium" / "net_weights.bin",
    REPO / "engine" / "src" / "titanium" / "net_weights_frozen.bin",
]

OLD_WS = 16
NEW_WS = 18
WS_BYTES_OLD = OLD_WS * 8  # 128 bytes
EXTRA_BYTES = (NEW_WS - OLD_WS) * 8  # 16 bytes of zeros


def extend_file(path: pathlib.Path) -> None:
    data = path.read_bytes()
    old_size = len(data)
    if old_size % 8 != 0:
        print(f"ERROR: {path.name} size {old_size} not a multiple of 8", file=sys.stderr)
        sys.exit(1)
    ws_section = data[:WS_BYTES_OLD]
    rest = data[WS_BYTES_OLD:]
    new_data = ws_section + struct.pack(f"<{NEW_WS - OLD_WS}d", *([0.0] * (NEW_WS - OLD_WS))) + rest
    path.write_bytes(new_data)
    print(f"  {path.name}: {old_size} -> {len(new_data)} bytes (+{EXTRA_BYTES})")


def main():
    print(f"Extending ws from {OLD_WS} to {NEW_WS} f64 values (+{EXTRA_BYTES} bytes per file):")
    for p in WEIGHT_FILES:
        if not p.exists():
            print(f"  SKIP (not found): {p}")
            continue
        # Check if already extended
        size = p.stat().st_size
        # Old size should be divisible by 8; after extension it grows by EXTRA_BYTES
        extend_file(p)
    print("Done. Rebuild the engine to apply.")


if __name__ == "__main__":
    main()
