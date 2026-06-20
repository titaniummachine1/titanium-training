"""Extend net_weights.bin with per-player field-plane weights.

Planes (each 81×32, blob order matches net.rs):
  goal_inv×2, pawn_fwd×2, corridor_delta×2, path_cross×2,
  choke×2, contested (11 total)

Migrates:
  - base blob (no field planes)
  - legacy 3-plane STM layout
  - 6-plane / 8-plane layouts
  - 11-plane → no-op

Run once from repo root:
    python training/extend_field_planes.py
"""
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "engine" / "src" / "acev13" / "net_weights.bin"

NET_H = 32
WSKIP = 16
W1C_LEN = 9 * 128 * NET_H
PO_LEN = 81 * NET_H
PX_LEN = 81 * NET_H
FIELD_LEN = 81 * NET_H

BASE_F64S = WSKIP + NET_H + NET_H + W1C_LEN + PO_LEN + PX_LEN
LEGACY_F64S = BASE_F64S + FIELD_LEN * 3
PLANES6_F64S = BASE_F64S + FIELD_LEN * 6
PLANES8_F64S = BASE_F64S + FIELD_LEN * 8
PLANES11_F64S = BASE_F64S + FIELD_LEN * 11


def main() -> None:
    data = SRC.read_bytes()
    n = len(data) // 8
    if n == PLANES11_F64S:
        print("All 11 field planes present — nothing to do.")
        return
    if n == PLANES8_F64S:
        zeros = struct.pack(f"<{FIELD_LEN * 3}d", *([0.0] * (FIELD_LEN * 3)))
        SRC.write_bytes(data + zeros)
        print(f"Extended {SRC.name}: {PLANES8_F64S} -> {PLANES11_F64S} f64s (+zero choke×2 contested)")
        return
    if n == PLANES6_F64S:
        zeros = struct.pack(f"<{FIELD_LEN * 5}d", *([0.0] * (FIELD_LEN * 5)))
        SRC.write_bytes(data + zeros)
        print(f"Extended {SRC.name}: {PLANES6_F64S} -> {PLANES11_F64S} f64s (+zero path_cross/choke/contested)")
        return
    if n == LEGACY_F64S:
        vals = list(struct.unpack(f"<{n}d", data))
        tail = vals[BASE_F64S:]
        goal_inv, pawn_fwd, corridor_delta = (
            tail[0:FIELD_LEN],
            tail[FIELD_LEN : FIELD_LEN * 2],
            tail[FIELD_LEN * 2 : FIELD_LEN * 3],
        )
        head = vals[:BASE_F64S]
        zeros = [0.0] * FIELD_LEN
        out = head + goal_inv + zeros + pawn_fwd + zeros + corridor_delta + zeros + zeros + zeros + zeros + zeros + zeros
        assert len(out) == PLANES11_F64S
        SRC.write_bytes(struct.pack(f"<{PLANES11_F64S}d", *out))
        print(f"Migrated {SRC.name}: legacy -> {PLANES11_F64S} f64s")
        return
    if n == BASE_F64S:
        zeros = struct.pack(f"<{FIELD_LEN * 11}d", *([0.0] * (FIELD_LEN * 11)))
        SRC.write_bytes(data + zeros)
        print(f"Extended {SRC.name}: {BASE_F64S} -> {PLANES11_F64S} f64s")
        return
    raise ValueError(
        f"Unexpected size {n} f64s (expected {BASE_F64S}, {LEGACY_F64S}, "
        f"{PLANES6_F64S}, {PLANES8_F64S}, or {PLANES11_F64S})"
    )


if __name__ == "__main__":
    main()
