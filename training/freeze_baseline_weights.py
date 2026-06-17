"""Build net_weights_frozen.bin — original v13 HalfPW, schema-matched to training blob.

Source: engine/baseline/net_weights.baseline.bin (WSKIP=13, no field planes).
Output: engine/src/acev13/net_weights_frozen.bin (+ copy under engine/baseline/).

Training/deploy weights stay in net_weights.bin — this file is never overwritten
by micro-train or deploy. Re-run after changing the baseline source:

    python training/freeze_baseline_weights.py
"""

from __future__ import annotations

import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "engine" / "baseline" / "net_weights.baseline.bin"
OUT_ENGINE = ROOT / "engine" / "src" / "acev13" / "net_weights_frozen.bin"
OUT_BASELINE = ROOT / "engine" / "baseline" / "net_weights_frozen.bin"

OLD_WSKIP = 13
NEW_WSKIP = 16
NET_H = 32
W1C_LEN = 9 * 128 * NET_H
PO_LEN = 81 * NET_H
PX_LEN = 81 * NET_H
FIELD_LEN = 81 * NET_H

OLD_BASE_F64S = OLD_WSKIP + NET_H + NET_H + W1C_LEN + PO_LEN + PX_LEN
NEW_BASE_F64S = NEW_WSKIP + NET_H + NET_H + W1C_LEN + PO_LEN + PX_LEN
PLANES11_F64S = NEW_BASE_F64S + FIELD_LEN * 11


def extend_wskip(data: bytes) -> bytes:
    n = len(data) // 8
    if n == NEW_BASE_F64S:
        return data
    if n != OLD_BASE_F64S:
        raise ValueError(f"baseline size {n} f64s (expected {OLD_BASE_F64S} or {NEW_BASE_F64S})")
    split = OLD_WSKIP * 8
    extra = struct.pack("<3d", 0.0, 0.0, 0.0)
    return data[:split] + extra + data[split:]


def extend_field_planes(data: bytes) -> bytes:
    n = len(data) // 8
    if n == PLANES11_F64S:
        return data
    if n != NEW_BASE_F64S:
        raise ValueError(f"after wskip extend: {n} f64s (expected {NEW_BASE_F64S} or {PLANES11_F64S})")
    zeros = struct.pack(f"<{FIELD_LEN * 11}d", *([0.0] * (FIELD_LEN * 11)))
    return data + zeros


def main() -> None:
    if not SRC.exists():
        raise FileNotFoundError(f"missing baseline source: {SRC}")
    raw = SRC.read_bytes()
    out = extend_field_planes(extend_wskip(raw))
    assert len(out) == PLANES11_F64S * 8, (len(out), PLANES11_F64S * 8)
    OUT_ENGINE.parent.mkdir(parents=True, exist_ok=True)
    OUT_ENGINE.write_bytes(out)
    OUT_BASELINE.parent.mkdir(parents=True, exist_ok=True)
    OUT_BASELINE.write_bytes(out)
    print(f"frozen HalfPW: {PLANES11_F64S} f64s ({len(out)} bytes)")
    print(f"  -> {OUT_ENGINE}")
    print(f"  -> {OUT_BASELINE}")
    print("Rebuild titanium.exe so include_bytes! picks up net_weights_frozen.bin")


if __name__ == "__main__":
    main()
