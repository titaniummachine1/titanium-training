"""Extend net_weights.bin from WSKIP_LEN=13 to WSKIP_LEN=16.

Appends 3 zero-valued f64 scalars immediately after the original 13 ws[] values,
shifting the remaining blobs (b1, w2, w1c, po, px) by 24 bytes.  The new weights
(ws[13]=fragile-lead, ws[14]=corridor-width-me, ws[15]=corridor-width-opp) are
zero-initialised so the net is behaviour-identical before retraining.

Run once from the repo root:
    python training/extend_weights.py

Writes the extended file in-place.  The original baseline copy is in
engine/baseline/net_weights.baseline.bin and is NOT modified.
"""
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC  = ROOT / "engine" / "src" / "acev13" / "net_weights.bin"
BSLN = ROOT / "engine" / "baseline" / "net_weights.baseline.bin"

OLD_WSKIP = 13
NEW_WSKIP = 16
EXTRA     = NEW_WSKIP - OLD_WSKIP  # 3 extra zero f64s

NET_H    = 32
W1C_LEN  = 9 * 128 * NET_H
PO_LEN   = 81 * NET_H
PX_LEN   = 81 * NET_H
OLD_TOTAL = OLD_WSKIP + NET_H + NET_H + W1C_LEN + PO_LEN + PX_LEN
NEW_TOTAL = NEW_WSKIP + NET_H + NET_H + W1C_LEN + PO_LEN + PX_LEN

data = SRC.read_bytes()
if len(data) == NEW_TOTAL * 8:
    print("Already extended — nothing to do.")
elif len(data) != OLD_TOTAL * 8:
    raise ValueError(f"Unexpected file size {len(data)} (expected {OLD_TOTAL*8} or {NEW_TOTAL*8})")
else:
    # Splice 3 zeros after the first OLD_WSKIP f64s (= 13*8 = 104 bytes)
    split = OLD_WSKIP * 8
    extended = data[:split] + struct.pack(f"<{EXTRA}d", *([0.0] * EXTRA)) + data[split:]
    assert len(extended) == NEW_TOTAL * 8
    SRC.write_bytes(extended)
    print(f"Extended {SRC.name}: {OLD_TOTAL} -> {NEW_TOTAL} f64s ({EXTRA} zero scalars inserted at ws[{OLD_WSKIP}])")
    print("Baseline copy is untouched:", BSLN)
