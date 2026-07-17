#!/usr/bin/env python3
"""Quick NNUE input profile — where compute likely lives (no code changes)."""
from __future__ import annotations

import struct
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WEIGHTS = REPO / "engine" / "src" / "titanium" / "net_weights.bin"

NET_H = 32
WSKIP = 18
W1C = 9 * 128 * NET_H
PO = 81 * NET_H
PX = 81 * NET_H
PLANES = 5 * 81
F64 = WSKIP + NET_H + NET_H + W1C + PO + PX + PLANES


def main() -> None:
    data = WEIGHTS.read_bytes()
    n = len(data) // 8
    print(f"net_weights.bin: {len(data)} bytes ({n} f64)")
    print(f"Expected f64 count: {F64}")
    print()
    print("Parameter blocks (f64 count / bytes):")
    blocks = [
        ("ws skip (scalars)", WSKIP),
        ("b1 hidden bias", NET_H),
        ("w2 output", NET_H),
        ("w1c wall buckets 9x128x32", W1C),
        ("po pawn me 81x32", PO),
        ("px pawn opp 81x32", PX),
        ("route planes 5x81", PLANES),
    ]
    off = 0
    for name, cnt in blocks:
        print(f"  {name:28} {cnt:6}  {cnt*8:7} B  ({100*cnt/n:5.1f}%)")
        off += cnt
    print()
    print("Hot-path notes (eval/search):")
    print("  - w1c dot: 9 bucket x up to 128 wall slots x 32 — sparse if few walls")
    print("  - po/px: 2x81 indexed adds (always 2 lookups per eval)")
    print("  - route planes: 5x81 sparse multiply-add when route_active")
    print("  - ws[0..17]: 18 scalar features — cheap")
    print("  - BFS field planes: computed once per eval-packed-batch, not in search hot loop")
    print("  - Incremental accumulator in search skips full w1c rebuild on pawn moves")
    print()
    print("CLI vs WASM:")
    print("  - CLI: fresh process per genmove OR session; TITANIUM_NET_WEIGHTS_PATH optional")
    print("  - WASM: embed-tables build uses include_bytes net_weights.bin — independent of website state")
    print("  - Website must not be required for training/self-play (subprocess CLI only)")


if __name__ == "__main__":
    main()
