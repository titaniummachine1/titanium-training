"""HalfPW (gen13 ACE) net — Python port of the engine forward pass.

This is the trainer's reference forward pass. It must reproduce the Rust engine
(`acev13/search.rs::evaluate`, walls-present net path) bit-for-bit; verified
against `titanium eval <moves> --json` by `parity_check.py`. Getting this exact
BEFORE training is the discipline that lets us fine-tune the existing weights
(and later add the new input planes) without the net silently mis-evaluating
in-engine — the classic NNUE train/inference-mismatch trap.

Blob layout (little-endian f64), from `acev13/net.rs`:
    ws[16]  b1[32]  w2[32]  w1c[9*128*32]  po[81*32]  px[81*32]

ws[0..12] = original scalar terms (unchanged from gen13).
ws[13]    = tempo * opp-wall-count / 10  (fragile-lead cross-term).
ws[14]    = corridor-width-me  (cells sharing the me-pawn's distance-to-goal rank).
ws[15]    = corridor-width-opp (same for opponent).
ws[13..15] are zero-initialised so the net is behaviour-identical before retraining.
"""

import struct
from dataclasses import dataclass

NET_H = 32
WSKIP_LEN = 16
W1C_LEN = 9 * 128 * NET_H
PO_LEN = 81 * NET_H
PX_LEN = 81 * NET_H

# Symmetry tables — exact ports of net.rs build_mirc / build_mirs / build_bkt.
NET_MIRC = [(8 - i // 9) * 9 + i % 9 for i in range(81)]   # mirror pawn row
NET_MIRS = [(7 - i // 8) * 8 + i % 8 for i in range(64)]   # mirror wall slot
NET_BKT = [(i // 9 // 3) * 3 + (i % 9) // 3 for i in range(81)]  # 3x3 bucket


@dataclass
class Net:
    ws: list
    b1: list
    w2: list
    w1c: list
    po: list
    px: list

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            raw = f.read()
        total = WSKIP_LEN + NET_H + NET_H + W1C_LEN + PO_LEN + PX_LEN
        assert len(raw) == total * 8, f"size {len(raw)} != {total*8}"
        vals = list(struct.unpack(f"<{total}d", raw))
        o = 0
        def take(n):
            nonlocal o
            s = vals[o:o + n]
            o += n
            return s
        return Net(take(WSKIP_LEN), take(NET_H), take(NET_H),
                   take(W1C_LEN), take(PO_LEN), take(PX_LEN))


def forward(net, rec):
    """Reproduce the engine's walls-present net eval for one feature record.

    `rec` is the JSON from `titanium eval ... --json`:
      turn, pawn0, pawn1, wl0, wl1, d0, d1,
      d0_field[81], d1_field[81],   <- full BFS distance arrays (new in ws[16] format)
      hw[64], vw[64].
    Returns the integer centipawn eval (truncated toward zero, as the Rust `as i32`).
    """
    me = rec["turn"]
    opp = 1 - me
    wl = [rec["wl0"], rec["wl1"]]
    dist = [rec["d0"], rec["d1"]]
    d_me = float(dist[me])
    d_opp = float(dist[opp])
    w_me = float(wl[me])
    w_opp = float(wl[opp])
    ws = net.ws

    pd = d_opp - d_me
    wd = w_me - w_opp
    out = (ws[0] + ws[1] * pd + ws[2] * wd + ws[3] * d_me + ws[4] * d_opp
           + ws[9] * pd * (w_me + w_opp) / 20.0
           + ws[10] * wd * (d_me + d_opp) / 16.0)
    if w_opp == 0.0:
        out += ws[6]
        if d_me <= d_opp:
            out += ws[5]
    elif w_me == 0.0:
        out += ws[8]
        if d_opp <= d_me - 1.0:
            out += ws[7]
    if d_opp <= 4.0:
        out += ws[11] * (w_me if w_me < 3.0 else 3.0)
    if d_me <= 4.0:
        out += ws[12] * (w_opp if w_opp < 3.0 else 3.0)

    # ws[13]: fragile-lead (tempo * opp walls)
    out += ws[13] * pd * w_opp / 10.0

    # ws[14..15]: corridor-width proxies.
    # d0_field[cell] = BFS distance from cell to P0's goal row (=d0 inverse BFS).
    # Count cells sharing the pawn's distance rank = how wide the corridor is at that depth.
    d_me_i = int(d_me)
    d_opp_i = int(d_opp)
    if me == 0:
        d0f = rec.get("d0_field", [])
        d1f = rec.get("d1_field", [])
    else:
        d0f = rec.get("d0_field", [])
        d1f = rec.get("d1_field", [])
    # width_me: cells where P0's BFS dist == d_me (for me=0) / P1's BFS dist == d_opp (for me=1)
    width_me  = sum(1 for d in (d0f if me == 0 else d1f) if d == d_me_i)
    width_opp = sum(1 for d in (d1f if me == 0 else d0f) if d == d_opp_i)
    out += ws[14] * width_me + ws[15] * width_opp

    pawn0, pawn1 = rec["pawn0"], rec["pawn1"]
    hw, vw = rec["hw"], rec["vw"]
    hid = [0.0] * NET_H

    if me == 0:
        b0 = NET_BKT[pawn0]
        acc = [0.0] * NET_H
        for s in range(64):
            if hw[s]:
                o = (b0 * 128 + s) * NET_H
                for j in range(NET_H):
                    acc[j] += net.w1c[o + j]
            if vw[s]:
                o = (b0 * 128 + 64 + s) * NET_H
                for j in range(NET_H):
                    acc[j] += net.w1c[o + j]
        po0 = pawn0 * NET_H
        px1 = pawn1 * NET_H
        for j in range(NET_H):
            hid[j] = net.b1[j] + acc[j] + net.po[po0 + j] + net.px[px1 + j]
    else:
        b1v = NET_BKT[NET_MIRC[pawn1]]
        acc = [0.0] * NET_H
        for s in range(64):
            if hw[s]:
                o = (b1v * 128 + NET_MIRS[s]) * NET_H
                for j in range(NET_H):
                    acc[j] += net.w1c[o + j]
            if vw[s]:
                o = (b1v * 128 + 64 + NET_MIRS[s]) * NET_H
                for j in range(NET_H):
                    acc[j] += net.w1c[o + j]
        po0 = NET_MIRC[pawn1] * NET_H
        px1 = NET_MIRC[pawn0] * NET_H
        for j in range(NET_H):
            hid[j] = net.b1[j] + acc[j] + net.po[po0 + j] + net.px[px1 + j]

    for j in range(NET_H):
        a2 = min(1.0, max(0.0, hid[j]))
        out += net.w2[j] * a2 * 200.0

    return int(out)  # truncate toward zero, matching Rust `out as i32`
