"""HalfPW (gen13 ACE) net — Python port of the engine forward pass.

Must match `titanium/search.rs::evaluate` bit-for-bit (`parity_check.py`).

Field plane names: see `training/field_planes.py` and `engine/src/titanium/field_planes.rs`.
Blob: 11 planes × 81×32 (goal_inv, pawn_fwd, corridor_delta, path_cross, choke×2, contested).
"""

import struct
from dataclasses import dataclass

from titanium_training.models.field_planes import (
    CAT_RAW_ME,
    CAT_RAW_OPP,
    CAT_PROPAGATED_ME,
    CAT_PROPAGATED_OPP,
    CAT_PROPAGATED_COMBINED,
    CHOKE_P0,
    CHOKE_P1,
    CONTESTED,
    CORRIDOR_DELTA_P0,
    CORRIDOR_DELTA_P1,
    encode_contested,
    FIELD_PLANE_COUNT,
    GOAL_INV_P0,
    GOAL_INV_P1,
    PATH_CROSS_P0,
    PATH_CROSS_P1,
    PAWN_FWD_P0,
    PAWN_FWD_P1,
    ROUTE_CONTESTED,
    ROUTE_ME,
    ROUTE_NEAR_ME,
    ROUTE_NEAR_OPP,
    ROUTE_OPP,
    compact_catv5_precise_vectors,
    compact_route_vectors,
    rec_field,
)

H_HEADER_LEN = 8
WSKIP_LEN = 20
FIELD_LEN = 81
# Backward-compat default (only used by standalone tools that still assume a
# fixed 32-wide legacy blob). Real loads determine width from the blob's own
# 8-byte NET_H header -- see Net.load().
NET_H = 32
W1C_LEN = 9 * 128 * NET_H
PO_LEN = 81 * NET_H
PX_LEN = 81 * NET_H
NET_WEIGHT_F64S = WSKIP_LEN + NET_H + NET_H + W1C_LEN + PO_LEN + PX_LEN + FIELD_LEN * FIELD_PLANE_COUNT


def _payload_f64s(h: int) -> int:
    return WSKIP_LEN + h + h + 9 * 128 * h + 81 * h + 81 * h + FIELD_LEN * FIELD_PLANE_COUNT

# True 180-degree side-to-move canonicalization (reverse row and column).
NET_MIRC = [(8 - i // 9) * 9 + (8 - i % 9) for i in range(81)]
NET_MIRS = [(7 - i // 8) * 8 + (7 - i % 8) for i in range(64)]
NET_BKT = [(i // 9 // 3) * 3 + (i % 9) // 3 for i in range(81)]
LEGAL_WALL_SLOTS = 128


def legal_wall_norm(rec: dict) -> float:
    """Retired ws[14] input. The engine now feeds zero here."""
    return 0.0


def opponent_corridor_width(rec: dict, me: int, _d_me_i: int, d_opp_i: int) -> int:
    """ws[15] input — opponent cells on their shortest-path rank."""
    d0f = rec_field(rec, GOAL_INV_P0)
    d1f = rec_field(rec, GOAL_INV_P1)
    field = d1f if me == 0 else d0f
    return sum(1 for d in field if d == d_opp_i)


@dataclass
class Net:
    h: int
    ws: list
    b1: list
    w2: list
    w1c: list
    po: list
    px: list
    route_me: list
    route_opp: list
    route_near_me: list
    route_near_opp: list
    route_contested: list
    cat_raw_me: list
    cat_raw_opp: list
    cat_propagated_me: list
    cat_propagated_opp: list
    cat_propagated_combined: list

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            raw = f.read()
        (h,) = struct.unpack("<Q", raw[:H_HEADER_LEN])
        body = raw[H_HEADER_LEN:]
        w1c_len, po_len, px_len = 9 * 128 * h, 81 * h, 81 * h
        route_only_f64s = _payload_f64s(h) - 5 * FIELD_LEN
        cat_v5_f64s = route_only_f64s + FIELD_LEN
        cat_v5_witness_f64s = route_only_f64s + 3 * FIELD_LEN
        full_f64s = _payload_f64s(h)
        input_f64s = len(body) // 8
        assert input_f64s in (route_only_f64s, cat_v5_f64s, cat_v5_witness_f64s, full_f64s), (
            f"size {len(body)} is not a supported HalfPW blob size for declared NET_H={h}"
        )
        vals = list(struct.unpack(f"<{len(body) // 8}d", body))
        o = 0

        def take(n):
            nonlocal o
            s = vals[o:o + n]
            o += n
            return s

        prefix = [
            h,
            take(WSKIP_LEN), take(h), take(h),
            take(w1c_len), take(po_len), take(px_len),
            take(FIELD_LEN), take(FIELD_LEN), take(FIELD_LEN), take(FIELD_LEN),
            take(FIELD_LEN),
        ]
        if input_f64s == full_f64s:
            cats = [take(FIELD_LEN) for _ in range(5)]
        elif input_f64s == cat_v5_witness_f64s:
            raw_me, raw_opp, combined = take(FIELD_LEN), take(FIELD_LEN), take(FIELD_LEN)
            cats = [
                [w * 4.0 for w in raw_me],
                [w * 4.0 for w in raw_opp],
                [0.0] * FIELD_LEN,
                [0.0] * FIELD_LEN,
                [w * (400.0 / 256.0) for w in combined],
            ]
        elif input_f64s == cat_v5_f64s:
            combined = take(FIELD_LEN)
            cats = [[0.0] * FIELD_LEN for _ in range(4)] + [
                [w * (400.0 / 256.0) for w in combined]
            ]
        else:
            cats = [[0.0] * FIELD_LEN for _ in range(5)]
        return Net(*prefix, *cats)


def _cell_feats(goal_f, player_f, delta_f, cross_f, choke_f) -> tuple:
    gf, pf, df, cf, chf = [], [], [], [], []
    for i in range(81):
        dg = goal_f[i] if i < len(goal_f) else 255
        if dg == 255:
            gf.append(0.0)
            pf.append(0.0)
            df.append(0.0)
            cf.append(0.0)
            chf.append(0.0)
            continue
        gf.append(dg / 16.0)
        ps = player_f[i] if i < len(player_f) else 255
        pf.append(0.0 if ps == 255 else ps / 16.0)
        dt = delta_f[i] if i < len(delta_f) else 255
        df.append(0.0 if dt == 255 else dt / 16.0)
        cv = cross_f[i] if i < len(cross_f) else 0
        cf.append(0.0 if not cv else cv / 16.0)
        hv = choke_f[i] if i < len(choke_f) else 0
        chf.append(hv / 16.0 if hv else 0.0)
    return gf, pf, df, cf, chf


def _contested_vec(delta0_raw, delta1_raw, contested_raw) -> list[float]:
    out = []
    for i in range(81):
        if contested_raw and i < len(contested_raw) and contested_raw[i]:
            out.append(contested_raw[i] / 16.0)
            continue
        d0 = delta0_raw[i] if i < len(delta0_raw) else 255
        d1 = delta1_raw[i] if i < len(delta1_raw) else 255
        out.append(encode_contested(d0, d1))
    return out


def _route_score(net: Net, rec: dict) -> float:
    route_me, route_opp, near_me, near_opp, contested = compact_route_vectors(rec, NET_MIRC)
    cat_raw_me, cat_raw_opp, cat_prop_me, cat_prop_opp, cat_combined = compact_catv5_precise_vectors(rec, NET_MIRC)
    return sum(
        net.route_me[i] * route_me[i]
        + net.route_opp[i] * route_opp[i]
        + net.route_near_me[i] * near_me[i]
        + net.route_near_opp[i] * near_opp[i]
        + net.route_contested[i] * contested[i]
        + net.cat_raw_me[i] * cat_raw_me[i]
        + net.cat_raw_opp[i] * cat_raw_opp[i]
        + net.cat_propagated_me[i] * cat_prop_me[i]
        + net.cat_propagated_opp[i] * cat_prop_opp[i]
        + net.cat_propagated_combined[i] * cat_combined[i]
        for i in range(81)
    )


def forward_trace(net, rec, normed: bool = False):
    """Full forward with intermediate tensors (``normed=False`` matches the engine)."""
    if normed:
        raise ValueError("forward_trace only supports normed=False (engine raw path)")
    from titanium_training.models.eval_forward import forward_trace_from_record

    return forward_trace_from_record(net, rec)


def forward(net, rec, normed: bool = False):
    """Reproduce the engine's walls-present net eval for one feature record.

    DEFAULT ``normed=False`` — the engine (`search.rs evaluate()`) feeds RAW
    distance/wall inputs (`d_me = d_me_i as f64`, products scaled by /20 and /16);
    there is NO `normed` branch in the engine. This raw path matches the engine
    bit-for-bit (parity_check: 6/6 within 1cp). Training MUST use this so the net
    is optimized against the eval the engine actually computes — `normed=True`
    trained a normalized formula the engine never applies, which silently
    miscalibrated the `ws` skip weights (a model-collapse cause).

    ``normed=True`` is retained only as a dead legacy branch; nothing should use it.
    """
    if normed:
        raise ValueError("normed=True is retired; the engine uses raw scalars only")
    return forward_trace(net, rec, normed=False).final_cp
