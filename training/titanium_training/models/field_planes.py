"""NNUE field plane names — single source of truth for engine JSON and trainer.

Philosophy: BFS/search owns exact geometry; NN compresses topology into priors (H=32).
See engine/src/titanium/field_planes.rs for the full table and pre-training notes.

Do NOT add extra BFS / wall-delta planes here — those belong in search.
Optional later: block_pressure (pawn interferes with route) if tactical losses warrant it.
"""

# Canonical JSON keys (eval --json / datagen)
GOAL_INV_P0 = "goal_inv_p0_field"
GOAL_INV_P1 = "goal_inv_p1_field"
PAWN_FWD_P0 = "pawn_fwd_p0_field"
PAWN_FWD_P1 = "pawn_fwd_p1_field"
CORRIDOR_DELTA_P0 = "corridor_delta_p0_field"
CORRIDOR_DELTA_P1 = "corridor_delta_p1_field"
PATH_CROSS_P0 = "path_cross_p0_field"
PATH_CROSS_P1 = "path_cross_p1_field"
CHOKE_P0 = "choke_p0_field"
CHOKE_P1 = "choke_p1_field"
CONTESTED = "contested_field"

# Legacy aliases (older overnight JSONL before rename)
_LEGACY = {
    GOAL_INV_P0: ("d0_field",),
    GOAL_INV_P1: ("d1_field",),
    PAWN_FWD_P0: ("player0_field",),
    PAWN_FWD_P1: ("player1_field",),
    CORRIDOR_DELTA_P0: ("delta0_field",),
    CORRIDOR_DELTA_P1: ("delta1_field",),
    PATH_CROSS_P0: ("cross0_field",),
    PATH_CROSS_P1: ("cross1_field",),
}


def encode_contested(delta_p0: int, delta_p1: int) -> float:
    """Continuous shared importance: 1/(1+d0+d1), u8 stored as round(16×value)÷16."""
    if delta_p0 == 255 or delta_p1 == 255:
        return 0.0
    raw = min(round(16 / (1 + delta_p0 + delta_p1)), 16)
    return raw / 16.0


def rec_field(rec: dict, canonical_key: str) -> list:
    """Read a per-cell field from a training record (canonical or legacy key)."""
    val = rec.get(canonical_key)
    if val:
        return val
    for alt in _LEGACY.get(canonical_key, ()):
        val = rec.get(alt)
        if val:
            return val
    return []

# Compact sparse route features consumed by the network. The larger fields
# above remain the engine's analysis/data format and are reduced to these masks
# only when a position is expanded for training.
ROUTE_ME = "route_me"
ROUTE_OPP = "route_opp"
ROUTE_NEAR_ME = "route_near_me"
ROUTE_NEAR_OPP = "route_near_opp"
ROUTE_CONTESTED = "route_contested"
# CATv5 precise inputs. The raw path-ID/rank maps are 0..4 and paths are
# disjoint after their permitted shared first ply, so two compact maps retain
# the four exact witnesses without eight extra hot-eval planes.
CAT_RAW_ME = "cat_raw_me"
CAT_RAW_OPP = "cat_raw_opp"
CAT_PROPAGATED_ME = "cat_propagated_me"
CAT_PROPAGATED_OPP = "cat_propagated_opp"
CAT_PROPAGATED_COMBINED = "cat_propagated_combined"
CAT_WITNESS_P0_FIELD = "cat_witness_p0_field"
CAT_WITNESS_P1_FIELD = "cat_witness_p1_field"
CAT_PROPAGATED_P0_FIELD = "cat_propagated_p0_field"
CAT_PROPAGATED_P1_FIELD = "cat_propagated_p1_field"
CAT_PROPAGATED_FIELD = "cat_propagated_field"
ROUTE_P0_FIELD = "route_p0_field"
ROUTE_P1_FIELD = "route_p1_field"
ROUTE_FLANK_P0_FIELD = "route_flank_p0_field"
ROUTE_FLANK_P1_FIELD = "route_flank_p1_field"

# Weight blob plane order (must match titanium/net.rs load order).
WEIGHT_PLANE_ORDER = (
    ROUTE_ME,
    ROUTE_OPP,
    ROUTE_NEAR_ME,
    ROUTE_NEAR_OPP,
    ROUTE_CONTESTED,
    CAT_RAW_ME,
    CAT_RAW_OPP,
    CAT_PROPAGATED_ME,
    CAT_PROPAGATED_OPP,
    CAT_PROPAGATED_COMBINED,
)
FIELD_PLANE_COUNT = len(WEIGHT_PLANE_ORDER)


def compact_route_vectors(rec: dict, mirc: list[int]) -> tuple[list[float], ...]:
    """Canonical sparse masks derived from exact forward/goal distance fields."""
    goal0 = rec_field(rec, GOAL_INV_P0)
    goal1 = rec_field(rec, GOAL_INV_P1)
    from0 = rec_field(rec, PAWN_FWD_P0)
    from1 = rec_field(rec, PAWN_FWD_P1)
    if not all(len(v) == 81 for v in (goal0, goal1, from0, from1)):
        raise KeyError("compact route inputs require both 81-cell goal and pawn fields")
    route0_raw = rec.get(ROUTE_P0_FIELD)
    route1_raw = rec.get(ROUTE_P1_FIELD)
    near0_raw = rec.get(ROUTE_FLANK_P0_FIELD)
    near1_raw = rec.get(ROUTE_FLANK_P1_FIELD)
    if all(v and len(v) == 81 for v in (route0_raw, route1_raw, near0_raw, near1_raw)):
        route0 = [float(v) for v in route0_raw]
        route1 = [float(v) for v in route1_raw]
        near0 = [float(v) for v in near0_raw]
        near1 = [float(v) for v in near1_raw]
    else:
        shortest0 = goal0[rec["pawn0"]]
        shortest1 = goal1[rec["pawn1"]]
        route0 = [0.0] * 81
        near0 = [0.0] * 81
        route1 = [0.0] * 81
        near1 = [0.0] * 81
        for i in range(81):
            t0 = from0[i] + goal0[i]
            t1 = from1[i] + goal1[i]
            route0[i] = float(t0 == shortest0)
            near0[i] = float(t0 == shortest0 + 2)
            route1[i] = float(t1 == shortest1)
            near1[i] = float(t1 == shortest1 + 2)

    if rec["turn"] == 0:
        me, opp, near_me, near_opp = route0, route1, near0, near1
    else:
        me = [route1[mirc[i]] for i in range(81)]
        opp = [route0[mirc[i]] for i in range(81)]
        near_me = [near1[mirc[i]] for i in range(81)]
        near_opp = [near0[mirc[i]] for i in range(81)]
    contested = [float((me[i] or near_me[i]) and (opp[i] or near_opp[i])) for i in range(81)]
    return me, opp, near_me, near_opp, contested


def compact_catv5_precise_vectors(rec: dict, mirc: list[int]) -> tuple[list[float], ...]:
    """Canonical exact-path and propagated CAT inputs, all bounded in [0,1]."""
    witness0 = rec.get(CAT_WITNESS_P0_FIELD)
    witness1 = rec.get(CAT_WITNESS_P1_FIELD)
    prop0 = rec.get(CAT_PROPAGATED_P0_FIELD)
    prop1 = rec.get(CAT_PROPAGATED_P1_FIELD)
    combined = rec.get(CAT_PROPAGATED_FIELD)
    if not all(
        isinstance(v, list) and len(v) == 81
        for v in (witness0, witness1, prop0, prop1, combined)
    ):
        zeros = [0.0] * 81
        return tuple(zeros.copy() for _ in range(5))

    turn = int(rec.get("turn", 0))
    index = list(range(81)) if turn == 0 else mirc
    own_prop = prop0 if turn == 0 else prop1
    opp_prop = prop1 if turn == 0 else prop0

    own_witness = witness0 if turn == 0 else witness1
    opp_witness = witness1 if turn == 0 else witness0
    raw_me = [min(1.0, max(0.0, float(own_witness[index[i]]) / 4.0)) for i in range(81)]
    raw_opp = [min(1.0, max(0.0, float(opp_witness[index[i]]) / 4.0)) for i in range(81)]
    prop_me = [min(1.0, max(0.0, float(own_prop[index[i]]) / 200.0)) for i in range(81)]
    prop_opp = [min(1.0, max(0.0, float(opp_prop[index[i]]) / 200.0)) for i in range(81)]
    prop_combined = [
        min(1.0, max(0.0, float(combined[index[i]]) / 400.0)) for i in range(81)
    ]
    return raw_me, raw_opp, prop_me, prop_opp, prop_combined


def compact_catv5_propagated_vector(rec: dict, mirc: list[int]) -> list[float]:
    """Backward-compatible accessor for the normalized CATv5 heat plane."""
    return compact_catv5_precise_vectors(rec, mirc)[-1]
