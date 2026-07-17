#!/usr/bin/env python3
"""Build a fresh-init HalfPW net that keeps ONLY the trained cat_heat field
and randomizes everything else.

Rationale (2026-07-10): the current lineage's hidden layer (ws/b1/w2/w1c/po/px)
and route_* fields have already been trained repeatedly on the accumulated
dataset -- continuing to train the SAME weights on the SAME backlog risks
overfitting/diminishing returns rather than learning anything new, especially
now that label_resolution.py's blending/confidence recalibration makes that
backlog meaningfully higher quality than what these weights originally saw.
cat_heat is kept as-is because it is a small (81-float), width-independent,
already-correctly-trained field (see NNUE v16 architecture memory) -- CAT
attention coverage is not something this reinit should discard.

Output is a valid net_weights.bin-format blob (H_HEADER_LEN + little-endian
f64 payload, field order matching halfpw.py's Net.load() / trainer.py's
HalfPW.__init__ exactly) at a caller-chosen hidden width (default 32, matching
the ACE v13 baseline width so the plateau-gated net2net auto-widen policy
regrows it from scratch rather than starting mid-curriculum).
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import torch

_TRAINING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TRAINING))

H_HEADER_LEN = 8
WSKIP_LEN = 20
FIELD_LEN = 81
FIELD_SHAPE = (81,)

# Field order MUST match halfpw.py Net.load() / trainer.py HalfPW.__init__.
_RAW_ORDER = [
    "ws", "b1", "w2", "w1c", "po", "px",
    "route_me", "route_opp", "route_near_me", "route_near_opp",
    "route_contested", "cat_heat",
]

# Random-init scales chosen to match the *typical* trained magnitude of each
# field in the current live net (epoch 42, h=96: b1 std~0.16, w2 std~0.14,
# w1c std~0.08, po std~0.07, px std~0.16, route_* std~0.05-0.08) so the fresh
# net starts in a numerically sane range rather than either collapsing to
# zero-signal or exploding. ws is a deliberate exception: those scale raw
# large-magnitude scalar inputs directly to centipawns (trained std ~260,
# max ~800) -- initializing at that scale would make early predictions wildly
# overconfident, so it gets a smaller random start and lets gradient descent
# grow it, same as any other field here.
_INIT_STD = {
    "ws": 20.0,
    "b1": 0.1,
    "w2": 0.1,
    "w1c": 0.05,
    "po": 0.05,
    "px": 0.05,
    "route_me": 0.05,
    "route_opp": 0.05,
    "route_near_me": 0.05,
    "route_near_opp": 0.05,
    "route_contested": 0.05,
}


FIELD_PLANE_COUNT = 6  # route_me, route_opp, route_near_me, route_near_opp, route_contested, cat_heat


def _payload_f64s(h: int) -> int:
    return WSKIP_LEN + h + h + 9 * 128 * h + 81 * h + 81 * h + FIELD_LEN * FIELD_PLANE_COUNT


def load_cat_heat(bin_path: Path) -> tuple[list[float], int]:
    """Extract the cat_heat field (last 81 doubles) and the source's own
    hidden width. The width is returned so callers default to reinitializing
    at the SAME width -- this tool changes weight values, not network shape;
    shrinking/growing h is a separate, deliberate decision (net2net widen),
    not something a reinit should do as a side effect."""
    raw = bin_path.read_bytes()
    (h,) = struct.unpack("<Q", raw[:H_HEADER_LEN])
    body = raw[H_HEADER_LEN:]
    full_f64s = _payload_f64s(h)
    legacy_f64s = full_f64s - FIELD_LEN
    n_vals = len(body) // 8
    assert n_vals in (legacy_f64s, full_f64s), (
        f"unexpected blob size {len(body)} for declared h={h}"
    )
    vals = list(struct.unpack(f"<{n_vals}d", body))
    if n_vals == legacy_f64s:
        raise ValueError(f"{bin_path} predates cat_heat (legacy blob) -- nothing to keep")
    return vals[-FIELD_LEN:], h


def load_full_state(bin_path: Path) -> tuple[dict, int]:
    """Load every field (cat_heat zero-padded if the blob predates it)."""
    raw = bin_path.read_bytes()
    (h,) = struct.unpack("<Q", raw[:H_HEADER_LEN])
    body = raw[H_HEADER_LEN:]
    full_f64s = _payload_f64s(h)
    legacy_f64s = full_f64s - FIELD_LEN
    n_vals = len(body) // 8
    assert n_vals in (legacy_f64s, full_f64s), (
        f"unexpected blob size {len(body)} for declared h={h}"
    )
    vals = list(struct.unpack(f"<{n_vals}d", body))
    if n_vals == legacy_f64s:
        vals = vals + [0.0] * FIELD_LEN
    o = 0

    def take(n):
        nonlocal o
        s = vals[o:o + n]
        o += n
        return s

    return {
        "ws": take(WSKIP_LEN), "b1": take(h), "w2": take(h),
        "w1c": take(9 * 128 * h), "po": take(81 * h), "px": take(81 * h),
        "route_me": take(FIELD_LEN), "route_opp": take(FIELD_LEN),
        "route_near_me": take(FIELD_LEN), "route_near_opp": take(FIELD_LEN),
        "route_contested": take(FIELD_LEN), "cat_heat": take(FIELD_LEN),
    }, h


def build_fresh_net(
    new_h: int, cat_heat: list[float], *, seed: int, base: dict | None = None, base_h: int | None = None
) -> dict:
    """Build the h=new_h net. Fields whose shape matches `base` at `base_h`
    are transplanted verbatim from base (real trained values, e.g. ACE v13);
    everything else falls back to small random noise. cat_heat is always the
    caller-supplied trained field, regardless of base."""
    g = torch.Generator().manual_seed(seed)

    def randn(*shape, std):
        return (torch.randn(*shape, generator=g) * std).tolist()

    use_base = base is not None and base_h == new_h

    state = {
        "ws": list(base["ws"]) if use_base else randn(WSKIP_LEN, std=_INIT_STD["ws"]),
        "b1": list(base["b1"]) if use_base else randn(new_h, std=_INIT_STD["b1"]),
        "w2": list(base["w2"]) if use_base else randn(new_h, std=_INIT_STD["w2"]),
        "w1c": list(base["w1c"]) if use_base else randn(9 * 128 * new_h, std=_INIT_STD["w1c"]),
        "po": list(base["po"]) if use_base else randn(81 * new_h, std=_INIT_STD["po"]),
        "px": list(base["px"]) if use_base else randn(81 * new_h, std=_INIT_STD["px"]),
        "route_me": list(base["route_me"]) if use_base else randn(FIELD_LEN, std=_INIT_STD["route_me"]),
        "route_opp": list(base["route_opp"]) if use_base else randn(FIELD_LEN, std=_INIT_STD["route_opp"]),
        "route_near_me": list(base["route_near_me"]) if use_base else randn(FIELD_LEN, std=_INIT_STD["route_near_me"]),
        "route_near_opp": list(base["route_near_opp"]) if use_base else randn(FIELD_LEN, std=_INIT_STD["route_near_opp"]),
        "route_contested": list(base["route_contested"]) if use_base else randn(FIELD_LEN, std=_INIT_STD["route_contested"]),
        "cat_heat": list(cat_heat),
    }
    assert len(state["cat_heat"]) == FIELD_LEN
    return state


def export_raw_bin(state: dict, new_h: int, out_path: Path) -> None:
    vals = []
    for key in _RAW_ORDER:
        vals.extend(state[key])
    expected = _payload_f64s(new_h)
    assert len(vals) == expected, f"payload len {len(vals)} != expected {expected}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(struct.pack("<Q", new_h) + struct.pack(f"<{len(vals)}d", *vals))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, help="net blob to take cat_heat from")
    ap.add_argument("--out", required=True, help="output net_weights.bin-format blob")
    ap.add_argument(
        "--new-h", type=int, default=None,
        help="hidden width for the output net; default keeps the source's own "
             "width unchanged (this tool re-randomizes weight values, it does "
             "not resize the network -- use net2net_widen.py for that)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--base", default=None,
        help="net blob to transplant real trained weights from wherever its "
             "shape matches --new-h (e.g. ACE v13 frozen weights at h=32); "
             "fields that don't match fall back to random init",
    )
    args = ap.parse_args()

    source = Path(args.source)
    out = Path(args.out)
    cat_heat, source_h = load_cat_heat(source)
    new_h = args.new_h if args.new_h is not None else source_h

    base_state, base_h = (None, None)
    if args.base is not None:
        base_state, base_h = load_full_state(Path(args.base))
        if base_h != new_h:
            print(f"WARNING: --base h={base_h} != target h={new_h}; base fields ignored, using random init")

    state = build_fresh_net(new_h, cat_heat, seed=args.seed, base=base_state, base_h=base_h)
    export_raw_bin(state, new_h, out)
    args.new_h = new_h  # for the round-trip check below

    # Round-trip sanity: re-parse what we just wrote, confirm shape + cat_heat
    # exact match, before this is ever handed to a trainer/bootstrap step.
    raw = out.read_bytes()
    (h_check,) = struct.unpack("<Q", raw[:H_HEADER_LEN])
    assert h_check == args.new_h
    body = raw[H_HEADER_LEN:]
    n_vals = len(body) // 8
    assert n_vals == _payload_f64s(args.new_h)
    vals = list(struct.unpack(f"<{n_vals}d", body))
    round_trip_cat_heat = vals[-FIELD_LEN:]
    assert round_trip_cat_heat == cat_heat, "cat_heat corrupted on round-trip"

    print(f"wrote {out} h={args.new_h} bytes={len(raw)}")
    print(f"cat_heat preserved exactly ({FIELD_LEN} floats) from {source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
