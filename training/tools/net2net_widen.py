#!/usr/bin/env python3
"""Net2Net-style function-preserving widening of the HalfPW hidden layer.

Standard Net2WiderNet trick (Chen et al. 2016): pick a subset of existing
hidden units (without replacement, so each source unit is duplicated at most
once), copy their full incoming weights (b1, w1c, po, px) to new unit slots,
and HALVE the outgoing weight (w2) of both the original and its duplicate so
the two together sum to the original contribution. Net effect: output is
unchanged (aside from a tiny symmetry-breaking noise term) immediately after
widening, but the new units have real nonzero gradient from step one (unlike
naively zero-initializing new units' outgoing weights, which would silently
freeze them forever since their gradient is the outgoing weight itself).

ws / route_* / cat_heat are independent of hidden width (81-cell field maps,
not NET_H-shaped) and are copied unchanged.
"""
from __future__ import annotations

import argparse
import copy
import struct
import sys
from pathlib import Path

import torch

_TRAINING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TRAINING))

# Field order matches HalfPW.__init__ / halfpw.py Net.load() exactly.
_RAW_ORDER = [
    "ws", "b1", "w2", "w1c", "po", "px",
    "route_me", "route_opp", "route_near_me", "route_near_opp",
    "route_contested", "cat_heat",
]


def export_raw_bin(state: dict, out_path: Path) -> None:
    """Serialize a widened state_dict to the same little-endian f64 blob
    layout `HalfPW.__init__`/`halfpw.py Net.load()` read, so it can be passed
    directly as `trainer.py --weights` (bootstrap init, no --resume needed --
    the produced checkpoint's optimizer is fresh anyway)."""
    h = state["b1"].shape[0]
    vals = []
    for key in _RAW_ORDER:
        vals.extend(state[key].reshape(-1).tolist())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(struct.pack("<Q", h) + struct.pack(f"<{len(vals)}d", *vals))


def widen_state_dict(state: dict, old_h: int, new_h: int, *, seed: int, noise_std: float = 1e-3) -> dict:
    assert old_h < new_h <= 2 * old_h, (
        f"widen requires old_h < new_h <= 2*old_h (no-replacement duplication); "
        f"got old_h={old_h} new_h={new_h}"
    )
    grow = new_h - old_h
    g = torch.Generator().manual_seed(seed)
    dup_idx = torch.randperm(old_h, generator=g)[:grow]

    out = copy.deepcopy(state)

    b1_old, w2_old = state["b1"], state["w2"]
    w1c_old, po_old, px_old = state["w1c"], state["po"], state["px"]

    b1_new = torch.zeros(new_h, dtype=b1_old.dtype)
    w2_new = torch.zeros(new_h, dtype=w2_old.dtype)
    w1c_new = torch.zeros((w1c_old.shape[0], w1c_old.shape[1], new_h), dtype=w1c_old.dtype)
    po_new = torch.zeros((po_old.shape[0], new_h), dtype=po_old.dtype)
    px_new = torch.zeros((px_old.shape[0], new_h), dtype=px_old.dtype)

    b1_new[:old_h] = b1_old
    w1c_new[:, :, :old_h] = w1c_old
    po_new[:, :old_h] = po_old
    px_new[:, :old_h] = px_old

    w2_new[:old_h] = w2_old
    for k, i in enumerate(dup_idx.tolist()):
        n = old_h + k
        b1_new[n] = b1_old[i]
        w1c_new[:, :, n] = w1c_old[:, :, i] + torch.randn(w1c_old.shape[:2], generator=g) * noise_std
        po_new[:, n] = po_old[:, i] + torch.randn(po_old.shape[0], generator=g) * noise_std
        px_new[:, n] = px_old[:, i] + torch.randn(px_old.shape[0], generator=g) * noise_std
        # Split the original unit's outgoing weight between the original and
        # its duplicate so their summed contribution is unchanged.
        w2_new[i] = w2_old[i] / 2.0
        w2_new[n] = w2_old[i] / 2.0

    out["b1"] = b1_new
    out["w2"] = w2_new
    out["w1c"] = w1c_new
    out["po"] = po_new
    out["px"] = px_new
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--in-ckpt", type=Path, help="trainer.py torch checkpoint (has ckpt['model'])")
    src.add_argument("--in-bin", type=Path,
                      help="raw net_weights.bin-format blob (e.g. an accepted chain snapshot); "
                           "--old-h is inferred from its own header and need not be passed")
    ap.add_argument("--out-ckpt", type=Path, default=None,
                     help="widened torch checkpoint (only written if given)")
    ap.add_argument("--out-bin", type=Path, default=None,
                     help="also export a raw net_weights.bin-format blob for --weights bootstrap")
    ap.add_argument("--old-h", type=int, default=32)
    ap.add_argument("--new-h", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    schema = None
    if args.in_bin is not None:
        from titanium_training.training.trainer import HalfPW

        model = HalfPW(str(args.in_bin))
        state = model.state_dict()
        old_h = model.h
        in_desc = str(args.in_bin)
    else:
        ckpt = torch.load(args.in_ckpt, weights_only=False)
        state = ckpt["model"]
        old_h = args.old_h
        schema = ckpt.get("schema")
        in_desc = str(args.in_ckpt)

    widened_model = widen_state_dict(state, old_h, args.new_h, seed=args.seed)

    print(f"Widened {old_h} -> {args.new_h}: {in_desc}")
    if args.out_ckpt is not None:
        out_ckpt = {
            "schema": schema,
            "step": 0,
            "epoch": 0,
            "best_val": None,
            "model": widened_model,
            "optimizer": None,  # fresh optimizer state — widened shapes don't match old state
        }
        args.out_ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(out_ckpt, args.out_ckpt)
        print(f"  -> {args.out_ckpt}")
    if args.out_bin is not None:
        export_raw_bin(widened_model, args.out_bin)
        print(f"  raw bootstrap blob -> {args.out_bin}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
