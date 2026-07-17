#!/usr/bin/env python3
"""Export EMA (or raw model) HalfPW weights from a trainer .pt checkpoint.

Writes a temporary engine-format .bin without touching accepted/ or net_weights.bin.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

_TRAINING = Path(__file__).resolve().parent
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

from titanium_training.training.trainer import HalfPW, save_export_weights


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--arch-bin", required=True, type=Path, help="Architecture/schema sibling .bin")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--source", choices=("ema", "model"), default="ema")
    args = ap.parse_args()

    raw = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = HalfPW(args.arch_bin)
    if args.source == "ema":
        ema = raw.get("ema_state")
        if not ema:
            print("FAIL: checkpoint has no ema_state", file=sys.stderr)
            return 1
        save_export_weights(model, args.out, ema)
        n_tensors = len(ema)
    else:
        model.load_state_dict(raw["model"])
        model.save_weights(args.out)
        n_tensors = len(raw["model"])

    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    meta = {
        "ok": True,
        "ckpt": str(args.ckpt.resolve()),
        "arch_bin": str(args.arch_bin.resolve()),
        "out": str(args.out.resolve()),
        "source": args.source,
        "tensors": n_tensors,
        "step": raw.get("step"),
        "epoch": raw.get("epoch"),
        "best_val": raw.get("best_val"),
        "schema": raw.get("schema"),
        "sha256": digest,
        "size": args.out.stat().st_size,
    }
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
