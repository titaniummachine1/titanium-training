#!/usr/bin/env python3
"""Prove full-state resume from accepted epoch_0002.pt without discarding Adam/EMA."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

_TRAINING = Path(__file__).resolve().parent
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

from titanium_training.training.trainer import (
    TRAINING_SCHEMA,
    HalfPW,
    build_optimizer,
    load_checkpoint,
)


def main() -> int:
    ckpt_path = Path(r"C:\gitProjects\Quoridor best AI\training\runs\v16\accepted\epoch_0002.pt")
    weights = Path(r"C:\gitProjects\Quoridor best AI\training\runs\v16\accepted\epoch_0002.bin")
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    expected = {
        "schema": TRAINING_SCHEMA,
        "step": 223,
        "epoch": 1,
    }
    for key, want in expected.items():
        got = raw.get(key)
        if got != want:
            print(f"FAIL: {key}={got!r} want={want!r}")
            return 1

    model = HalfPW(weights)
    optimizer = build_optimizer(model, kind="adam", lr=2e-4, weight_decay=1e-5)
    step, epoch, best_val, optimizer, ema_state = load_checkpoint(
        ckpt_path, model, optimizer, weights_path=weights
    )

    if step != 223 or epoch != 1:
        print(f"FAIL: resumed step/epoch = {step}/{epoch}")
        return 1
    if abs(float(best_val) - float(raw["best_val"])) > 1e-12:
        print(f"FAIL: best_val {best_val} != {raw['best_val']}")
        return 1
    if ema_state is None or len(ema_state) != 16:
        print(f"FAIL: ema_state entries={None if ema_state is None else len(ema_state)}")
        return 1
    if len(raw["model"]) != 16:
        print(f"FAIL: model tensors={len(raw['model'])}")
        return 1

    opt_state = optimizer.state_dict()["state"]
    if len(opt_state) != 16:
        print(f"FAIL: adam state entries={len(opt_state)}")
        return 1

    # Exact tensor identity vs raw checkpoint payloads.
    for name, tensor in model.state_dict().items():
        if not torch.equal(tensor.cpu(), raw["model"][name].cpu()):
            print(f"FAIL: model tensor drift: {name}")
            return 1
    for name, tensor in ema_state.items():
        if not torch.equal(tensor.cpu(), raw["ema_state"][name].cpu()):
            print(f"FAIL: ema tensor drift: {name}")
            return 1

    report = {
        "ok": True,
        "ckpt": str(ckpt_path),
        "schema": raw["schema"],
        "step": step,
        "epoch": epoch,
        "best_val": float(best_val),
        "model_tensors": len(raw["model"]),
        "adam_state_entries": len(opt_state),
        "ema_tensors": len(ema_state),
    }
    print(json.dumps(report, indent=2))
    print("PASS: full-state resume identity from epoch_0002.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
