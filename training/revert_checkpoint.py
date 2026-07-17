#!/usr/bin/env python3
"""Export a checkpoint to engine net_weights.bin and value_oracle best weights."""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch

_TRAINING = Path(__file__).resolve().parent
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

from titanium_training.training.trainer import HalfPW, WEIGHTS, TRAINING_SCHEMA

REPO = _TRAINING.parent
RUN_DIR = REPO / "training" / "runs" / "value_oracle"
ENGINE_WEIGHTS = REPO / "engine" / "src" / "titanium" / "net_weights.bin"
FROZEN_WEIGHTS = REPO / "engine" / "src" / "titanium" / "net_weights_frozen.bin"


def export_checkpoint(ckpt_path: Path, *, deploy_engine: bool = True) -> Path:
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    schema = payload.get("schema")
    if schema and schema != TRAINING_SCHEMA:
        print(f"WARN: checkpoint schema {schema!r} != {TRAINING_SCHEMA!r}")

    model = HalfPW(WEIGHTS)
    model.load_state_dict(payload["model"])

    out_best = RUN_DIR / "net_weights_best.bin"
    out_prev = RUN_DIR / "net_weights_previous.bin"
    if out_best.is_file():
        shutil.copy2(out_best, out_prev)

    model.save_weights(out_best)
    print(f"Exported -> {out_best}")
    print(f"  epoch={payload.get('epoch')} step={payload.get('step')} best_val={payload.get('best_val')}")

    if deploy_engine:
        frozen_before = FROZEN_WEIGHTS.read_bytes() if FROZEN_WEIGHTS.is_file() else None
        shutil.copy2(out_best, ENGINE_WEIGHTS)
        print(f"Deployed -> {ENGINE_WEIGHTS} (live / v15 / website only)")
        if frozen_before is not None and FROZEN_WEIGHTS.read_bytes() != frozen_before:
            raise RuntimeError(
                f"REFUSING: deploy touched frozen weights at {FROZEN_WEIGHTS}"
            )

    marker = RUN_DIR / "RESTORED_CHECKPOINT.txt"
    marker.write_text(f"restored_from={ckpt_path}\n", encoding="utf-8")
    return out_best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--ckpt",
        default=str(RUN_DIR / "ckpt_epoch0001.pt"),
        help="Checkpoint to restore (default: epoch 1)",
    )
    ap.add_argument("--no-engine", action="store_true", help="Do not copy to engine/src/titanium/net_weights.bin")
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    if not ckpt.is_file():
        print(f"ERROR: checkpoint missing: {ckpt}")
        return 1
    export_checkpoint(ckpt, deploy_engine=not args.no_engine)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
