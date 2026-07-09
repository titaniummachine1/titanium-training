#!/usr/bin/env python3
"""Export and preserve cycle-38 repair candidate from a trainer checkpoint."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_TRAINING = Path(__file__).resolve().parents[1]
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

from streaming_checkpoint_chain import (
    PREVIOUS_WEIGHTS,
    RUN_DIR,
    atomic_copy2,
    resolve_latest_accepted_weights,
    sha256_file,
)
from training_coordinator import VALIDATION_BLOCKED_DIR, utc_now, write_json

REPO = _TRAINING.parent
sys.path.insert(0, str(REPO / "training" / "titanium_training"))
from titanium_training.training.trainer import HalfPW  # noqa: E402


def export_candidate(
    *,
    checkpoint: Path,
    architecture_bin: Path,
    out_bin: Path,
) -> dict:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = HalfPW(architecture_bin)
    model.load_state_dict(payload["model"])
    out_bin.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(out_bin)
    return {
        "checkpoint": str(checkpoint),
        "architecture_bin": str(architecture_bin),
        "candidate_bin": str(out_bin),
        "sha256": sha256_file(out_bin),
        "hidden_size": model.h,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=RUN_DIR / "ckpt_epoch0001.pt",
    )
    ap.add_argument(
        "--architecture-bin",
        type=Path,
        default=None,
        help="weights.bin with matching NET_H header (default: latest accepted)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=VALIDATION_BLOCKED_DIR,
    )
    args = ap.parse_args()

    arch = args.architecture_bin
    if arch is None:
        arch = resolve_latest_accepted_weights()
    out_bin = args.out_dir / "cycle_0038_candidate.bin"
    info = export_candidate(
        checkpoint=args.checkpoint,
        architecture_bin=arch,
        out_bin=out_bin,
    )
    if args.checkpoint.is_file():
        atomic_copy2(args.checkpoint, args.out_dir / args.checkpoint.name)
    if PREVIOUS_WEIGHTS.is_file():
        atomic_copy2(PREVIOUS_WEIGHTS, args.out_dir / "net_weights_previous.bin")
    for rel in ("epoch_diagnostics_0001.json", "epoch_weight_diagnostics_0001.json"):
        src = RUN_DIR / rel
        if src.is_file():
            atomic_copy2(src, args.out_dir / rel.name)
    manifest = {
        "preserved_at": utc_now(),
        "source": "manual_export_from_checkpoint",
        **info,
    }
    write_json(args.out_dir / "BLOCKED.json", manifest)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
