#!/usr/bin/env python3
"""Run post-train validation + strength gate on a preserved candidate."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_TRAINING = Path(__file__).resolve().parents[1]
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

from streaming_checkpoint_chain import PREVIOUS_WEIGHTS, sha256_file
from streaming_epoch_validation import run_epoch_validation


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--candidate",
        type=Path,
        default=_TRAINING / "runs" / "quarantine" / "cycle_0038_validation_blocked" / "cycle_0038_candidate.bin",
    )
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=_TRAINING / "runs" / "quarantine" / "cycle_0038_validation_blocked" / "ckpt_epoch0001.pt",
    )
    ap.add_argument(
        "--parent",
        type=Path,
        default=PREVIOUS_WEIGHTS,
    )
    args = ap.parse_args()
    ckpt = args.checkpoint if args.checkpoint.is_file() else args.candidate
    report = run_epoch_validation(
        checkpoint=ckpt,
        candidate_bin=args.candidate,
        previous_bin=args.parent if args.parent.is_file() else None,
    )
    out = {
        "candidate": str(args.candidate),
        "candidate_sha256": sha256_file(args.candidate),
        "parent_sha256": sha256_file(args.parent) if args.parent.is_file() else None,
        "validation": report,
        "passed": bool(report.get("passed")),
    }
    print(json.dumps(out, indent=2))
    return 0 if out["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
