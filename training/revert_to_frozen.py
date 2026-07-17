#!/usr/bin/env python3
"""Restore live + training best weights from net_weights_frozen.bin (never modifies frozen)."""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

_TRAINING = Path(__file__).resolve().parent
REPO = _TRAINING.parent
FROZEN = REPO / "engine" / "src" / "titanium" / "net_weights_frozen.bin"
LIVE = REPO / "engine" / "src" / "titanium" / "net_weights.bin"
RUN_DIR = REPO / "training" / "runs" / "value_oracle"
BEST = RUN_DIR / "net_weights_best.bin"
MARKER = RUN_DIR / "RESTORED_CHECKPOINT.txt"


def revert(*, backup: bool = True, archive_ckpts: bool = False) -> None:
    if not FROZEN.is_file():
        raise SystemExit(f"missing frozen weights: {FROZEN}")

    frozen_hash = hashlib.sha256(FROZEN.read_bytes()).hexdigest()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    if backup:
        for src, label in ((LIVE, "live"), (BEST, "best")):
            if src.is_file() and src.read_bytes() != FROZEN.read_bytes():
                dest = RUN_DIR / f"net_weights_corrupted_{label}_{stamp}.bin"
                shutil.copy2(src, dest)
                print(f"Backed up corrupted {label} -> {dest.name}")

    shutil.copy2(FROZEN, LIVE)
    shutil.copy2(FROZEN, BEST)
    print(f"Restored frozen -> {LIVE.relative_to(REPO)}")
    print(f"Restored frozen -> {BEST.relative_to(REPO)}")
    print(f"  sha256={frozen_hash[:16]}…  ({FROZEN.stat().st_size} bytes)")

    if hashlib.sha256(FROZEN.read_bytes()).hexdigest() != frozen_hash:
        raise RuntimeError(f"REFUSING: frozen weights changed during restore at {FROZEN}")

    MARKER.write_text(
        f"restored_from={FROZEN}\n"
        f"restored_at={stamp}\n"
        f"note=live and best aligned to frozen baseline; do not resume corrupted ckpts\n",
        encoding="utf-8",
    )

    if archive_ckpts:
        ckpts = sorted(RUN_DIR.glob("ckpt_epoch*.pt")) + sorted(RUN_DIR.glob("ckpt_step*.pt"))
        if ckpts:
            dest_dir = RUN_DIR / f"corrupted_ckpts_{stamp}"
            dest_dir.mkdir(parents=True, exist_ok=True)
            for ckpt in ckpts:
                shutil.move(str(ckpt), dest_dir / ckpt.name)
            print(f"Archived {len(ckpts)} checkpoints -> {dest_dir.relative_to(REPO)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-backup", action="store_true", help="Skip backup of current live/best")
    ap.add_argument(
        "--archive-ckpts",
        action="store_true",
        help="Move ckpt_epoch*.pt / ckpt_step*.pt out of value_oracle run dir",
    )
    args = ap.parse_args()
    revert(backup=not args.no_backup, archive_ckpts=args.archive_ckpts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
