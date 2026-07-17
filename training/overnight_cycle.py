#!/usr/bin/env python3
"""One overnight cycle: 1 epoch train -> deploy -> self-play -> saturation check.

Uses base teacher_dataset feature cache (not extended). Position usage retires
rows after 5 epoch touches.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_TRAINING = _REPO / "training"
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

CACHE_DIR = _TRAINING / "data" / "feature_cache"
RUN_DIR = _TRAINING / "runs" / "value_oracle"
LOG_DIR = _TRAINING / "data" / "overnight_logs"


def log(msg: str) -> None:
    print(msg, flush=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with (LOG_DIR / "cycle.log").open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def run(cmd: list[str]) -> int:
    log(f"$ {' '.join(cmd)}")
    env = {**dict(__import__("os").environ), "PYTHONPATH": str(_TRAINING)}
    return subprocess.call(cmd, cwd=str(_REPO), env=env)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--games", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--time", type=float, default=4.0)
    ap.add_argument("--revert-epoch1", action="store_true", help="Restore ckpt_epoch0001 before training")
    ap.add_argument(
        "--from-frozen",
        action="store_true",
        help="Restore live+best from net_weights_frozen.bin and train fresh (no resume)",
    )
    ap.add_argument(
        "--archive-corrupted-ckpts",
        action="store_true",
        help="With --from-frozen: move existing ckpt_*.pt into archived subdir",
    )
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-selfplay", action="store_true")
    args = ap.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log(f"\n=== overnight cycle {stamp} ===")

    if args.from_frozen:
        revert_cmd = [sys.executable, str(_TRAINING / "revert_to_frozen.py")]
        if args.archive_corrupted_ckpts:
            revert_cmd.append("--archive-ckpts")
        rc = run(revert_cmd)
        if rc != 0:
            return rc
    elif args.revert_epoch1:
        rc = run([sys.executable, str(_TRAINING / "revert_checkpoint.py"), "--ckpt", str(RUN_DIR / "ckpt_epoch0001.pt")])
        if rc != 0:
            return rc

    if not args.skip_train:
        trainer = _TRAINING / "titanium_training" / "training" / "trainer.py"
        train_cmd = [
            sys.executable, str(trainer),
            "--cache-dir", str(CACHE_DIR),
            "--out-dir", str(RUN_DIR),
            "--epochs", str(args.epochs),
            "--batch", "512",
            "--lr", "0.0005",
            "--checkpoint-steps", "999999",
            "--val-split", "0.05",
            "--patience", "0",
            "--cpu",
        ]
        if not args.from_frozen:
            ckpts = sorted(RUN_DIR.glob("ckpt_epoch*.pt"))
            ckpt = ckpts[-1] if ckpts else RUN_DIR / "ckpt_epoch0001.pt"
            train_cmd.extend(["--resume", "--ckpt", str(ckpt)])
        rc = run(train_cmd)
        if rc != 0:
            log(f"TRAIN FAILED rc={rc}")
            return rc

        from revert_checkpoint import export_checkpoint

        latest = sorted(RUN_DIR.glob("ckpt_epoch*.pt"))[-1]
        export_checkpoint(latest, deploy_engine=True)

    if not args.skip_selfplay:
        rc = run([
            sys.executable, str(_TRAINING / "self_play_overnight.py"),
            "--games", str(args.games),
            "--threads", str(args.threads),
            "--time", str(args.time),
        ])
        if rc != 0:
            return rc

        report_path = _TRAINING / "data" / "overnight_selfplay_last.json"
        if report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report.get("saturated"):
                log("SATURATION: current net losing to previous — stop training loop.")
                (RUN_DIR / "SATURATED.txt").write_text(json.dumps(report, indent=2), encoding="utf-8")
                return 2

    from position_usage import status as usage_status
    if CACHE_DIR.is_dir():
        log(f"Usage status: {usage_status(CACHE_DIR)}")

    log("Cycle complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
