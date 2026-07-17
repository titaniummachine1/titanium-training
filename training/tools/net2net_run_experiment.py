#!/usr/bin/env python3
"""Isolated training loop for the Net2Net-widened (NET_H=48) experiment.

Repeatedly calls trainer.py streaming training, each cycle initialized from
the previous cycle's exported net_weights_best.bin -- same chaining pattern
training_coordinator.py uses for the production chain, but entirely isolated
under runs/v16/net2net_experiment/ and NEVER touching the production chain,
BEST_WEIGHTS, or ENGINE_WEIGHTS. Logs loss per cycle to a JSONL file so
progress can be inspected without re-parsing subprocess stdout each time.

Retains history (first run overwrote the same filename every cycle, so the
best-by-val_loss point from that run -- cycle 45 -- was unrecoverable once
training overfit past it): every cycle's weights are snapshotted, the
best-by-val_loss snapshot is kept separately and never overwritten by a worse
cycle, and training stops automatically once val_loss hasn't improved for
`--patience` cycles instead of blindly running a fixed count into overfitting.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

_TRAINING = Path(__file__).resolve().parents[1]
_REPO = _TRAINING.parent
EXP_DIR = _TRAINING / "runs" / "v16" / "net2net_experiment"
RUN_DIR = EXP_DIR / "run"
SNAP_DIR = EXP_DIR / "snapshots"
BEST_VAL_BIN = EXP_DIR / "best_val.bin"
BEST_VAL_META = EXP_DIR / "best_val_meta.json"
LOG_PATH = _TRAINING / "data" / "overnight_logs" / "net2net_h48_cycles.jsonl"
PY312 = r"C:\Users\Terminatort8000\AppData\Local\Programs\Python\Python312\python.exe"
LABELS_DB = _TRAINING / "data" / "canonical" / "labels.db"


def _atomic_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def run_cycle(cycle: int, weights_bin: Path) -> dict:
    cmd = [
        PY312,
        str(_TRAINING / "titanium_training" / "training" / "trainer.py"),
        "--labels-db", str(LABELS_DB),
        "--weights", str(weights_bin),
        "--out-dir", str(RUN_DIR),
        "--epochs", "1",
        "--batch", "512",
        "--lr", "0.001",
        "--weight-decay", "0.00001",
        "--stream-old-refresh-fraction", "0.05",
        "--stream-retired-replay-fraction", "0.05",
        "--val-split", "0.05",
        "--checkpoint-steps", "999999",
        "--patience", "0",
        "--cpu",
        "--no-parity",
        "--defer-usage-commit",
        "--log-every", "200",
        "--log-interval-sec", "30",
    ]
    env = {**os.environ, "TITANIUM_NET_H": "48", "RUSTFLAGS": "-C target-cpu=native"}
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(_REPO), capture_output=True, text=True, env=env, timeout=1800)
    elapsed = time.perf_counter() - t0
    diag_path = RUN_DIR / "epoch_diagnostics_0001.json"
    diag = json.loads(diag_path.read_text()) if diag_path.is_file() else {}
    return {
        "cycle": cycle,
        "returncode": proc.returncode,
        "elapsed_sec": round(elapsed, 1),
        "stdout_tail": proc.stdout[-1500:],
        "stderr_tail": proc.stderr[-1500:] if proc.returncode != 0 else "",
        "diag": diag,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("n_cycles", type=int, nargs="?", default=300)
    ap.add_argument("--patience", type=int, default=25,
                     help="stop after this many cycles with no val_loss improvement")
    ap.add_argument("--snapshot-every", type=int, default=5)
    ap.add_argument("--init-weights", default=None,
                     help="resume from a specific weights file instead of widened_h48_init.bin")
    args = ap.parse_args()

    weights_bin = Path(args.init_weights) if args.init_weights else EXP_DIR / "widened_h48_init.bin"
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    best_cycle = 0
    if BEST_VAL_META.is_file():
        meta = json.loads(BEST_VAL_META.read_text())
        best_val = meta.get("val_loss", float("inf"))
        best_cycle = meta.get("cycle", 0)
        print(f"resuming: current best val_loss={best_val} at cycle={best_cycle}", flush=True)

    no_improve = 0
    with LOG_PATH.open("a", encoding="utf-8") as logf:
        for cycle in range(1, args.n_cycles + 1):
            result = run_cycle(cycle, weights_bin)
            logf.write(json.dumps(result) + "\n")
            logf.flush()
            if result["returncode"] != 0:
                print(f"cycle {cycle} FAILED rc={result['returncode']}: {result['stderr_tail']}", flush=True)
                return 1
            candidate = RUN_DIR / "net_weights_best.bin"
            if not candidate.is_file():
                print(f"cycle {cycle}: no candidate exported, stopping", flush=True)
                return 1
            weights_bin = candidate
            diag = result.get("diag") or {}
            val_loss = diag.get("val_loss")
            train_end = diag.get("train_loss_end")

            if cycle % args.snapshot_every == 0:
                _atomic_copy(candidate, SNAP_DIR / f"cycle_{cycle:04d}.bin")

            improved = val_loss is not None and val_loss < best_val
            if improved:
                _atomic_copy(candidate, BEST_VAL_BIN)
                BEST_VAL_META.write_text(json.dumps({"cycle": cycle, "val_loss": val_loss}, indent=2))
                best_val = val_loss
                best_cycle = cycle
                no_improve = 0
            else:
                no_improve += 1

            print(
                f"cycle {cycle}/{args.n_cycles} elapsed={result['elapsed_sec']}s "
                f"train_loss_end={train_end} val_loss={val_loss} "
                f"best={best_val:.4f}@{best_cycle} no_improve={no_improve}",
                flush=True,
            )

            if no_improve >= args.patience:
                print(
                    f"EARLY STOP at cycle {cycle}: no val_loss improvement for {args.patience} cycles "
                    f"(best={best_val:.4f} at cycle {best_cycle}, retained at {BEST_VAL_BIN})",
                    flush=True,
                )
                return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
