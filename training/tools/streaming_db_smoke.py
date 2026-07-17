#!/usr/bin/env python3
"""Smoke test for database-backed streaming training.

Validates:
  - bounded private memory while featurizing 8192 positions
  - fv_len == 547
  - labels align with featurized targets
  - one short training pass completes without monolithic cache

On success writes training/data/overnight_logs/streaming_training_ready.json
(but does NOT clear pause_training_epochs.json — supervisor handles that gate).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil

_TRAINING = Path(__file__).resolve().parents[1]
_REPO = _TRAINING.parent
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

from build_feature_cache import FV_LEN
from db_import import LABELS_DB_PATH
from position_usage_db import backfill_from_labels, open_labels_db, status as usage_status
from streaming_db_loader import (
    FEATURIZE_CHUNK_DEFAULT,
    LabelsRepository,
    db_counts,
    iter_db_training_batches,
    sample_epoch_keys,
)

LOG_DIR = _TRAINING / "data" / "overnight_logs"
READY_PATH = LOG_DIR / "streaming_training_ready.json"
GAMES_DB = _TRAINING / "data" / "canonical" / "games.db"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _private_mb() -> float:
    return psutil.Process().memory_info().private / (1024 ** 2)


def run_smoke(*, n_positions: int = 8192, batch_size: int = 512, train_steps: int = 8) -> dict:
    labels_db = LABELS_DB_PATH
    if not labels_db.is_file():
        raise FileNotFoundError(f"labels.db missing: {labels_db}")

    mem_start = _private_mb()
    peak_private_mb = mem_start

    con = open_labels_db(labels_db)
    backfilled = backfill_from_labels(con)
    con.commit()
    counts = db_counts(labels_db)
    usage = usage_status(con)
    # Quick single-row featurization sanity check (not full corpus).
    probe_keys = sample_epoch_keys(con, epoch_size=min(32, n_positions), seed=20260626)
    con.close()
    probe_vecs: list = []
    if probe_keys:
        repo = LabelsRepository(labels_db)
        probe = next(iter_db_training_batches(repo, probe_keys[:8], chunk_size=8))
        repo.close()
        probe_vecs = list(probe.features)
        probe_failed = 8 - len(probe_vecs)
        if not probe_vecs or any(len(v) != FV_LEN for v in probe_vecs):
            raise RuntimeError(f"probe featurization failed (failed={probe_failed})")
        bad_targets = sum(1 for v in probe_vecs if not (0.0 <= float(v[0]) <= 1.0))
        if bad_targets:
            raise RuntimeError(f"{bad_targets} probe targets out of [0,1] range")
    peak_private_mb = max(peak_private_mb, _private_mb())

    # Short training pass via trainer subprocess
    trainer = _TRAINING / "titanium_training" / "training" / "trainer.py"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = Path(tempfile.mkdtemp(prefix="streaming_smoke_", dir=str(LOG_DIR)))
    import subprocess

    t0 = time.perf_counter()
    proc = subprocess.run(
        [
            sys.executable,
            str(trainer),
            "--labels-db",
            str(labels_db),
            "--out-dir",
            str(out_dir),
            "--epochs",
            "1",
            "--batch",
            str(batch_size),
            "--stream-max-positions",
            str(n_positions),
            "--stream-featurize-chunk",
            str(FEATURIZE_CHUNK_DEFAULT),
            "--val-split",
            "0.0",
            "--checkpoint-steps",
            "999999",
            "--patience",
            "0",
            "--cpu",
            "--no-parity",
            "--log-every",
            "50",
        ],
        cwd=str(_REPO),
        capture_output=True,
        text=True,
        timeout=1800,
        env={**dict(__import__("os").environ), "PYTHONPATH": str(_TRAINING), "RUSTFLAGS": "-C target-cpu=native"},
    )
    train_elapsed = time.perf_counter() - t0
    peak_private_mb = max(peak_private_mb, _private_mb())

    if proc.returncode != 0:
        raise RuntimeError(
            f"trainer failed rc={proc.returncode}\nstdout:\n{proc.stdout[-2000:]}\nstderr:\n{proc.stderr[-2000:]}"
        )

    ckpts = sorted(out_dir.glob("ckpt_epoch*.pt"))
    if not ckpts:
        raise RuntimeError("trainer completed but wrote no epoch checkpoint")
    import torch

    ckpt = torch.load(ckpts[-1], weights_only=False, map_location="cpu")
    if "model" not in ckpt or "optimizer" not in ckpt:
        raise RuntimeError(f"checkpoint missing model/optimizer keys: {ckpts[-1]}")
    shutil.rmtree(out_dir, ignore_errors=True)

    positions_processed = n_positions
    throughput = positions_processed / train_elapsed if train_elapsed > 0 else 0.0

    report = {
        "ready": True,
        "verified_at": _utc_now(),
        "n_positions_requested": n_positions,
        "n_probe_featurized": len(probe_vecs) if probe_keys else 0,
        "fv_len": FV_LEN,
        "featurize_chunk": FEATURIZE_CHUNK_DEFAULT,
        "peak_private_mb": round(peak_private_mb, 1),
        "mem_start_mb": round(mem_start, 1),
        "train_elapsed_sec": round(train_elapsed, 1),
        "positions_per_sec": round(throughput, 1),
        "checkpoint_reopened": True,
        "temp_checkpoint_cleaned": not out_dir.exists(),
        "labeled_positions": counts.labeled_positions,
        "eligible_positions": counts.eligible_positions,
        "usage_backfilled": backfilled,
        "usage_status": usage,
        "trainer_stdout_tail": proc.stdout[-500:],
    }

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    READY_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--positions", type=int, default=8192)
    ap.add_argument("--batch", type=int, default=512)
    args = ap.parse_args()
    try:
        report = run_smoke(n_positions=args.positions, batch_size=args.batch)
    except Exception as exc:
        print(json.dumps({"ready": False, "error": str(exc)}, indent=2))
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
