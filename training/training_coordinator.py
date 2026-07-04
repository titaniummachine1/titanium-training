#!/usr/bin/env python3
"""Database-first training coordinator.

Long-lived service responsibilities:
  - watch labels.db pending_new_eligible counter
  - claim exactly 2048 positions per training cycle, preserving overflow
  - train directly from labels.db with the existing NNUE trainer
  - run existing validation/promotion gate

It never starts game generation, Oracle import, cache rebuilds, supervisors, or
restart loops for other services.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil

_TRAINING = Path(__file__).resolve().parent
_REPO = _TRAINING.parent
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

from db_import import LABELS_DB_PATH
from pool_lock import TrainerRunLock
from position_usage_db import claim_training_trigger, open_labels_db, pending_new_eligible
from streaming_db_loader import db_counts
from streaming_checkpoint_chain import (
    BEST_WEIGHTS,
    ENGINE_WEIGHTS,
    PREVIOUS_WEIGHTS,
    RUN_DIR,
    accept_checkpoint,
    atomic_copy2,
    ensure_checkpoint_integrity,
    ensure_epoch_zero,
    latest_accepted,
    load_chain,
    quarantine_checkpoint,
    refresh_pool_weights_snapshot,
    resolve_latest_accepted_weights,
    restore_candidate_from_last_accepted,
    sha256_file,
    snapshot_cycle_candidate,
    snapshot_cycle_pre_train,
)
from streaming_epoch_report import usage_distribution, write_epoch_report
from streaming_epoch_validation import run_epoch_validation

LOG_DIR = _TRAINING / "data" / "overnight_logs"
PID_FILE = LOG_DIR / "training_coordinator.pid"
LOCK_FILE = LOG_DIR / "training_coordinator.lock.json"
LOG_FILE = LOG_DIR / "training_coordinator.log"
STATE_FILE = LOG_DIR / "training_coordinator_state.json"
SMOKE_READY_FILE = LOG_DIR / "streaming_training_ready.json"
PAUSE_FILE = LOG_DIR / "TRAINING_PAUSED.json"


def training_paused() -> bool:
    if not PAUSE_FILE.is_file():
        return False
    try:
        return bool(json.loads(PAUSE_FILE.read_text(encoding="utf-8")).get("paused"))
    except json.JSONDecodeError:
        return True

# Positions needed before a new epoch trains. 2048 was calibrated for the old
# one-cold-process-per-move self-play (~4-5min/epoch is too fast for a
# meaningful gradient update AND leaves near-zero prior-epoch strength signal
# accumulated). With warm per-game engine sessions (continuous_pool.py /
# engine_session.py) throughput is roughly 350-400 new positions/min on this
# 4-core box; 16384 lands epochs around ~40-50 minutes (~1-1.5/hour) so each
# epoch also carries ~4900 real current-vs-previous positions (30% share) --
# well past the 600-position floor streaming_epoch_validation.py needs to
# trust the strength gate. Override with STREAM_TRIGGER_THRESHOLD if real
# measured throughput calls for retuning.
TRIGGER_THRESHOLD = int(os.environ.get("STREAM_TRIGGER_THRESHOLD", "16384"))
POLL_SEC = 30.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(msg: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{utc_now()}] {msg}"
    print(line, flush=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def smoke_ready() -> bool:
    return bool(read_json(SMOKE_READY_FILE).get("ready"))


def completed_training_cycles(state: dict[str, Any]) -> int:
    """Accepted streaming epochs (excludes deployed epoch_0)."""
    chain_epochs = len(load_chain().get("epochs") or [])
    return max(0, chain_epochs - 1)


def next_streaming_epoch_number() -> int:
    """Next chain index: epoch_0=deployed, epoch_1=first accepted streaming, ..."""
    return len(load_chain().get("epochs") or [])


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot_previous_weights() -> dict[str, Any]:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    try:
        source = resolve_latest_accepted_weights()
    except (FileNotFoundError, RuntimeError):
        source = BEST_WEIGHTS if BEST_WEIGHTS.is_file() else ENGINE_WEIGHTS
    if not source.is_file():
        return {"ok": False, "reason": "no_existing_best_or_engine_weights"}
    atomic_copy2(source, PREVIOUS_WEIGHTS)
    return {
        "ok": True,
        "source": str(source),
        "path": str(PREVIOUS_WEIGHTS),
        "sha256": _sha256_file(PREVIOUS_WEIGHTS),
    }


def run_training_cycle(*, epoch_size: int, batch: int, featurize_chunk: int, full_active_epoch: bool = False) -> dict[str, Any]:
    ensure_epoch_zero()
    integrity = ensure_checkpoint_integrity(repair_best=True)
    if integrity.get("missing"):
        log(f"checkpoint integrity: missing accepted snapshots {integrity['missing']}")
    if integrity.get("repaired_best"):
        log("checkpoint integrity: repaired net_weights_best.bin from last recoverable accepted")
    RUN_DIR = BEST_WEIGHTS.parent
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    cycle_num = completed_training_cycles(read_json(STATE_FILE)) + 1
    if (la := latest_accepted()) is not None:
        try:
            init_weights = resolve_latest_accepted_weights()
        except FileNotFoundError as exc:
            log(f"WARN: {exc}; training from deployed engine weights")
            init_weights = ENGINE_WEIGHTS
    else:
        init_weights = ENGINE_WEIGHTS
    previous_snapshot = snapshot_previous_weights()
    ckpt_path = RUN_DIR / "best.pt"
    if not ckpt_path.is_file():
        ckpt_path = max(RUN_DIR.glob("ckpt_epoch*.pt"), default=None, key=lambda p: p.stat().st_mtime)

    pre_snap = snapshot_cycle_pre_train(
        cycle=cycle_num,
        init_weights=init_weights,
        ckpt=ckpt_path if ckpt_path and ckpt_path.is_file() else None,
    )

    trainer = _TRAINING / "titanium_training" / "training" / "trainer.py"
    retired_frac = os.environ.get("STREAM_RETIRED_REPLAY_FRACTION", "0.05")
    cmd = [
        sys.executable,
        str(trainer),
        "--labels-db",
        str(LABELS_DB_PATH),
        "--out-dir",
        str(RUN_DIR),
        "--weights",
        str(init_weights),
        "--epochs",
        "1",
        "--batch",
        str(batch),
        "--lr",
        "0.001",
        "--weight-decay",
        "0.00001",
        "--stream-max-positions",
        str(epoch_size),
        "--stream-featurize-chunk",
        str(featurize_chunk),
        "--stream-retired-replay-fraction",
        str(retired_frac),
        "--stream-old-refresh-fraction",
        os.environ.get("STREAM_OLD_REFRESH_FRACTION", "0.05"),
        "--val-split",
        "0.05",
        "--checkpoint-steps",
        "999999",
        "--patience",
        "0",
        "--cpu",
        "--defer-usage-commit",
        "--log-every",
        "100",
        "--log-interval-sec",
        "30",
    ]
    if full_active_epoch:
        cmd.append("--stream-full-active-epoch")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_TRAINING)
    env["RUSTFLAGS"] = "-C target-cpu=native"

    # Every process that spawns trainer.py and touches RUN_DIR / BEST_WEIGHTS /
    # ENGINE_WEIGHTS must hold this lock — continuous_pool.py (or any other
    # trainer) does the same. Busy means "skip this cycle, the coordinator
    # loop retries on its next poll" — never block waiting for it.
    with TrainerRunLock() as lock:
        if lock is None:
            # returncode=0 so the coordinator state machine treats this as a
            # normal idle tick (retry next poll), not a training failure.
            return {
                "decision": "lock_busy",
                "accepted": False,
                "returncode": 0,
                "reason": "training lock held by another process",
            }

        started = time.perf_counter()
        proc = subprocess.run(
            cmd,
            cwd=str(_REPO),
            capture_output=True,
            text=True,
            timeout=24 * 3600,
            env=env,
        )
        elapsed = time.perf_counter() - started
        result: dict[str, Any] = {
            "returncode": proc.returncode,
            "elapsed_sec": round(elapsed, 1),
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-4000:],
            "initializer_sha256": sha256_file(init_weights),
            "previous_snapshot": previous_snapshot,
            "pre_train_snapshot": pre_snap,
        }
        if proc.returncode != 0:
            result.update({"decision": "train_failed", "accepted": False})
            return result

        candidate_bin = RUN_DIR / "net_weights_best.bin"
        if not candidate_bin.is_file():
            result.update({"decision": "no_export", "accepted": False})
            return result

        post_ckpt = RUN_DIR / "best.pt"
        result["post_train_snapshot"] = snapshot_cycle_candidate(
            cycle=cycle_num,
            candidate_bin=candidate_bin,
            ckpt=post_ckpt if post_ckpt.is_file() else None,
        )

        prev_bin = PREVIOUS_WEIGHTS if PREVIOUS_WEIGHTS.is_file() else None
        validation = run_epoch_validation(
            checkpoint=RUN_DIR / "best.pt" if (RUN_DIR / "best.pt").is_file() else candidate_bin,
            candidate_bin=candidate_bin,
            previous_bin=prev_bin,
            short_games=int(os.environ.get("STREAM_VALIDATION_GAMES", "20")),
        )
        result["validation"] = validation
        result["candidate_sha256"] = sha256_file(candidate_bin)

        epoch_num = completed_training_cycles(read_json(STATE_FILE)) + 1
        diag_path = RUN_DIR / f"epoch_diagnostics_{epoch_num:04d}.json"
        diag = read_json(diag_path) if diag_path.is_file() else {}

        con = open_labels_db(LABELS_DB_PATH)
        usage_dist = usage_distribution(con)
        con.close()

        report_path = write_epoch_report(
            epoch_num,
            {
                "trigger": result.get("claim"),
                "training": diag,
                "validation": validation,
                "usage_distribution": usage_dist,
                "initializer_sha256": result["initializer_sha256"],
                "candidate_sha256": result["candidate_sha256"],
            },
        )
        result["epoch_report"] = str(report_path)

        if validation.get("passed"):
            accept_checkpoint(
                weights_path=candidate_bin,
                epoch=epoch_num,
                validation=validation,
                ckpt_path=post_ckpt if post_ckpt.is_file() else None,
            )
            result.update({"decision": "accepted", "accepted": True, "promoted": False})
            log(f"epoch {epoch_num} ACCEPTED sha={result['candidate_sha256'][:16]} snap=accepted/epoch_{epoch_num:04d}.bin")
        else:
            quarantine_checkpoint(
                weights_path=candidate_bin,
                reason=validation.get("reject_reason", "validation_failed"),
                validation=validation,
                ckpt_path=post_ckpt if post_ckpt.is_file() else None,
                cycle=cycle_num,
            )
            restore_candidate_from_last_accepted()
            result.update({"decision": "quarantined", "accepted": False, "promoted": False})
            log(f"epoch {epoch_num} QUARANTINED reason={validation.get('reject_reason')}")
        return result


def pid_alive(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        proc = psutil.Process(pid)
        return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return False


def acquire_lock() -> bool:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if LOCK_FILE.is_file():
        old = read_json(LOCK_FILE)
        old_pid = old.get("pid")
        try:
            pid = int(old_pid) if old_pid is not None else None
        except (TypeError, ValueError):
            pid = None
        if pid_alive(pid):
            return False
    payload = {"pid": os.getpid(), "started_at": utc_now()}
    write_json(LOCK_FILE, payload)
    PID_FILE.write_text(str(os.getpid()), encoding="ascii")
    return True


def coordinator_loop(*, poll_sec: float, epoch_size: int, batch: int, featurize_chunk: int) -> int:
    if not acquire_lock():
        log("another training_coordinator owns the lock; exiting")
        return 0
    log("training_coordinator started")
    if training_paused():
        log("TRAINING_PAUSED.json set — coordinator idle until gates pass")
    ensure_epoch_zero()
    boot = ensure_checkpoint_integrity(repair_best=True)
    if boot.get("missing"):
        log(f"checkpoint integrity: {len(boot['missing'])} accepted epoch(s) missing immutable snapshot")
    if boot.get("repaired_best"):
        log("checkpoint integrity: reset net_weights_best.bin to last recoverable accepted weights")
    state = read_json(STATE_FILE)
    while True:
        try:
            con = open_labels_db(LABELS_DB_PATH)
            pending = pending_new_eligible(con)
            counts = db_counts(LABELS_DB_PATH)
            state.update(
                {
                    "updated_at": utc_now(),
                    "pid": os.getpid(),
                    "pending_new_eligible": pending,
                    "eligible_positions": counts.eligible_positions,
                    "labeled_positions": counts.labeled_positions,
                    "state": "IDLE",
                    "last_error": None,
                }
            )
            should_train = False
            claim: dict[str, Any] = {"claimed": False, "claimed_count": 0, "remaining": pending}
            bootstrap = (
                pending < TRIGGER_THRESHOLD
                and completed_training_cycles(state) == 0
                and smoke_ready()
            )
            if training_paused():
                state["state"] = "PAUSED"
                write_json(STATE_FILE, state)
                time.sleep(poll_sec)
                continue

            if pending >= TRIGGER_THRESHOLD:
                claim = claim_training_trigger(con, TRIGGER_THRESHOLD)
                should_train = bool(claim.get("claimed"))
                if should_train:
                    log(
                        f"claimed trigger count={claim['claimed_count']} "
                        f"remaining={claim['remaining']} eligible={counts.eligible_positions}"
                    )
            elif bootstrap:
                should_train = True
                log(
                    "bootstrap first production epoch: smoke passed, corpus ready, "
                    f"pending={pending}/{TRIGGER_THRESHOLD} — training now"
                )
            con.close()

            if should_train:
                state["state"] = "TRAINING"
                state["last_claim"] = claim
                if bootstrap:
                    state["bootstrap_epoch"] = True
                write_json(STATE_FILE, state)
                result = run_training_cycle(
                    epoch_size=epoch_size,
                    batch=batch,
                    featurize_chunk=featurize_chunk,
                    full_active_epoch=(
                        os.environ.get("STREAM_FULL_ACTIVE_EPOCH", "").strip() in ("1", "true", "yes")
                        or completed_training_cycles(state) == 0
                    ),
                )
                result["claim"] = claim
                state["last_training_result"] = result
                state["state"] = "IDLE" if result["returncode"] == 0 else "TRAIN_FAILED"
                if result["returncode"] == 0 and result.get("accepted"):
                    state["completed_training_cycles"] = completed_training_cycles(state) + 1
                state.pop("bootstrap_epoch", None)
                log(
                    "training cycle complete "
                    f"rc={result['returncode']} decision={result.get('decision')} "
                    f"accepted={result.get('accepted')} "
                    f"cycles={state.get('completed_training_cycles', 0)}"
                )
            write_json(STATE_FILE, state)
        except Exception as exc:
            state["state"] = "ERROR"
            state["last_error"] = str(exc)
            state["updated_at"] = utc_now()
            write_json(STATE_FILE, state)
            log(f"coordinator error: {exc}")
        time.sleep(poll_sec)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--poll-sec", type=float, default=POLL_SEC)
    ap.add_argument("--epoch-size", type=int, default=TRIGGER_THRESHOLD)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--featurize-chunk", type=int, default=4096)
    args = ap.parse_args()
    return coordinator_loop(
        poll_sec=args.poll_sec,
        epoch_size=args.epoch_size,
        batch=args.batch,
        featurize_chunk=args.featurize_chunk,
    )


if __name__ == "__main__":
    raise SystemExit(main())
