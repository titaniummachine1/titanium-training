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
import sqlite3
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

from db_import import GAMES_DB_PATH, LABELS_DB_PATH
from pool_lock import TrainerRunLock
from position_usage_db import (
    claim_all_pending,
    claim_training_trigger,
    commit_epoch_training_visits,
    open_labels_db,
    pending_new_eligible,
    release_pending_claim,
)
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
from streaming_epoch_validation import is_validation_infrastructure_error, run_epoch_validation
from diversity.promotion_record import build_promotion_record
from engine_semantic_contract import prep_placeholder_contract
from net2net_auto_widen import maybe_auto_widen

LOG_DIR = _TRAINING / "data" / "overnight_logs"
PID_FILE = LOG_DIR / "training_coordinator.pid"
LOCK_FILE = LOG_DIR / "training_coordinator.lock.json"
LOG_FILE = LOG_DIR / "training_coordinator.log"
STATE_FILE = LOG_DIR / "training_coordinator_state.json"
SMOKE_READY_FILE = LOG_DIR / "streaming_training_ready.json"
PAUSE_FILE = LOG_DIR / "TRAINING_PAUSED.json"
VALIDATION_BLOCKED_DIR = _TRAINING / "runs" / "quarantine" / "cycle_0038_validation_blocked"


def training_paused() -> bool:
    if not PAUSE_FILE.is_file():
        return False
    try:
        return bool(json.loads(PAUSE_FILE.read_text(encoding="utf-8")).get("paused"))
    except json.JSONDecodeError:
        return True

# Positions needed before a new epoch trains. Retained as a bootstrap-only
# fallback (see `bootstrap` below) and as the safety net that still fires a
# cycle if game generation stalls but positions somehow keep trickling in.
# The PRIMARY trigger is now games-completed (see GAMES_TRIGGER_THRESHOLD) --
# positions-per-game varies wildly between local self-play (~40-50 new
# positions/game) and Oracle imports (many transpositions, often near 0 new
# positions/game), so a pure position count was an unreliable proxy for "how
# much real play has actually accumulated," and combined with the
# --stream-epoch-size wiring bug (fixed 2026-07-05: only --stream-max-positions
# was ever passed, so the trainer's own epoch_size fallback of 8192 always won
# regardless of this value) every cycle silently trained on the same ~8k-row
# slice. Override with STREAM_TRIGGER_THRESHOLD if needed.
TRIGGER_THRESHOLD = int(os.environ.get("STREAM_TRIGGER_THRESHOLD", "16384"))
# Games (local pool + Oracle combined, counted directly from games.db so it's
# immune to the positions-per-game variance above) needed before a new cycle
# trains. 450 sits in the middle of the requested 300-600 range.
GAMES_TRIGGER_THRESHOLD = int(os.environ.get("STREAM_GAMES_TRIGGER_THRESHOLD", "450"))
# Back off the games-needed trigger after repeated consecutive quarantines --
# a candidate that keeps failing the strength gate against the SAME parent
# needs more data to separate real signal from ~100-game noise, not just
# another same-sized retry. 2 in a row doubles it, 3 in a row quadruples it,
# etc.; capped so a systemic problem doesn't stall the pipeline forever.
# Resets to 1x the moment a candidate is accepted.
QUARANTINE_BACKOFF_CAP = int(os.environ.get("STREAM_QUARANTINE_BACKOFF_CAP", "8"))
TRAIN_FAILED_BACKOFF_SEC = float(os.environ.get("STREAM_TRAIN_FAILED_BACKOFF_SEC", "90"))
POLL_SEC = 30.0


def effective_games_trigger_threshold(consecutive_quarantines: int) -> int:
    if consecutive_quarantines < 2:
        return GAMES_TRIGGER_THRESHOLD
    multiplier = min(2 ** (consecutive_quarantines - 1), QUARANTINE_BACKOFF_CAP)
    return GAMES_TRIGGER_THRESHOLD * multiplier


def training_cycle_consumed(result: dict[str, Any]) -> bool:
    """Training finished end-to-end; games trigger and claim may be consumed."""
    return result.get("returncode") == 0 and result.get("decision") in (
        "accepted",
        "quarantined",
    )


def training_cycle_failed(result: dict[str, Any]) -> bool:
    return result.get("decision") in (
        "train_failed",
        "validation_failed",
        "validation_infrastructure_failed",
        "no_export",
        "lock_busy",
    ) or int(result.get("returncode") or 1) != 0


def validation_blocked(state: dict[str, Any]) -> bool:
    if state.get("validation_blocked"):
        return True
    blocked = VALIDATION_BLOCKED_DIR / "BLOCKED.json"
    return blocked.is_file()


def preserve_validation_blocked_artifacts(
    *,
    result: dict[str, Any],
    cycle_num: int | None,
) -> Path:
    """Copy candidate + diagnostics to a safe quarantine directory."""
    VALIDATION_BLOCKED_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "preserved_at": utc_now(),
        "cycle": cycle_num,
        "decision": result.get("decision"),
        "candidate_sha256": result.get("candidate_sha256"),
        "initializer_sha256": result.get("initializer_sha256"),
        "validation_error": result.get("validation_error"),
        "post_train_snapshot": result.get("post_train_snapshot"),
    }
    copies: list[str] = []
    post = result.get("post_train_snapshot") or {}
    for key in ("candidate_weights", "candidate_ckpt"):
        entry = post.get(key) or {}
        src = Path(str(entry.get("path", "")))
        if src.is_file():
            dest = VALIDATION_BLOCKED_DIR / src.name
            atomic_copy2(src, dest)
            copies.append(str(dest))
    for rel in (
        "epoch_diagnostics_0001.json",
        "epoch_weight_diagnostics_0001.json",
        "pending_usage_keys.json",
        "best.pt",
        "ckpt_epoch0001.pt",
    ):
        src = RUN_DIR / rel
        if src.is_file():
            dest = VALIDATION_BLOCKED_DIR / src.name
            atomic_copy2(src, dest)
            copies.append(str(dest))
    stdout_path = VALIDATION_BLOCKED_DIR / "trainer_stdout_tail.txt"
    stdout_path.write_text(str(result.get("stdout_tail") or ""), encoding="utf-8")
    stderr_path = VALIDATION_BLOCKED_DIR / "trainer_stderr_tail.txt"
    stderr_path.write_text(str(result.get("stderr_tail") or ""), encoding="utf-8")
    manifest["copied_paths"] = copies
    write_json(VALIDATION_BLOCKED_DIR / "BLOCKED.json", manifest)
    log(f"validation blocked: preserved artifacts under {VALIDATION_BLOCKED_DIR}")
    return VALIDATION_BLOCKED_DIR


def train_failed_backoff_active(state: dict[str, Any]) -> bool:
    retry_after = state.get("train_failed_retry_after")
    if not retry_after:
        return False
    try:
        deadline = datetime.fromisoformat(str(retry_after))
        if deadline.tzinfo is None:
            deadline = deadline.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) < deadline
    except ValueError:
        return False


def cleanup_incomplete_training_artifacts(*, cycle_num: int | None = None) -> list[str]:
    """Remove partial trainer outputs that could be mistaken for a finished epoch."""
    removed: list[str] = []
    for rel in (
        "pending_usage_keys.json",
        "epoch_weight_diagnostics_0001.json",
        "epoch_diagnostics_0001.json",
        "best.pt",
    ):
        path = RUN_DIR / rel
        if path.is_file():
            path.unlink()
            removed.append(str(path))
    if cycle_num is not None:
        stem = f"cycle_{cycle_num:04d}_candidate"
        for suffix in (".bin", ".pt", ".bin.sha256", ".pt.sha256"):
            path = RUN_DIR / "cycles" / f"{stem}{suffix}"
            if path.is_file():
                path.unlink()
                removed.append(str(path))
    return removed


def rollback_failed_training_attempt(
    *,
    state: dict[str, Any],
    claim: dict[str, Any],
    result: dict[str, Any],
) -> None:
    """Preserve trigger + parent weights after a crash or trainer launch failure."""
    decision = str(result.get("decision") or "")
    if decision == "train_failed":
        try:
            restored = restore_candidate_from_last_accepted()
            log(f"train_failed: restored parent weights from {restored}")
        except Exception as exc:
            log(f"train_failed: WARN could not restore parent weights: {exc}")
        cycle_num = (result.get("pre_train_snapshot") or {}).get("cycle")
        removed = cleanup_incomplete_training_artifacts(
            cycle_num=int(cycle_num) if cycle_num is not None else None,
        )
        if removed:
            log(f"train_failed: removed incomplete artifacts: {', '.join(removed)}")
    elif decision in ("validation_failed", "validation_infrastructure_failed"):
        log(f"{decision}: {result.get('validation_error', 'unknown')}")
        if decision == "validation_infrastructure_failed":
            try:
                restore_candidate_from_last_accepted()
            except Exception as exc:
                log(f"{decision}: WARN could not restore parent weights: {exc}")
            cycle_num = (result.get("pre_train_snapshot") or {}).get("cycle")
            preserve_validation_blocked_artifacts(
                result=result,
                cycle_num=int(cycle_num) if cycle_num is not None else None,
            )
            state["validation_blocked"] = True
            state["state"] = "VALIDATION_BLOCKED"
            PAUSE_FILE.write_text(
                json.dumps(
                    {
                        "paused": True,
                        "reason": "validation_infrastructure_failed",
                        "updated_at": utc_now(),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            return
        cycle_num = (result.get("pre_train_snapshot") or {}).get("cycle")
        removed = cleanup_incomplete_training_artifacts(
            cycle_num=int(cycle_num) if cycle_num is not None else None,
        )
        if removed:
            log(f"{decision}: removed incomplete artifacts: {', '.join(removed)}")

    claimed_count = int(claim.get("claimed_count") or 0)
    if claim.get("claimed") and claimed_count > 0 and decision in (
        "train_failed",
        "validation_failed",
        "lock_busy",
    ):
        con = open_labels_db(LABELS_DB_PATH)
        try:
            pending = release_pending_claim(con, claimed_count)
            log(
                f"{decision}: released pending claim count={claimed_count} "
                f"(pending_new_eligible now {pending})"
            )
        finally:
            con.close()

    retry_at = datetime.now(timezone.utc).timestamp() + TRAIN_FAILED_BACKOFF_SEC
    state["train_failed_retry_after"] = datetime.fromtimestamp(
        retry_at, tz=timezone.utc
    ).isoformat()
    state["state"] = "TRAIN_FAILED"


def games_db_max_rowid() -> int:
    """Latest games.db rowid -- a monotonic, source-agnostic 'games so far' counter."""
    if not GAMES_DB_PATH.is_file():
        return 0
    con = sqlite3.connect(str(GAMES_DB_PATH), timeout=30)
    try:
        row = con.execute("SELECT MAX(rowid) FROM games").fetchone()
        return int(row[0]) if row and row[0] is not None else 0
    finally:
        con.close()


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


def should_use_full_active_epoch(state: dict[str, Any]) -> bool:
    """Full-active training is explicit/bootstrap only, not implied by repair mode."""
    env = os.environ.get("STREAM_FULL_ACTIVE_EPOCH", "").strip().lower()
    return env in ("1", "true", "yes") or completed_training_cycles(state) == 0


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
    from prep_guard import guard_real_work

    guard_real_work("optimizer_training", detail="run_training_cycle")
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
    repair_mode = os.environ.get("STREAM_REPAIR_MODE", "").strip().lower() in ("1", "true", "yes", "on")
    default_lr = "0.0002" if repair_mode else "0.001"
    train_lr = os.environ.get("STREAM_TRAIN_LR", default_lr)
    if repair_mode:
        stale_ckpt = RUN_DIR / "best.pt"
        if stale_ckpt.is_file():
            stale_ckpt.unlink()
            log("repair mode: removed stale best.pt before fresh-optimizer cycle")
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
        str(train_lr),
        "--weight-decay",
        "0.00001",
        "--stream-epoch-size",
        str(epoch_size),
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
            try:
                restore_candidate_from_last_accepted()
            except Exception:
                pass
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
        try:
            validation = run_epoch_validation(
                checkpoint=RUN_DIR / "best.pt" if (RUN_DIR / "best.pt").is_file() else candidate_bin,
                candidate_bin=candidate_bin,
                previous_bin=prev_bin,
                short_games=int(os.environ.get("STREAM_VALIDATION_GAMES", "20")),
            )
        except Exception as exc:
            restore_candidate_from_last_accepted()
            infra = is_validation_infrastructure_error(exc)
            result.update(
                {
                    "decision": (
                        "validation_infrastructure_failed"
                        if infra
                        else "validation_failed"
                    ),
                    "accepted": False,
                    "validation_error": str(exc),
                    "returncode": 1,
                }
            )
            if infra and candidate_bin.is_file():
                result["candidate_sha256"] = sha256_file(candidate_bin)
            return result
        result["validation"] = validation
        result["candidate_sha256"] = sha256_file(candidate_bin)

        # Apply the training-visit commit trainer.py deferred to us, now that
        # accept/quarantine is actually known. A quarantined candidate never
        # shaped a promoted checkpoint, so its sampled positions must NOT be
        # marked as visited -- only the winning cycle's positions age. This
        # is a one-shot apply (the keys file is consumed either way) so a
        # quarantine never gets a second chance to sneak the same commit in.
        keys_path = RUN_DIR / "pending_usage_keys.json"
        if keys_path.is_file():
            try:
                pos_keys = json.loads(keys_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pos_keys = []
            keys_path.unlink(missing_ok=True)
            if validation.get("passed") and pos_keys:
                usage_con = open_labels_db(LABELS_DB_PATH)
                try:
                    u_commit = commit_epoch_training_visits(usage_con, pos_keys)
                    log(
                        f"usage commit (accepted candidate): touched={u_commit.get('touched', 0)} "
                        f"retired_total={u_commit.get('retired_total', 0)}"
                    )
                finally:
                    usage_con.close()

        epoch_num = completed_training_cycles(read_json(STATE_FILE)) + 1
        # trainer.py is always invoked with --epochs 1, so its own internal
        # counter starts at 0 and it always names the file
        # epoch_diagnostics_0001.json -- using the real streaming epoch_num
        # here instead never matched an existing file (that number only
        # exists once the chain has run that many cycles), so `diag` silently
        # stayed {} in every persisted epoch report forever.
        diag_path = RUN_DIR / "epoch_diagnostics_0001.json"
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

        parent_entry = latest_accepted()
        parent_epoch = int(parent_entry["epoch"]) if parent_entry else None
        grandparent_match = validation.get("match_vs_grandparent") or {}
        promotion_meta = build_promotion_record(
            epoch_id=epoch_num,
            candidate_weights_sha256=result["candidate_sha256"],
            parent_weights_sha256=sha256_file(prev_bin) if prev_bin and prev_bin.is_file() else None,
            parent_accepted_epoch=parent_epoch,
            grandparent_validation_epoch=grandparent_match.get("grandparent_epoch"),
            engine_semantic_hash=prep_placeholder_contract().semantics_hash(),
            validation=validation,
            decision="accepted" if validation.get("passed") else "quarantined",
        )

        if validation.get("passed"):
            accept_checkpoint(
                weights_path=candidate_bin,
                epoch=epoch_num,
                validation=validation,
                ckpt_path=post_ckpt if post_ckpt.is_file() else None,
                promotion_record=promotion_meta.to_dict(),
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
                promotion_record=promotion_meta.to_dict(),
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
    from prep_guard import guard_real_work

    guard_real_work("optimizer_training", detail="coordinator_loop")
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
    if "last_train_games_rowid" not in state:
        # First run under games-based triggering: start counting from now
        # rather than treating the entire historical games.db as "pending".
        state["last_train_games_rowid"] = games_db_max_rowid()
        write_json(STATE_FILE, state)
    while True:
        try:
            persisted = read_json(STATE_FILE)
            for key in (
                "last_train_games_rowid",
                "train_failed_retry_after",
                "consecutive_quarantines",
            ):
                if key in persisted:
                    state[key] = persisted[key]
            if "train_failed_retry_after" not in persisted:
                state.pop("train_failed_retry_after", None)

            con = open_labels_db(LABELS_DB_PATH)
            pending = pending_new_eligible(con)
            counts = db_counts(LABELS_DB_PATH)
            games_now = games_db_max_rowid()
            games_since = max(0, games_now - int(state.get("last_train_games_rowid", games_now)))
            consecutive_quarantines = int(state.get("consecutive_quarantines", 0))
            games_trigger_threshold = effective_games_trigger_threshold(consecutive_quarantines)
            state.update(
                {
                    "updated_at": utc_now(),
                    "pid": os.getpid(),
                    "pending_new_eligible": pending,
                    "eligible_positions": counts.eligible_positions,
                    "labeled_positions": counts.labeled_positions,
                    "games_since_last_train": games_since,
                    "games_trigger_threshold": games_trigger_threshold,
                    "consecutive_quarantines": consecutive_quarantines,
                    "state": "IDLE",
                    "last_error": None,
                }
            )
            should_train = False
            claim: dict[str, Any] = {"claimed": False, "claimed_count": 0, "remaining": pending}
            bootstrap = (
                games_since < games_trigger_threshold
                and completed_training_cycles(state) == 0
                and smoke_ready()
            )
            if training_paused() or validation_blocked(state):
                state["state"] = "VALIDATION_BLOCKED" if validation_blocked(state) else "PAUSED"
                write_json(STATE_FILE, state)
                time.sleep(poll_sec)
                continue

            backoff_active = train_failed_backoff_active(state)

            # Primary trigger: enough completed games (pool + Oracle combined,
            # counted directly from games.db) have accumulated -- train on
            # EVERYTHING currently eligible, not a fixed-size slice. Threshold
            # backs off (doubles, doubles again, ...) after repeated
            # consecutive quarantines -- see effective_games_trigger_threshold.
            if games_since >= games_trigger_threshold and not backoff_active:
                claim = claim_all_pending(con)
                should_train = bool(claim.get("claimed"))
                if should_train:
                    log(
                        f"games trigger: {games_since}/{games_trigger_threshold} games since last train "
                        f"(consecutive_quarantines={consecutive_quarantines}) — "
                        f"claiming full backlog count={claim['claimed_count']} eligible={counts.eligible_positions}"
                    )
            elif backoff_active and games_since >= games_trigger_threshold:
                state["state"] = "TRAIN_FAILED_BACKOFF"
            # Fallback safety net: if game-count tracking ever stalls (e.g. a
            # games.db path issue) but positions are still somehow piling up,
            # don't sit idle forever.
            elif pending >= TRIGGER_THRESHOLD and not backoff_active:
                claim = claim_all_pending(con)
                should_train = bool(claim.get("claimed"))
                if should_train:
                    log(
                        f"position-count fallback trigger: pending={pending} "
                        f"(games_since={games_since}/{games_trigger_threshold}) — "
                        f"claiming full backlog count={claim['claimed_count']}"
                    )
            elif bootstrap:
                should_train = True
                claim = claim_all_pending(con)
                state["bootstrap_epoch"] = True
                log(
                    "bootstrap first production epoch: smoke passed, corpus ready, "
                    f"games_since={games_since}/{games_trigger_threshold} — training now"
                )
            con.close()

            if should_train:
                state["state"] = "TRAINING"
                state["last_claim"] = claim
                write_json(STATE_FILE, state)
                games_triggered = games_since >= games_trigger_threshold
                cycle_epoch_size = claim.get("claimed_count") or epoch_size
                if games_triggered:
                    cycle_epoch_size = max(int(cycle_epoch_size), int(epoch_size))
                repair_mode = os.environ.get("STREAM_REPAIR_MODE", "").strip().lower() in (
                    "1",
                    "true",
                    "yes",
                    "on",
                )
                try:
                    result = run_training_cycle(
                        epoch_size=cycle_epoch_size,
                        batch=batch,
                        featurize_chunk=featurize_chunk,
                        full_active_epoch=should_use_full_active_epoch(state),
                    )
                except Exception as exc:
                    log(f"training cycle exception: {exc}")
                    result = {
                        "returncode": 1,
                        "decision": "train_failed",
                        "accepted": False,
                        "error": str(exc),
                    }
                result["claim"] = claim
                state["last_training_result"] = result
                if training_cycle_consumed(result):
                    state["last_train_games_rowid"] = games_now
                    state.pop("train_failed_retry_after", None)
                    state["state"] = "IDLE"
                    if result.get("accepted"):
                        state["completed_training_cycles"] = completed_training_cycles(state) + 1
                        state["consecutive_quarantines"] = 0
                    elif result.get("decision") == "quarantined":
                        state["consecutive_quarantines"] = consecutive_quarantines + 1
                elif training_cycle_failed(result):
                    rollback_failed_training_attempt(state=state, claim=claim, result=result)
                state.pop("bootstrap_epoch", None)

                if result["returncode"] == 0 and training_cycle_consumed(result):
                    widen_result = maybe_auto_widen(
                        state, completed_cycles=completed_training_cycles(state)
                    )
                    if widen_result:
                        state["last_net2net_widen"] = widen_result
                        if widen_result.get("widened"):
                            state["completed_training_cycles"] = completed_training_cycles(state) + 1
                            state["consecutive_quarantines"] = 0
                            log(
                                "NET2NET AUTO-WIDEN: plateau after "
                                f"{widen_result.get('trigger_consecutive_quarantines')} "
                                f"consecutive quarantines — h {widen_result['old_h']} -> {widen_result['new_h']} "
                                f"(from accepted epoch {widen_result['source_epoch']}, "
                                f"new epoch {widen_result['epoch']}); resuming training on widened net"
                            )
                        else:
                            log(f"NET2NET AUTO-WIDEN skipped: {widen_result.get('reason')}")
                log(
                    "training cycle complete "
                    f"rc={result['returncode']} decision={result.get('decision')} "
                    f"accepted={result.get('accepted')} "
                    f"epoch_size={cycle_epoch_size} "
                    f"cycles={state.get('completed_training_cycles', 0)} "
                    f"consecutive_quarantines={state.get('consecutive_quarantines', 0)}"
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
    from prep_guard import guard_real_work

    guard_real_work("optimizer_training", detail="training_coordinator")
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
