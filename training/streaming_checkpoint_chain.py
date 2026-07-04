"""Accepted checkpoint chain for database-first streaming NNUE training."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_TRAINING = Path(__file__).resolve().parent
_REPO = _TRAINING.parent
LOG_DIR = _TRAINING / "data" / "overnight_logs"
CHAIN_PATH = LOG_DIR / "accepted_checkpoint_chain.json"
RUN_DIR = _TRAINING / "runs" / "v16"
BEST_WEIGHTS = RUN_DIR / "net_weights_best.bin"
PREVIOUS_WEIGHTS = RUN_DIR / "net_weights_previous.bin"
# Self-play reads this copy so coordinator can overwrite BEST_WEIGHTS during
# quarantine restore while titanium subprocesses still hold genmove handles.
POOL_ACTIVE_WEIGHTS = RUN_DIR / "net_weights_pool_active.bin"
POOL_WEIGHTS_STAMP = RUN_DIR / "net_weights_pool_active.sha256"
ENGINE_WEIGHTS = _REPO / "engine" / "src" / "titanium" / "net_weights.bin"
FROZEN_WEIGHTS = _REPO / "engine" / "src" / "titanium" / "net_weights_frozen.bin"
QUARANTINE_DIR = RUN_DIR / "quarantine"
ACCEPTED_DIR = RUN_DIR / "accepted"
CYCLES_DIR = RUN_DIR / "cycles"
INTEGRITY_PATH = LOG_DIR / "checkpoint_integrity.json"

# Invalid fixed-run checkpoint — never load, deploy, or use as opponent.
INVALID_WEIGHT_PATHS = frozenset(
    {
        (_TRAINING / "runs" / "h32_control_retrain" / "net_weights_control_best.bin").resolve(),
        (_TRAINING / "runs" / "h32_control_retrain" / "net_weights_best.bin").resolve(),
    }
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_file_lock_error(exc: BaseException) -> bool:
    if isinstance(exc, PermissionError):
        return True
    if isinstance(exc, OSError):
        if getattr(exc, "winerror", None) == 32:
            return True
        if exc.errno in (13, 16, 32):
            return True
    return False


def atomic_copy2(
    src: Path,
    dst: Path,
    *,
    retries: int = 40,
    delay_sec: float = 0.25,
) -> None:
    """Copy *src* to *dst* with temp-file replace and Win32 lock retries."""
    src = Path(src)
    dst = Path(dst)
    if not src.is_file():
        raise FileNotFoundError(f"atomic copy source missing: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    last_err: BaseException | None = None
    for _ in range(retries):
        try:
            # In-place overwrite works on Windows when readers use shared read.
            shutil.copy2(src, dst)
            return
        except OSError as exc:
            last_err = exc
            if not _is_file_lock_error(exc):
                raise
            time.sleep(delay_sec)

    tmp = dst.with_name(f"{dst.name}.tmp.{os.getpid()}")
    for _ in range(retries):
        try:
            shutil.copy2(src, tmp)
            if sys.platform == "win32" and dst.is_file():
                try:
                    dst.unlink()
                except OSError as exc:
                    if _is_file_lock_error(exc):
                        raise
                    raise
            tmp.replace(dst)
            return
        except OSError as exc:
            last_err = exc
            if not _is_file_lock_error(exc):
                raise
            time.sleep(delay_sec)
        finally:
            tmp.unlink(missing_ok=True)
    if last_err is not None:
        raise last_err
    raise RuntimeError(f"atomic copy failed without error: {src} -> {dst}")


def refresh_pool_weights_snapshot(*, source: Path | None = None) -> bool:
    """Copy latest accepted/candidate weights to the pool-only shadow file."""
    src = Path(source) if source is not None else candidate_weights_path()
    if not src.is_file():
        return False
    sha = sha256_file(src)
    if (
        sha
        and POOL_WEIGHTS_STAMP.is_file()
        and POOL_ACTIVE_WEIGHTS.is_file()
        and POOL_WEIGHTS_STAMP.read_text(encoding="utf-8").strip() == sha
    ):
        return True
    atomic_copy2(src, POOL_ACTIVE_WEIGHTS)
    if sha:
        POOL_WEIGHTS_STAMP.write_text(sha + "\n", encoding="utf-8")
    return True


def pool_weights_path() -> Path:
    """Weights path for self-play — never the live training candidate file."""
    if POOL_ACTIVE_WEIGHTS.is_file():
        return POOL_ACTIVE_WEIGHTS
    return candidate_weights_path()


def sync_pool_weights_after_checkpoint(*, source: Path | None = None) -> bool:
    """Refresh pool shadow weights after accept/quarantine restore."""
    return refresh_pool_weights_snapshot(source=source)


def worker_game_weights_dir() -> Path:
    d = RUN_DIR / "pool_worker_weights"
    d.mkdir(parents=True, exist_ok=True)
    return d


def freeze_worker_game_weights(worker_id: int, *, current: Path, previous: Path) -> tuple[Path, Path]:
    """Snapshot weights ONCE at game start into a private per-worker pair of files.

    A game spawns one titanium subprocess per move; if it read the shared,
    mutable pool-active/previous-opponent paths directly, a checkpoint accept
    landing mid-game could silently swap weights partway through a single
    game on a single side. These frozen copies are only refreshed between
    games, never while a game is in flight, so one side's weights never
    change until that game ends.
    """
    d = worker_game_weights_dir()
    cur_dst = d / f"worker{worker_id:02d}_current.bin"
    prev_dst = d / f"worker{worker_id:02d}_previous.bin"
    atomic_copy2(current, cur_dst)
    if previous.resolve() != current.resolve():
        atomic_copy2(previous, prev_dst)
    else:
        atomic_copy2(current, prev_dst)
    return cur_dst, prev_dst


def assert_not_invalid(path: Path) -> None:
    resolved = path.resolve()
    if resolved in INVALID_WEIGHT_PATHS:
        raise ValueError(f"refusing invalid quarantined checkpoint: {resolved}")


def load_chain() -> dict[str, Any]:
    if CHAIN_PATH.is_file():
        return json.loads(CHAIN_PATH.read_text(encoding="utf-8"))
    return {"epochs": [], "quarantined": [], "invalid_excluded": [str(p) for p in INVALID_WEIGHT_PATHS]}


def save_chain(chain: dict[str, Any]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    chain["updated_at"] = _utc_now()
    CHAIN_PATH.write_text(json.dumps(chain, indent=2) + "\n", encoding="utf-8")


def ensure_epoch_zero() -> dict[str, Any]:
    """Initialize run dir and chain from deployed H32 weights (never invalid control run)."""
    assert_not_invalid(ENGINE_WEIGHTS)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    chain = load_chain()
    if not chain.get("epochs"):
        if not ENGINE_WEIGHTS.is_file():
            raise FileNotFoundError(f"deployed weights missing: {ENGINE_WEIGHTS}")
        if not BEST_WEIGHTS.is_file():
            atomic_copy2(ENGINE_WEIGHTS, BEST_WEIGHTS)
        refresh_pool_weights_snapshot(source=ENGINE_WEIGHTS)
        epoch0 = {
            "epoch": 0,
            "role": "deployed_baseline",
            "path": str(ENGINE_WEIGHTS),
            "sha256": sha256_file(ENGINE_WEIGHTS),
            "accepted_at": _utc_now(),
        }
        chain["epochs"] = [epoch0]
        save_chain(chain)
    return chain


def latest_accepted() -> dict[str, Any] | None:
    chain = load_chain()
    epochs = chain.get("epochs") or []
    return epochs[-1] if epochs else None


def previous_accepted() -> dict[str, Any] | None:
    chain = load_chain()
    epochs = chain.get("epochs") or []
    return epochs[-2] if len(epochs) >= 2 else None


def candidate_weights_path() -> Path:
    """Current training candidate — v16 run best, falling back to deployed."""
    if BEST_WEIGHTS.is_file():
        assert_not_invalid(BEST_WEIGHTS)
        return BEST_WEIGHTS
    assert_not_invalid(ENGINE_WEIGHTS)
    return ENGINE_WEIGHTS


def previous_opponent_weights_path() -> Path | None:
    """Immediately previous accepted checkpoint for 30% generation games."""
    prev = previous_accepted()
    if prev is None:
        return None
    try:
        return resolve_accepted_weights(prev)
    except FileNotFoundError:
        return None


def accepted_snapshot_path(epoch: int) -> Path:
    return ACCEPTED_DIR / f"epoch_{epoch:04d}.bin"


def accepted_pt_snapshot_path(epoch: int) -> Path:
    return ACCEPTED_DIR / f"epoch_{epoch:04d}.pt"


def cycle_snapshot_stem(cycle: int) -> str:
    return f"cycle_{cycle:04d}"


def _copy_optional(src: Path, dst: Path) -> bool:
    if not src.is_file():
        return False
    atomic_copy2(src, dst)
    return True


def snapshot_weights(src: Path, dst: Path) -> dict[str, Any]:
    """Immutable weight snapshot with sha256 sidecar."""
    src = Path(src)
    dst = Path(dst)
    atomic_copy2(src, dst)
    sha = sha256_file(dst)
    stamp = dst.with_suffix(dst.suffix + ".sha256")
    if sha:
        stamp.write_text(sha + "\n", encoding="utf-8")
    return {"path": str(dst.resolve()), "sha256": sha, "bytes": dst.stat().st_size}


def snapshot_cycle_pre_train(
    *,
    cycle: int,
    init_weights: Path,
    ckpt: Path | None = None,
) -> dict[str, Any]:
    """Before trainer overwrites anything — preserve starting point."""
    CYCLES_DIR.mkdir(parents=True, exist_ok=True)
    stem = cycle_snapshot_stem(cycle)
    out: dict[str, Any] = {"cycle": cycle, "phase": "pre_train"}
    out["init_weights"] = snapshot_weights(init_weights, CYCLES_DIR / f"{stem}_init.bin")
    if ckpt is not None and ckpt.is_file():
        out["init_ckpt"] = snapshot_weights(ckpt, CYCLES_DIR / f"{stem}_init.pt")
    return out


def snapshot_cycle_candidate(
    *,
    cycle: int,
    candidate_bin: Path,
    ckpt: Path | None = None,
) -> dict[str, Any]:
    """After trainer, before validation — candidate cannot be lost to restore."""
    CYCLES_DIR.mkdir(parents=True, exist_ok=True)
    stem = cycle_snapshot_stem(cycle)
    out: dict[str, Any] = {"cycle": cycle, "phase": "post_train_candidate"}
    out["candidate_weights"] = snapshot_weights(candidate_bin, CYCLES_DIR / f"{stem}_candidate.bin")
    if ckpt is not None and ckpt.is_file():
        out["candidate_ckpt"] = snapshot_weights(ckpt, CYCLES_DIR / f"{stem}_candidate.pt")
    return out


def _find_sha_anywhere(expected: str) -> Path | None:
    """Search run dirs for a weight blob matching *expected* sha256."""
    if not expected:
        return None
    roots = (
        ACCEPTED_DIR,
        CYCLES_DIR,
        QUARANTINE_DIR,
        RUN_DIR,
    )
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*.bin"):
            try:
                if path.stat().st_size < 100_000:
                    continue
                if sha256_file(path) == expected:
                    return path
            except OSError:
                continue
    if ENGINE_WEIGHTS.is_file() and sha256_file(ENGINE_WEIGHTS) == expected:
        return ENGINE_WEIGHTS
    return None


def backfill_accepted_snapshot(entry: dict[str, Any]) -> Path | None:
    """Ensure accepted/epoch_NNNN.bin exists; repair chain path if recovered."""
    epoch = int(entry.get("epoch", -1))
    expected = entry.get("sha256")
    if not expected:
        return None
    snap = accepted_snapshot_path(epoch)
    if snap.is_file() and sha256_file(snap) == expected:
        return snap
    found = _find_sha_anywhere(expected)
    if found is not None:
        atomic_copy2(found, snap)
        return snap
    return None


def ensure_checkpoint_integrity(*, repair_best: bool = True) -> dict[str, Any]:
    """Verify accepted snapshots, backfill where possible, fix live best if corrupt."""
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    ACCEPTED_DIR.mkdir(parents=True, exist_ok=True)
    CYCLES_DIR.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "checked_at": _utc_now(),
        "accepted": [],
        "missing": [],
        "repaired_chain_paths": [],
        "repaired_best": False,
    }
    chain = load_chain()
    changed = False
    for entry in chain.get("epochs") or []:
        epoch = int(entry["epoch"])
        expected = entry.get("sha256")
        snap = backfill_accepted_snapshot(entry)
        if epoch == 0 and snap is None and ENGINE_WEIGHTS.is_file():
            snap = accepted_snapshot_path(0)
            atomic_copy2(ENGINE_WEIGHTS, snap)
            entry["path"] = str(snap.resolve())
            entry["sha256"] = sha256_file(snap)
            changed = True
            report["repaired_chain_paths"].append(epoch)
        if snap is not None:
            canonical = str(snap.resolve())
            if entry.get("path") != canonical:
                entry["path"] = canonical
                changed = True
                report["repaired_chain_paths"].append(epoch)
            report["accepted"].append(
                {"epoch": epoch, "path": canonical, "sha256": expected, "ok": True}
            )
        else:
            report["missing"].append({"epoch": epoch, "sha256": expected})
    if changed:
        save_chain(chain)
    if repair_best:
        chain_epochs = chain.get("epochs") or []
        good: Path | None = None
        good_sha: str | None = None
        for entry in reversed(chain_epochs):
            try:
                good = resolve_accepted_weights(entry)
                good_sha = entry.get("sha256")
                break
            except FileNotFoundError:
                continue
        if good is not None:
            best_sha = sha256_file(BEST_WEIGHTS)
            if best_sha != good_sha:
                atomic_copy2(good, BEST_WEIGHTS)
                refresh_pool_weights_snapshot(source=good)
                report["repaired_best"] = True
                report["best_sha_before"] = best_sha
                report["best_sha_after"] = good_sha
                report["repaired_best_from_epoch"] = next(
                    (int(e["epoch"]) for e in reversed(chain_epochs) if e.get("sha256") == good_sha),
                    None,
                )
    INTEGRITY_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def resolve_accepted_weights(entry: dict[str, Any]) -> Path:
    """Return on-disk weights for an accepted chain entry (sha-verified)."""
    expected = entry.get("sha256")
    if not expected:
        raise ValueError("accepted chain entry missing sha256")
    epoch = int(entry.get("epoch", 0))
    snap = accepted_snapshot_path(epoch)
    if snap.is_file() and sha256_file(snap) == expected:
        return snap
    candidate = Path(entry["path"])
    assert_not_invalid(candidate)
    if candidate.is_file() and sha256_file(candidate) == expected:
        return candidate
    raise FileNotFoundError(
        f"accepted epoch {epoch} weights missing on disk "
        f"(expected sha256 {expected[:16]}…, looked at {snap.name} and {candidate.name})"
    )


def resolve_latest_accepted_weights() -> Path:
    last = latest_accepted()
    if last is None:
        raise RuntimeError("no accepted checkpoint in chain")
    return resolve_accepted_weights(last)


def accept_checkpoint(
    *,
    weights_path: Path,
    epoch: int,
    validation: dict[str, Any],
    ckpt_path: Path | None = None,
) -> dict[str, Any]:
    assert_not_invalid(weights_path)
    ACCEPTED_DIR.mkdir(parents=True, exist_ok=True)
    snap = accepted_snapshot_path(epoch)
    atomic_copy2(weights_path, snap)
    snap_sha = sha256_file(snap)
    if not snap_sha:
        raise RuntimeError(f"accepted snapshot write failed: {snap}")
    pt_snap: str | None = None
    ckpt = Path(ckpt_path) if ckpt_path is not None else RUN_DIR / "best.pt"
    if ckpt.is_file():
        pt_dest = accepted_pt_snapshot_path(epoch)
        atomic_copy2(ckpt, pt_dest)
        pt_snap = str(pt_dest.resolve())
    chain = load_chain()
    entry = {
        "epoch": epoch,
        "role": "streaming_accepted",
        "path": str(snap.resolve()),
        "sha256": snap_sha,
        "accepted_at": _utc_now(),
        "validation": validation,
    }
    if pt_snap:
        entry["checkpoint_path"] = pt_snap
        entry["checkpoint_sha256"] = sha256_file(Path(pt_snap))
    chain.setdefault("epochs", []).append(entry)
    save_chain(chain)
    atomic_copy2(snap, BEST_WEIGHTS)
    refresh_pool_weights_snapshot(source=snap)
    return entry


def quarantine_checkpoint(
    *,
    weights_path: Path,
    reason: str,
    validation: dict[str, Any] | None = None,
    ckpt_path: Path | None = None,
    cycle: int | None = None,
) -> dict[str, Any]:
    QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = _utc_now().replace(":", "")
    dest = QUARANTINE_DIR / f"rejected_{stamp}{weights_path.suffix}"
    if weights_path.is_file():
        atomic_copy2(weights_path, dest)
    pt_dest: Path | None = None
    ckpt = Path(ckpt_path) if ckpt_path is not None else RUN_DIR / "best.pt"
    if ckpt.is_file():
        pt_dest = QUARANTINE_DIR / f"rejected_{stamp}.pt"
        atomic_copy2(ckpt, pt_dest)
    entry = {
        "path": str(weights_path),
        "quarantine_copy": str(dest),
        "sha256": sha256_file(weights_path) if weights_path.is_file() else None,
        "reason": reason,
        "quarantined_at": _utc_now(),
        "validation": validation or {},
    }
    if pt_dest is not None:
        entry["quarantine_ckpt"] = str(pt_dest)
        entry["checkpoint_sha256"] = sha256_file(pt_dest)
    if cycle is not None:
        entry["cycle"] = cycle
    chain = load_chain()
    chain.setdefault("quarantined", []).append(entry)
    save_chain(chain)
    return entry


def restore_candidate_from_last_accepted() -> Path:
    """After quarantine, point candidate weights back to last accepted."""
    last = latest_accepted()
    if last is None:
        raise RuntimeError("no accepted checkpoint to restore")
    src = resolve_accepted_weights(last)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    atomic_copy2(src, BEST_WEIGHTS)
    refresh_pool_weights_snapshot(source=src)
    return BEST_WEIGHTS
