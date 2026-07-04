#!/usr/bin/env python3
"""Exclusive single-instance lock for continuous_pool.py (Windows + POSIX)."""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_LOG_DIR = _REPO / "training" / "data" / "overnight_logs"
LOCK_PATH = _LOG_DIR / "continuous_pool.lock.json"
LOCAL_GAME_POOL_LOCK_PATH = _LOG_DIR / "local_game_pool.lock.json"
ORACLE_IMPORTER_LOCK_PATH = _LOG_DIR / "oracle_importer.lock.json"
# Canonical single-writer lock for anything that spawns trainer.py and touches
# RUN_DIR / net_weights_best.bin / engine/src/titanium/net_weights.bin.
# Every training trigger site (continuous_pool.py, training_coordinator.py,
# and any future one) MUST hold this before spawning trainer.py or running
# the promotion/deploy gate. Non-blocking: a busy lock means "skip this cycle,
# retry next trigger" — never queue or wait, so game-generation loops never
# stall on a training run owned by a different process.
TRAINER_LOCK_PATH = _LOG_DIR / "trainer_run.lock.json"


@dataclass
class PoolLockInfo:
    pid: int
    started_at: str
    repo: str
    command_line: str


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _process_command_line(pid: int) -> str | None:
    if pid <= 0:
        return None
    if sys.platform == "win32":
        import subprocess

        ps = (
            f"Get-CimInstance Win32_Process -Filter 'ProcessId={pid}' | "
            "Select-Object -ExpandProperty CommandLine"
        )
        try:
            return subprocess.check_output(
                ["powershell", "-NoProfile", "-Command", ps],
                text=True,
                timeout=10,
            ).strip() or None
        except Exception:
            return None
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode(errors="replace")
    except Exception:
        return None


def _lock_owner_alive(existing: dict) -> bool:
    old_pid = int(existing.get("pid", 0))
    if not _pid_alive(old_pid):
        return False
    current_cmd = _process_command_line(old_pid) or ""
    recorded_cmd = str(existing.get("command_line") or "")
    for script_name in (
        "continuous_pool.py",
        "local_game_pool.py",
        "oracle_importer.py",
        "training_coordinator.py",
    ):
        if script_name in recorded_cmd:
            return script_name in current_cmd
    return bool(current_cmd)


def _read_lock(lock_path: Path) -> dict | None:
    if not lock_path.is_file():
        return None
    try:
        return json.loads(lock_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_lock(lock_path: Path, info: PoolLockInfo) -> None:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(
        json.dumps(
            {
                "pid": info.pid,
                "started_at": info.started_at,
                "repo": info.repo,
                "command_line": info.command_line,
                "lock_id": f"{info.pid}@{info.started_at}",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def find_processes_by_command(*patterns: str) -> list[tuple[int, str]]:
    """Return (pid, command_line) for live processes matching any pattern."""
    hits: list[tuple[int, str]] = []
    if sys.platform == "win32":
        import subprocess

        ps = (
            "Get-CimInstance Win32_Process | "
            "Where-Object { $_.CommandLine } | "
            "Select-Object ProcessId, CommandLine | ConvertTo-Json -Compress"
        )
        try:
            raw = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command", ps],
                text=True,
                timeout=30,
            ).strip()
        except Exception:
            return hits
        if not raw:
            return hits
        rows = json.loads(raw)
        if isinstance(rows, dict):
            rows = [rows]
        for row in rows:
            cmd = str(row.get("CommandLine") or "")
            if not cmd:
                continue
            if any(p in cmd for p in patterns):
                hits.append((int(row["ProcessId"]), cmd))
        return hits
    try:
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            try:
                cmd = (entry / "cmdline").read_bytes().replace(b"\x00", b" ").decode(errors="replace")
            except Exception:
                continue
            if cmd and any(p in cmd for p in patterns):
                hits.append((pid, cmd))
    except Exception:
        pass
    return hits


def find_legacy_pool_processes(*, exclude_pid: int | None = None) -> list[tuple[int, str]]:
    """Processes running coupled continuous_pool (generator + oracle)."""
    out: list[tuple[int, str]] = []
    for pid, cmd in find_processes_by_command("continuous_pool.py"):
        if exclude_pid and pid == exclude_pid:
            continue
        if "oracle_importer.py" in cmd or "local_game_pool.py" in cmd:
            continue
        out.append((pid, cmd))
    return out


def acquire_pool_lock(
    *,
    repo: Path | None = None,
    command_line: str | None = None,
    lock_path: Path | None = None,
) -> PoolLockInfo:
    """Acquire exclusive pool lock or raise RuntimeError."""
    repo = (repo or _REPO).resolve()
    lock_path = (lock_path or LOCK_PATH).resolve()
    existing = _read_lock(lock_path)
    if existing:
        old_pid = int(existing.get("pid", 0))
        if _lock_owner_alive(existing):
            raise RuntimeError(
                "Another instance is already running:\n"
                f"  pid={old_pid}\n"
                f"  started={existing.get('started_at')}\n"
                f"  repo={existing.get('repo')}\n"
                f"  cmd={existing.get('command_line')}\n"
                f"  lock={lock_path}"
            )
        try:
            lock_path.unlink(missing_ok=True)
        except Exception:
            pass

    info = PoolLockInfo(
        pid=os.getpid(),
        started_at=datetime.now(timezone.utc).isoformat(),
        repo=str(repo),
        command_line=command_line or " ".join(sys.argv),
    )
    _write_lock(lock_path, info)
    return info


def release_pool_lock(lock_path: Path | None = None) -> None:
    lock_path = (lock_path or LOCK_PATH).resolve()
    existing = _read_lock(lock_path)
    if existing and int(existing.get("pid", -1)) == os.getpid():
        try:
            lock_path.unlink(missing_ok=True)
        except Exception:
            pass


class PoolInstanceLock:
    def __init__(self, *, repo: Path | None = None, lock_path: Path | None = None):
        self._info: PoolLockInfo | None = None
        self._repo = repo
        self._lock_path = (lock_path or LOCK_PATH).resolve()

    def __enter__(self) -> PoolLockInfo:
        self._info = acquire_pool_lock(repo=self._repo, lock_path=self._lock_path)
        return self._info

    def __exit__(self, exc_type, exc, tb) -> None:
        release_pool_lock(self._lock_path)


def try_acquire_lock(
    *,
    lock_path: Path,
    repo: Path | None = None,
    command_line: str | None = None,
) -> PoolLockInfo | None:
    """Non-blocking variant of `acquire_pool_lock`: returns None instead of
    raising when another live process already holds the lock."""
    try:
        return acquire_pool_lock(repo=repo, command_line=command_line, lock_path=lock_path)
    except RuntimeError:
        return None


class TrainerRunLock:
    """Cross-process mutual exclusion around any trainer.py spawn + promotion
    gate. `__enter__` returns None (not an exception) if another process
    already holds it — the caller MUST treat that as "skip this training
    cycle, try again next trigger," never block waiting for it.

    Usage:
        with TrainerRunLock() as lock:
            if lock is None:
                log("training lock held elsewhere — skipping this cycle")
                return False
            ...spawn trainer.py, run promotion gate...
    """

    def __init__(self, *, lock_path: Path = TRAINER_LOCK_PATH):
        self._lock_path = lock_path.resolve()
        self._acquired = False

    def __enter__(self) -> PoolLockInfo | None:
        info = try_acquire_lock(lock_path=self._lock_path)
        self._acquired = info is not None
        return info

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._acquired:
            release_pool_lock(self._lock_path)


def trainer_lock_held() -> bool:
    """True when another process holds TrainerRunLock (training/validation)."""
    existing = _read_lock(TRAINER_LOCK_PATH)
    if not existing:
        return False
    return _lock_owner_alive(existing)


# Cross-process lock around the teacher-dataset parquet read-modify-write.
# continuous_pool.py (local self-play, N worker threads in ONE process) and
# oracle_importer.py (a SEPARATE OS process) both call sync_single_game() on
# the exact same part-00000.parquet files. continuous_pool.py's own
# threading.Lock only serializes its own threads -- it does nothing across
# processes. Two processes doing read-modify-write on one parquet file with
# no cross-process synchronization is exactly how you get a truncated /
# corrupted parquet ("Page was smaller than expected" on the next read).
TEACHER_SYNC_LOCK_PATH = _LOG_DIR / "teacher_sync.lock.json"


class TeacherSyncLock:
    """Blocking cross-process mutual exclusion around one parquet
    read-modify-write. Unlike TrainerRunLock (skip-if-busy), callers here
    must wait their turn -- skipping would silently drop the game's positions
    instead of just delaying them.

    Usage:
        with TeacherSyncLock():
            ...read parquet, append rows, write parquet...
    """

    def __init__(self, *, lock_path: Path = TEACHER_SYNC_LOCK_PATH, timeout_sec: float = 120.0):
        self._lock_path = lock_path.resolve()
        self._timeout_sec = timeout_sec
        self._acquired = False

    def __enter__(self) -> "TeacherSyncLock":
        import time

        deadline = time.monotonic() + self._timeout_sec
        while True:
            info = try_acquire_lock(lock_path=self._lock_path)
            if info is not None:
                self._acquired = True
                return self
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"TeacherSyncLock: timed out after {self._timeout_sec}s waiting for "
                    f"{self._lock_path} (another process holding it too long, or stuck)"
                )
            time.sleep(0.2)

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._acquired:
            release_pool_lock(self._lock_path)
