#!/usr/bin/env python3
"""Light overnight watchdog: restart split local pool if commits stall."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_TRAINING = Path(__file__).resolve().parents[1]
_REPO = _TRAINING.parent
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

LOG_DIR = _TRAINING / "data" / "overnight_logs"
LOCAL_POOL_PID_FILE = LOG_DIR / "local_game_pool.pid"
GAMES_DB = _TRAINING / "data" / "canonical" / "games.db"
NOTE_PATH = LOG_DIR / "overnight_handoff_notes.jsonl"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _optional_pid(value) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _pid_alive(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    if sys.platform == "win32":
        try:
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            return str(pid) in out.stdout
        except Exception:
            return False
    try:
        import os

        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _latest_local_import_ts() -> str | None:
    try:
        import sqlite3

        con = sqlite3.connect(f"file:{GAMES_DB}?mode=ro", uri=True)
        try:
            row = con.execute(
                "SELECT MAX(imported_at) FROM games WHERE source LIKE 'pool_generation%'"
            ).fetchone()
            return row[0] if row and row[0] else None
        finally:
            con.close()
    except Exception:
        return None


def _append_note(payload: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with NOTE_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, separators=(",", ":")) + "\n")


def _restart_local_pool() -> int | None:
    ps1 = _TRAINING / "tools" / "start_local_game_pool_detached.ps1"
    subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ps1)],
        cwd=str(_REPO),
        timeout=120,
    )
    if LOCAL_POOL_PID_FILE.is_file():
        return _optional_pid(LOCAL_POOL_PID_FILE.read_text(encoding="ascii").strip())
    return None


def _stop_pool_tree(pid: int) -> None:
    if sys.platform != "win32" or pid <= 0:
        return
    ps = (
        f"$procId={pid}; "
        "Stop-Process -Id $procId -Force -EA SilentlyContinue; "
        "Get-CimInstance Win32_Process -EA SilentlyContinue | "
        "Where-Object { $_.ParentProcessId -eq $procId } | "
        "ForEach-Object { Stop-Process -Id $_.ProcessId -Force -EA SilentlyContinue }"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], timeout=60)


def watch(*, poll_sec: float, pool_stall_sec: float) -> int:
    last_local_ts = _latest_local_import_ts()
    last_progress_at = time.monotonic()
    pool_pid = _optional_pid(
        LOCAL_POOL_PID_FILE.read_text(encoding="ascii").strip()
        if LOCAL_POOL_PID_FILE.is_file()
        else None
    )

    while True:
        try:
            subprocess.run(
                [sys.executable, str(_TRAINING / "tools" / "kill_legacy_pool_processes.py")],
                cwd=str(_REPO),
                timeout=60,
                env={**dict(__import__("os").environ), "PYTHONPATH": str(_TRAINING)},
            )

            if LOCAL_POOL_PID_FILE.is_file():
                pool_pid = _optional_pid(LOCAL_POOL_PID_FILE.read_text(encoding="ascii").strip())

            local_ts = _latest_local_import_ts()
            pool_alive = _pid_alive(pool_pid)

            if local_ts and local_ts != last_local_ts:
                last_local_ts = local_ts
                last_progress_at = time.monotonic()

            stalled = pool_alive and (time.monotonic() - last_progress_at) >= pool_stall_sec

            _append_note({
                "ts": _utc_now(),
                "pool_pid": pool_pid,
                "pool_alive": pool_alive,
                "latest_local_import": local_ts,
                "stalled": stalled,
            })

            if not pool_alive:
                new_pid = _restart_local_pool()
                _append_note({"ts": _utc_now(), "action": "local_pool_restart_dead", "new_pid": new_pid})
                last_progress_at = time.monotonic()
            elif stalled:
                _append_note({"ts": _utc_now(), "action": "local_pool_restart_stalled", "old_pid": pool_pid})
                if pool_pid:
                    _stop_pool_tree(pool_pid)
                time.sleep(3)
                new_pid = _restart_local_pool()
                _append_note({"ts": _utc_now(), "action": "local_pool_restarted", "new_pid": new_pid})
                last_progress_at = time.monotonic()

        except Exception as exc:
            _append_note({"ts": _utc_now(), "watchdog_error": str(exc)})

        time.sleep(poll_sec)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--poll-sec", type=float, default=120.0)
    ap.add_argument("--pool-stall-sec", type=float, default=600.0)
    args = ap.parse_args()
    watch(poll_sec=args.poll_sec, pool_stall_sec=args.pool_stall_sec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
