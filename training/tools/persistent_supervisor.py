#!/usr/bin/env python3
"""Persistent self-healing overnight supervisor for rebuild + local_game_pool + oracle_importer.

Split architecture:
  - local_game_pool.py  : local Titanium self-play games only
  - oracle_importer.py  : Oracle result imports only
  - this supervisor     : health monitoring, restart policy, auto-finalize

Health signals:
  - Local generator: recent pool_generation% commits to games.db
  - Oracle importer: recent oracle_% commits to games.db + oracle /status heartbeat
  - Never uses instantaneous titanium.exe child count as primary health signal.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import psutil

_TRAINING = Path(__file__).resolve().parents[1]
_REPO = _TRAINING.parent
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

from rebuild_checkpoint import read_checkpoint, stderr_is_deterministic
from pool_lock import find_legacy_pool_processes

LOG_DIR = _TRAINING / "data" / "overnight_logs"
STATE_PATH = LOG_DIR / "supervisor_state.json"
LOCK_PATH = LOG_DIR / "supervisor.lock"
PID_PATH = LOG_DIR / "supervisor.pid"
SUP_LOG = LOG_DIR / "supervisor.log"
REBUILD_LOG = LOG_DIR / "safe_rebuild.log"
REBUILD_ERR = LOG_DIR / "safe_rebuild_err.log"
REBUILD_PID_FILE = LOG_DIR / "safe_rebuild.pid"
POOL_PID_FILE = LOG_DIR / "continuous_pool.pid"
LOCAL_POOL_PID_FILE = LOG_DIR / "local_game_pool.pid"
ORACLE_IMPORTER_PID_FILE = LOG_DIR / "oracle_importer.pid"
POOL_LOG = LOG_DIR / "local_game_pool.log"
PAUSE_PATH = LOG_DIR / "pause_training_epochs.json"
TEMP_CACHE = _TRAINING / "data" / "feature_cache_rebuild"
LIVE_CACHE = _TRAINING / "data" / "feature_cache"
GAMES_DB = _TRAINING / "data" / "canonical" / "games.db"
TOKEN_FILE = Path(os.environ.get("LOCALAPPDATA", "")) / "titanium-oracle-api-token"

REBUILD_RESTART_DELAYS = (30, 120, 300)
MAX_REBUILD_RESTARTS_12H = 3
MAX_POOL_RESTARTS_1H = 3
POLL_SEC = 60.0
POOL_STALL_SEC = 300.0
REBUILD_STALL_SEC = 600.0

BATCH_RE = re.compile(
    r"batches=\s*(\d+)/(\d+)\s+approx_done=\s*([\d,]+)/([\d,]+)\s+written=([\d,]+)"
)

TERMINAL = frozenset({"ACTIVATED", "FAILED_DETERMINISTIC", "STOPPED_BY_USER", "FAILED_SAFE"})

# Never stop these during legacy cleanup (set at bootstrap).
_PROTECTED_PIDS: set[int] = set()


# ---------------------------------------------------------------------------
# Safe PID helpers — no bare int() on optional values
# ---------------------------------------------------------------------------

def optional_pid(value: object) -> int | None:
    """Convert any value to a valid PID (>0) or None."""
    if value is None:
        return None
    try:
        pid = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return pid if pid > 0 else None


def optional_int(value: object, default: int = 0) -> int:
    """Convert any value to int, falling back to default on failure."""
    if value is None:
        return default
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def read_pid_file(path: Path) -> int | None:
    """Read a PID file and return a valid PID or None."""
    if not path.is_file():
        return None
    try:
        return optional_pid(path.read_text(encoding="utf-8").strip())
    except OSError:
        return None


# Keep private alias for internal use (avoids touching all call sites at once).
_read_pid_file = read_pid_file


def _resolve_live_pids(state: dict) -> tuple[int | None, int | None, int | None]:
    """Return (rebuild_pid, local_pool_pid, oracle_importer_pid) from files then state."""
    rb = _read_pid_file(REBUILD_PID_FILE) or optional_pid(state.get("rebuild_pid"))
    pool = _read_pid_file(LOCAL_POOL_PID_FILE) or _read_pid_file(POOL_PID_FILE) or optional_pid(state.get("pool_pid"))
    oracle = _read_pid_file(ORACLE_IMPORTER_PID_FILE) or optional_pid(state.get("oracle_importer_pid"))
    return rb, pool, oracle


def _purge_legacy_pools() -> list[int]:
    """Stop coupled continuous_pool.py zombies; never touch protected or split services."""
    stopped: list[int] = []
    for pid, _cmd in find_legacy_pool_processes():
        if pid in _PROTECTED_PIDS:
            continue
        _log(f"Stopping legacy coupled continuous_pool pid={pid}")
        _stop_tree(pid)
        stopped.append(pid)
    return stopped


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log(msg: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{_utc_now()}] {msg}"
    print(line, flush=True)
    with SUP_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return {}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def pid_alive(pid: int | None) -> bool:
    """Return True iff the process is running and not a zombie."""
    if pid is None or pid <= 0:
        return False
    try:
        process = psutil.Process(pid)
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return False


# Private alias used throughout this module.
_pid_alive = pid_alive


def _win_proc_stats(pid: int) -> dict:
    if sys.platform != "win32" or pid <= 0:
        return {}
    ps = (
        f"$p=Get-Process -Id {pid} -EA SilentlyContinue; if (-not $p){{exit 1}}; "
        f"$kids=Get-CimInstance Win32_Process -Filter \"ParentProcessId=$($p.Id)\" -EA SilentlyContinue; "
        f"$ti=($kids|?{{$_.Name -eq 'titanium.exe'}}).Count; "
        "[pscustomobject]@{cpu=$p.CPU;ws_mb=[math]::Round($p.WorkingSet64/1MB,1);"
        "io_read_mb=[math]::Round($p.IOReadBytes/1MB,2);io_write_mb=[math]::Round($p.IOWriteBytes/1MB,2);"
        "ti_children=$ti}|ConvertTo-Json -Compress"
    )
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=20)
        if out.returncode != 0:
            return {}
        return json.loads(out.stdout.strip() or "{}")
    except Exception:
        return {}


def _disk_free_gb(path: Path) -> float | None:
    try:
        u = shutil.disk_usage(path)
        return round(u.free / (1024**3), 2)
    except Exception:
        return None


def _parse_rebuild_progress() -> dict:
    if not REBUILD_LOG.is_file():
        return {}
    for line in reversed(REBUILD_LOG.read_text(encoding="utf-8", errors="replace").splitlines()):
        m = BATCH_RE.search(line)
        if m:
            return {
                "batch_current": int(m.group(1)),
                "batch_total": int(m.group(2)),
                "approx_done": int(m.group(3).replace(",", "")),
                "rows_written": int(m.group(5).replace(",", "")),
            }
    cp = read_checkpoint(TEMP_CACHE)
    if cp:
        return {
            "batch_current": cp.last_completed_batch + 1,
            "batch_total": (cp.expected_rows + cp.batch_size - 1) // cp.batch_size,
            "approx_done": cp.next_row,
            "rows_written": cp.rows_written,
            "from_checkpoint": True,
        }
    return {}


def _games_db_info() -> tuple[int | None, str | None]:
    if not GAMES_DB.is_file():
        return None, None
    try:
        con = sqlite3.connect(f"file:{GAMES_DB}?mode=ro", uri=True)
        try:
            row = con.execute("SELECT COUNT(*), MAX(imported_at) FROM games").fetchone()
            return int(row[0]), str(row[1]) if row[1] else None
        finally:
            con.close()
    except Exception:
        return None, None


def _oracle_token() -> str | None:
    if TOKEN_FILE.is_file():
        return TOKEN_FILE.read_text(encoding="ascii").strip()
    return None


def _oracle_get(path: str, timeout: float = 10.0) -> dict | None:
    token = _oracle_token()
    if not token:
        return None
    req = urllib.request.Request(
        f"http://127.0.0.1:8765{path}",
        headers={"Authorization": f"Bearer {token}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None


def _launch_rebuild(workers: int = 4, eval_timeout: int = 900) -> int | None:
    ps1 = _TRAINING / "tools" / "start_safe_rebuild_detached.ps1"
    subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ps1),
         "-Workers", str(workers), "-EvalTimeoutSec", str(eval_timeout)],
        cwd=str(_REPO),
        timeout=120,
    )
    return read_pid_file(REBUILD_PID_FILE)


def _launch_pool() -> int | None:
    ps1 = _TRAINING / "tools" / "start_local_game_pool_detached.ps1"
    subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ps1)],
        cwd=str(_REPO),
        timeout=120,
    )
    for path in (LOCAL_POOL_PID_FILE, POOL_PID_FILE):
        pid = read_pid_file(path)
        if pid is not None:
            return pid
    return None


def _launch_oracle_importer() -> int | None:
    ps1 = _TRAINING / "tools" / "start_oracle_importer_detached.ps1"
    subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ps1)],
        cwd=str(_REPO),
        timeout=120,
    )
    return read_pid_file(ORACLE_IMPORTER_PID_FILE)


def _latest_import_ts(*, source_like: str) -> str | None:
    if not GAMES_DB.is_file():
        return None
    con = sqlite3.connect(str(GAMES_DB), timeout=10)
    try:
        row = con.execute(
            "SELECT MAX(imported_at) FROM games WHERE source LIKE ?",
            (source_like,),
        ).fetchone()
        return row[0] if row and row[0] else None
    finally:
        con.close()


def _stop_tree(pid: int | None) -> None:
    if pid is None or pid <= 0:
        return
    if sys.platform != "win32":
        return
    ps = (
        f"$procId={pid}; Stop-Process -Id $procId -Force -EA SilentlyContinue; "
        "Get-CimInstance Win32_Process -EA SilentlyContinue | "
        "Where-Object { $_.ParentProcessId -eq $procId } | "
        "ForEach-Object { Stop-Process -Id $_.ProcessId -Force -EA SilentlyContinue }"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], timeout=60)


def _stderr_fingerprint() -> str:
    if not REBUILD_ERR.is_file():
        return ""
    return REBUILD_ERR.read_text(encoding="utf-8", errors="replace")[-4000:]


def _featurization_complete() -> bool:
    """True when pass-2 featurization finished and meta.json is ready for finalize."""
    return _finalize_readiness()["ready"]


def _finalize_readiness() -> dict[str, Any]:
    """All signals required before supervisor may run --finalize-v2."""
    out: dict[str, Any] = {
        "ready": False,
        "log_marker": False,
        "meta_exists": False,
        "row_count_ok": False,
        "batch_complete": False,
        "checkpoint_cleared": False,
        "fingerprint_ok": False,
        "positions_ok": False,
    }
    text = REBUILD_LOG.read_text(encoding="utf-8", errors="replace") if REBUILD_LOG.is_file() else ""
    out["log_marker"] = "Cache ready:" in text or "Featurization complete" in text

    meta_path = TEMP_CACHE / "meta.json"
    out["meta_exists"] = meta_path.is_file()
    if not out["meta_exists"]:
        return out

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return out

    expected = optional_int(meta.get("n_total"), 0)
    out["row_count_ok"] = expected >= 1_411_376
    out["fingerprint_ok"] = bool(meta.get("engine_sha256") or meta.get("dataset_fingerprint"))

    progress = _parse_rebuild_progress()
    batch_total = optional_int(progress.get("batch_total"), 345)
    batch_current = optional_int(progress.get("batch_current"), 0)
    out["batch_complete"] = batch_current >= batch_total and batch_total > 0

    cp_path = TEMP_CACHE / "rebuild_checkpoint.json"
    out["checkpoint_cleared"] = not cp_path.is_file()
    if cp_path.is_file():
        cp = read_checkpoint(TEMP_CACHE)
        if cp and cp.last_completed_batch + 1 >= batch_total:
            out["batch_complete"] = True

    pos_path = TEMP_CACHE / "positions.bin"
    fv_len = optional_int(meta.get("fv_len"), 547)
    if pos_path.is_file() and fv_len > 0:
        rows = pos_path.stat().st_size // (fv_len * 4)
        out["positions_ok"] = rows >= expected > 0
    else:
        out["positions_ok"] = out["row_count_ok"]

    out["ready"] = (
        out["log_marker"]
        and out["meta_exists"]
        and out["row_count_ok"]
        and out["batch_complete"]
        and out["fingerprint_ok"]
        and out["positions_ok"]
        and (out["checkpoint_cleared"] or out["log_marker"])
    )
    return out


def _try_finalize() -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_TRAINING)
    env["RUSTFLAGS"] = "-C target-cpu=native"
    proc = subprocess.run(
        [sys.executable, str(_TRAINING / "safe_rebuild.py"), "--finalize-v2", "--allow-activation"],
        cwd=str(_REPO),
        capture_output=True,
        text=True,
        timeout=7200,
        env=env,
    )
    return {"returncode": proc.returncode, "stdout_tail": proc.stdout[-3000:], "stderr_tail": proc.stderr[-2000:]}


def acquire_lock() -> bool:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if LOCK_PATH.is_file():
        try:
            old = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
            if _pid_alive(optional_pid(old.get("pid"))):
                return False
        except (json.JSONDecodeError, ValueError):
            pass
    payload = {"pid": os.getpid(), "started_at": _utc_now()}
    LOCK_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    PID_PATH.write_text(str(os.getpid()), encoding="ascii")
    return True


def _prune_restart_times(times: list[str], window_hours: float) -> list[str]:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)
    out = []
    for t in times:
        try:
            if datetime.fromisoformat(t) >= cutoff:
                out.append(t)
        except ValueError:
            pass
    return out


def bootstrap_live_state(*, reason: str) -> dict[str, Any]:
    """Initialize supervisor state from live split-service PIDs."""
    global _PROTECTED_PIDS
    rb, pool, oracle = _resolve_live_pids({})
    _PROTECTED_PIDS = {p for p in (rb, pool, oracle) if p}
    games_n, games_ts = _games_db_info()
    local_ts = _latest_import_ts(source_like="pool_generation%")
    oracle_ts = _latest_import_ts(source_like="oracle_%")
    now = _utc_now()
    fin = _finalize_readiness()
    return {
        "machine_state": "RUNNING",
        "rebuild_state": "BUILD_RUNNING" if rb and _pid_alive(rb) else "STARTING",
        "pool_state": "RUNNING" if pool and _pid_alive(pool) else "STARTING",
        "oracle_importer_state": "RUNNING" if oracle and _pid_alive(oracle) else "STARTING",
        "rebuild_pid": rb or 0,
        "pool_pid": pool or 0,
        "oracle_importer_pid": oracle or 0,
        "rebuild_restart_times": [],
        "pool_restart_times": [],
        "oracle_restart_times": [],
        "restart_counter_reset_reason": reason,
        "restart_counter_reset_at": now,
        "baseline_games_db": games_n,
        "started_at": now,
        "last_progress_at": now,
        "pool_last_progress_at": now,
        "pool_last_local_progress_at": now if local_ts else now,
        "pool_last_local_import_at": local_ts,
        "oracle_last_progress_at": now if oracle_ts else now,
        "oracle_last_import_at": oracle_ts,
        "pool_last_games_db": games_n,
        "pool_last_import_at": games_ts,
        "finalize_armed": fin,
        "architecture": "split: oracle_importer.py + local_game_pool.py",
    }


def supervise(*, poll_sec: float = POLL_SEC, bootstrap_live: bool = False) -> int:
    if not acquire_lock():
        _log("Another supervisor holds the lock — exiting")
        return 0

    if bootstrap_live:
        reason = (
            "Prior pool_restart_times exhausted on false ti_children==0 stalls "
            "and legacy coupled continuous_pool.py zombies; split architecture now enforced."
        )
        state = bootstrap_live_state(reason=reason)
        _write_json(STATE_PATH, state)
        _log(f"Supervisor bootstrapped from live PIDs: rebuild={state['rebuild_pid']} "
             f"local={state['pool_pid']} oracle={state['oracle_importer_pid']}")
    else:
        state = _read_json(STATE_PATH)
        if not state:
            state = bootstrap_live_state(reason="fresh supervisor start")
            _write_json(STATE_PATH, state)

    rb0, pool0, oracle0 = _resolve_live_pids(state)
    _PROTECTED_PIDS = {p for p in (rb0, pool0, oracle0) if p}

    _log("Persistent supervisor started")

    prev_rebuild_stats: dict = {}
    prev_progress = _parse_rebuild_progress()
    if prev_progress:
        state["last_progress_at"] = _utc_now()

    poll_errors = 0
    while state.get("machine_state") not in TERMINAL:
        try:
            now = _utc_now()
            _purge_legacy_pools()

            rb_pid, pool_pid, oracle_pid = _resolve_live_pids(state)
            rb_pid = rb_pid or 0
            pool_pid = pool_pid or 0
            oracle_pid = oracle_pid or 0

            rb_alive = _pid_alive(rb_pid)
            pool_alive = _pid_alive(pool_pid)
            oracle_alive = _pid_alive(oracle_pid)
            rb_stats = _win_proc_stats(rb_pid) if rb_alive else {}
            progress = _parse_rebuild_progress()
            stderr = _stderr_fingerprint()
            games_n, games_ts = _games_db_info()
            disk_gb = _disk_free_gb(_TRAINING / "data")
            oracle_status = _oracle_get("/status")
            finalize_ready = _finalize_readiness()
            pool_ti = optional_int(_win_proc_stats(pool_pid).get("ti_children"), 0) if pool_alive else 0
            rb_ti = optional_int((rb_stats or {}).get("ti_children"), 0)

            # --- initial launch ---
            if state.get("machine_state") == "STARTING":
                if not rb_alive:
                    new_pid = _launch_rebuild()
                    state["rebuild_pid"] = new_pid or 0
                    rb_pid = state["rebuild_pid"]
                    rb_alive = _pid_alive(rb_pid)
                if not pool_alive:
                    new_pool = _launch_pool()
                    state["pool_pid"] = new_pool or 0
                    pool_pid = state["pool_pid"]
                    pool_alive = _pid_alive(pool_pid)
                if not oracle_alive:
                    new_oracle = _launch_oracle_importer()
                    state["oracle_importer_pid"] = new_oracle or 0
                    oracle_pid = state["oracle_importer_pid"]
                    oracle_alive = _pid_alive(oracle_pid)
                state["machine_state"] = "RUNNING"
                state["rebuild_state"] = "BUILD_RUNNING" if rb_alive else "BUILD_CRASHED_TRANSIENT"
                state["pool_state"] = "RUNNING" if pool_alive else "STARTING"
                state["oracle_importer_state"] = "RUNNING" if oracle_alive else "STARTING"

            # --- progress detection ---
            if progress and progress != prev_progress:
                state["last_progress_at"] = now
                prev_progress = progress

            local_import_ts = _latest_import_ts(source_like="pool_generation%")
            oracle_import_ts = _latest_import_ts(source_like="oracle_%")
            if local_import_ts and local_import_ts != state.get("pool_last_local_import_at"):
                state["pool_last_local_import_at"] = local_import_ts
                state["pool_last_local_progress_at"] = now
            if oracle_import_ts and oracle_import_ts != state.get("oracle_last_import_at"):
                state["oracle_last_import_at"] = oracle_import_ts
                state["oracle_last_progress_at"] = now
            if games_n is not None and games_n != state.get("pool_last_games_db"):
                state["pool_last_games_db"] = games_n
                state["pool_last_progress_at"] = now
                state["pool_last_import_at"] = games_ts

            # --- rebuild state machine ---
            rb_state = state.get("rebuild_state", "STARTING")
            if finalize_ready["ready"] and (TEMP_CACHE / "meta.json").is_file():
                rb_state = "FINALIZING_V2"
            elif not rb_alive:
                if stderr_is_deterministic(stderr) and "Traceback" in stderr:
                    rb_state = "BUILD_FAILED_DETERMINISTIC"
                elif state.get("rebuild_state") == "FINALIZING_V2":
                    pass
                else:
                    rb_state = "BUILD_CRASHED_TRANSIENT"
            elif rb_alive and progress:
                rb_state = "BUILD_RUNNING"
                # stall check
                last_prog = datetime.fromisoformat(state["last_progress_at"])
                if (datetime.now(timezone.utc) - last_prog).total_seconds() > REBUILD_STALL_SEC:
                    cpu_delta = 0.0
                    if prev_rebuild_stats and rb_stats:
                        cpu_delta = float(rb_stats.get("cpu", 0)) - float(prev_rebuild_stats.get("cpu", 0))
                    if cpu_delta < 0.5 and rb_ti == 0:
                        rb_state = "BUILD_STALLED"

            if rb_state == "BUILD_FAILED_DETERMINISTIC":
                state["machine_state"] = "FAILED_DETERMINISTIC"
                state["failure_reason"] = stderr[-500:]
            elif rb_state == "FINALIZING_V2" and state.get("machine_state") != "ACTIVATED":
                state["rebuild_state"] = "FINALIZING_V2"
                _log("Featurization complete — running finalize-v2")
                fin = _try_finalize()
                state["finalize_result"] = fin
                if fin["returncode"] == 0:
                    state["machine_state"] = "ACTIVATED"
                    state["rebuild_state"] = "ACTIVATED"
                    if PAUSE_PATH.is_file():
                        PAUSE_PATH.unlink(missing_ok=True)
                        _log("All gates passed — training pause cleared")
                else:
                    state["machine_state"] = "FAILED_DETERMINISTIC"
                    state["failure_reason"] = fin.get("stderr_tail", "finalize failed")
            elif rb_state in ("BUILD_CRASHED_TRANSIENT", "BUILD_STALLED") and state.get("machine_state") not in TERMINAL:
                times = _prune_restart_times(state.get("rebuild_restart_times", []), 12.0)
                if rb_state in ("BUILD_CRASHED_TRANSIENT", "BUILD_STALLED") and len(times) >= MAX_REBUILD_RESTARTS_12H:
                    state["machine_state"] = "FAILED_SAFE"
                    state["failure_reason"] = "rebuild restart limit exceeded"
                elif not rb_alive:
                    delay = REBUILD_RESTART_DELAYS[min(len(times), len(REBUILD_RESTART_DELAYS) - 1)]
                    _log(f"Rebuild restart #{len(times)+1} in {delay}s")
                    time.sleep(delay)
                    new_pid = _launch_rebuild()
                    times.append(_utc_now())
                    state["rebuild_restart_times"] = times
                    state["rebuild_pid"] = new_pid or 0
                    state["rebuild_state"] = "BUILD_RUNNING"
                    state["last_progress_at"] = _utc_now()
                elif rb_state == "STARTING" and not rb_alive and state.get("rebuild_pid", 0) == 0:
                    new_pid = _launch_rebuild()
                    state["rebuild_pid"] = new_pid or 0
                    state["rebuild_state"] = "BUILD_RUNNING"

            # --- pool health (local generator + oracle importer are separate) ---
            pool_state = state.get("pool_state", "STARTING")
            local_stall = False
            if pool_alive:
                last_local = datetime.fromisoformat(
                    state.get("pool_last_local_progress_at", state.get("pool_last_progress_at", now))
                )
                if (datetime.now(timezone.utc) - last_local).total_seconds() > POOL_STALL_SEC:
                    local_stall = True
            oracle_stall = False
            if oracle_alive:
                last_oracle = datetime.fromisoformat(
                    state.get("oracle_last_progress_at", state.get("pool_last_progress_at", now))
                )
                if (datetime.now(timezone.utc) - last_oracle).total_seconds() > max(POOL_STALL_SEC, 120):
                    oracle_stall = True

            pool_times = _prune_restart_times(state.get("pool_restart_times", []), 1.0)
            protected_pool = pool_pid in _PROTECTED_PIDS
            if (local_stall or (not pool_alive and state.get("machine_state") not in TERMINAL)) and not (
                protected_pool and pool_alive and not local_stall
            ):
                if len(pool_times) >= MAX_POOL_RESTARTS_1H:
                    state["pool_state"] = "FAILED_SAFE"
                else:
                    if pool_alive and local_stall:
                        _log(f"Local pool unhealthy (last_local={local_import_ts}) — restarting pid {pool_pid}")
                        _stop_tree(pool_pid)
                        time.sleep(3)
                    elif not pool_alive:
                        _log("Local pool not running — launching local_game_pool.py")
                    if local_stall or not pool_alive:
                        new_pool = _launch_pool()
                        pool_times.append(_utc_now())
                        state["pool_restart_times"] = pool_times
                        state["pool_pid"] = new_pool or pool_pid
                        pool_pid = state["pool_pid"]
                    state["pool_state"] = "RUNNING" if _pid_alive(pool_pid) else "STARTING"
            elif pool_alive:
                state["pool_state"] = "RUNNING"

            oracle_times = _prune_restart_times(state.get("oracle_restart_times", []), 1.0)
            if oracle_stall or (not oracle_alive and state.get("machine_state") not in TERMINAL):
                if len(oracle_times) < MAX_POOL_RESTARTS_1H:
                    if oracle_alive:
                        _log(f"Oracle importer unhealthy — restarting pid {oracle_pid}")
                        _stop_tree(oracle_pid)
                        time.sleep(2)
                    new_oracle = _launch_oracle_importer()
                    oracle_times.append(_utc_now())
                    state["oracle_restart_times"] = oracle_times
                    state["oracle_importer_pid"] = new_oracle or 0
            elif not oracle_alive:
                new_oracle = _launch_oracle_importer()
                state["oracle_importer_pid"] = new_oracle or 0

            state["oracle_importer_state"] = (
                "RUNNING" if _pid_alive(optional_pid(state.get("oracle_importer_pid")) or oracle_pid) else "STARTING"
            )

            state["rebuild_state"] = rb_state
            state["updated_at"] = now
            state["rebuild_pid"] = rb_pid
            state["pool_pid"] = pool_pid
            state["oracle_importer_pid"] = optional_pid(state.get("oracle_importer_pid")) or oracle_pid
            state["rebuild_progress"] = progress
            state["rebuild_stats"] = rb_stats
            state["pool_titanium_children"] = pool_ti
            state["games_db_count"] = games_n
            state["games_db_latest_import"] = games_ts
            state["oracle_status_ok"] = oracle_status is not None
            state["oracle_workers"] = optional_int((oracle_status or {}).get("workers_configured"), 0) or None
            state["oracle_completed"] = optional_int((oracle_status or {}).get("completed"), 0) or None
            state["finalize_armed"] = finalize_ready
            state["disk_free_gb"] = disk_gb
            state["training_paused"] = PAUSE_PATH.is_file()
            state["stderr_fingerprint"] = stderr[-200:] if stderr else None
            state["last_error"] = None
            poll_errors = 0
            prev_rebuild_stats = dict(rb_stats)

            _write_json(STATE_PATH, state)
            report = {
                "supervision_status": state.get("machine_state", "RUNNING"),
                "updated_at": now,
                "supervisor_state": state,
            }
            _write_json(LOG_DIR / "overnight_status_report.json", report)

        except Exception as exc:
            poll_errors += 1
            _log(f"supervisor loop error: {exc}\n{traceback.format_exc()}")
            state["last_error"] = str(exc)
            state["poll_error_count"] = poll_errors
            _write_json(STATE_PATH, state)

        time.sleep(poll_sec)

    _write_json(STATE_PATH, state)
    _log(f"Supervisor terminal state: {state.get('machine_state')}")
    return 0 if state.get("machine_state") == "ACTIVATED" else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--poll-sec", type=float, default=POLL_SEC)
    ap.add_argument("--guardian-check", action="store_true", help="Used by guardian script (no-op)")
    ap.add_argument(
        "--bootstrap-live",
        action="store_true",
        help="Reset supervisor state from live rebuild/local/oracle PIDs",
    )
    args = ap.parse_args()

    if args.guardian_check:
        return 0

    return supervise(poll_sec=args.poll_sec, bootstrap_live=args.bootstrap_live)


if __name__ == "__main__":
    raise SystemExit(main())
