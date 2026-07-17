#!/usr/bin/env python3
"""Overnight pipeline supervisor — monitor only, no feature changes."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

_TRAINING = Path(__file__).resolve().parents[1]
_REPO = _TRAINING.parent
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

LOG_DIR = _TRAINING / "data" / "overnight_logs"
REPORT_PATH = LOG_DIR / "overnight_status_report.json"
STATE_PATH = LOG_DIR / "overnight_supervisor_state.json"
REBUILD_LOG = LOG_DIR / "safe_rebuild.log"
REBUILD_ERR = LOG_DIR / "safe_rebuild_err.log"
PAUSE_PATH = LOG_DIR / "pause_training_epochs.json"
LIVE_CACHE = _TRAINING / "data" / "feature_cache"
TEMP_CACHE = _TRAINING / "data" / "feature_cache_rebuild"
FINAL_REPORT = LOG_DIR / "safe_rebuild_report.json"
WATCHER_STATE = LOG_DIR / "safe_rebuild_watcher_state.json"
POOL_LOG = LOG_DIR / "continuous_pool.log"
GAMES_DB = _TRAINING / "data" / "canonical" / "games.db"

BATCH_RE = re.compile(
    r"batches=\s*(\d+)/(\d+)\s+approx_done=\s*([\d,]+)/([\d,]+)\s+written=([\d,]+)\s+failed=(\d+)\s+([\d.]+) pos/s\s+ETA\s+([\d.]+)s"
)
FEATURIZATION_DONE = (
    "Featurization complete",
    "Cache ready:",
    "Initial build elapsed",
)

DEFAULT_REBUILD_PID = 8068
DEFAULT_WATCHER_PID = 2424
DEFAULT_POOL_PID = 4812


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
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
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _disk_free_gb(path: Path) -> float | None:
    try:
        usage = shutil.disk_usage(path)
        return round(usage.free / (1024**3), 2)
    except Exception:
        return None


def _process_memory_mb(pid: int) -> float | None:
    if sys.platform != "win32" or pid <= 0:
        return None
    ps = (
        f"(Get-Process -Id {pid} -ErrorAction SilentlyContinue | "
        "Select-Object -ExpandProperty WorkingSet64)"
    )
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            capture_output=True,
            text=True,
            timeout=15,
        )
        val = out.stdout.strip()
        if val.isdigit():
            return round(int(val) / (1024 * 1024), 1)
    except Exception:
        pass
    return None


def _parse_rebuild_progress(log_text: str) -> dict:
    for line in reversed(log_text.splitlines()):
        m = BATCH_RE.search(line)
        if m:
            return {
                "batch_current": int(m.group(1)),
                "batch_total": int(m.group(2)),
                "approx_done": int(m.group(3).replace(",", "")),
                "target_rows": int(m.group(4).replace(",", "")),
                "rows_written": int(m.group(5).replace(",", "")),
                "failed": int(m.group(6)),
                "throughput_pos_s": float(m.group(7)),
                "eta_sec": float(m.group(8)),
            }
    return {}


def _featurization_complete(log_text: str) -> bool:
    return any(marker in log_text for marker in FEATURIZATION_DONE)


def _count_oracle_imports(log_text: str, since_marker: str | None) -> int:
    count = 0
    for line in log_text.splitlines():
        if "[oracle] imported" not in line:
            continue
        count += 1
    return count


def _games_db_count() -> int | None:
    if not GAMES_DB.is_file():
        return None
    try:
        import sqlite3

        con = sqlite3.connect(f"file:{GAMES_DB}?mode=ro", uri=True)
        try:
            return int(con.execute("SELECT COUNT(*) FROM games").fetchone()[0])
        finally:
            con.close()
    except Exception:
        return None


def _live_cache_rows() -> int | None:
    meta = LIVE_CACHE / "meta.json"
    if not meta.is_file():
        return None
    try:
        return int(_read_json(meta).get("n_total", -1))
    except Exception:
        return None


def _activation_gates(report: dict) -> dict:
    cache_fin = report.get("cache_finalize") or {}
    prefix = report.get("prefix") or {}
    validation = cache_fin.get("validation") or {}
    prefix_val = prefix.get("validation") or {}
    opening_gate = (LOG_DIR / "opening_exploration_enabled.json").is_file()
    return {
        "teacher_audit_ok": validation.get("teacher_audit_ok"),
        "expected_rows_ok": validation.get("expected_rows_ok"),
        "parity_samples_ok": validation.get("parity_samples_ok"),
        "row_delta_integrity_ok": validation.get("row_delta_integrity_ok"),
        "cache_activated": cache_fin.get("activated"),
        "prefix_activated": prefix.get("activated"),
        "opening_exploration_gate": opening_gate,
        "validation_passed": validation.get("passed"),
        "prefix_validation_passed": prefix_val.get("passed"),
        "all_gates": bool(
            cache_fin.get("activated")
            and prefix.get("activated")
            and opening_gate
            and validation.get("passed")
            and prefix_val.get("passed")
            and validation.get("row_delta_integrity_ok") is True
        ),
    }


def _restart_rebuild_once(state: dict) -> dict:
    if state.get("rebuild_restart_used"):
        return {"skipped": True, "reason": "single restart already used"}
    ps1 = _TRAINING / "tools" / "start_safe_rebuild_detached.ps1"
    proc = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ps1), "-Workers", "4"],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(_REPO),
    )
    new_pid = None
    pid_file = LOG_DIR / "safe_rebuild.pid"
    if pid_file.is_file():
        try:
            new_pid = int(pid_file.read_text(encoding="ascii").strip())
        except ValueError:
            pass
    state["rebuild_restart_used"] = True
    state["rebuild_restart_at"] = _utc_now()
    state["rebuild_restart_new_pid"] = new_pid
    return {
        "restarted": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-2000:],
        "stderr": proc.stderr[-2000:],
        "new_pid": new_pid,
    }


def _record_failure(state: dict, *, reason: str, exc: str | None = None) -> None:
    log_tail = REBUILD_LOG.read_text(encoding="utf-8", errors="replace") if REBUILD_LOG.is_file() else ""
    err_tail = REBUILD_ERR.read_text(encoding="utf-8", errors="replace") if REBUILD_ERR.is_file() else ""
    progress = _parse_rebuild_progress(log_tail)
    payload = {
        "supervision_status": "FAILED",
        "failure_reason": reason,
        "failure_at": _utc_now(),
        "exception": exc,
        "rebuild_progress": progress,
        "disk_free_gb": _disk_free_gb(_TRAINING / "data"),
        "processes": state.get("last_process_check", {}),
        "rebuild_log_tail": log_tail[-4000:],
        "rebuild_err_tail": err_tail[-2000:],
        "training_paused": PAUSE_PATH.is_file(),
        "live_cache_rows": _live_cache_rows(),
        "morning_summary": _build_morning_summary(state, failed=True, reason=reason),
    }
    existing = _read_json(REPORT_PATH)
    existing.update(payload)
    _write_json(REPORT_PATH, existing)
    state["status"] = "failed"
    state["failure"] = reason
    _write_json(STATE_PATH, state)


def _build_morning_summary(state: dict, *, failed: bool = False, reason: str | None = None) -> dict:
    final = _read_json(FINAL_REPORT)
    watcher = _read_json(WATCHER_STATE)
    gates = _activation_gates(final) if final else {}
    games_now = _games_db_count()
    games_start = state.get("baseline_games_db_count")
    new_games = (games_now - games_start) if games_now is not None and games_start is not None else None
    pool_log = POOL_LOG.read_text(encoding="utf-8", errors="replace") if POOL_LOG.is_file() else ""
    return {
        "rebuild_success": final.get("status") == "SUCCESS" and not failed,
        "rebuild_failure": failed or final.get("status") == "FAILED",
        "failure_reason": reason or final.get("failure"),
        "final_cache_row_count": _live_cache_rows() if gates.get("cache_activated") else None,
        "temp_cache_row_count": _read_json(TEMP_CACHE / "meta.json").get("n_total") if (TEMP_CACHE / "meta.json").is_file() else None,
        "validation_result": final.get("cache_finalize", {}).get("validation"),
        "prefix_index_result": final.get("prefix"),
        "training_paused": PAUSE_PATH.is_file(),
        "training_resumed": not PAUSE_PATH.is_file() and gates.get("all_gates"),
        "new_games_imported_overnight": new_games,
        "oracle_import_lines": _count_oracle_imports(pool_log, None),
        "oracle_worker_count": None,
        "oracle_games_per_hour": None,
        "candidate_trained": None,
        "promotion_result": None,
        "manual_attention": state.get("manual_attention", []),
        "watcher_status": watcher.get("status"),
        "gates": gates,
    }


def supervise(
    *,
    rebuild_pid: int,
    watcher_pid: int,
    pool_pid: int,
    poll_sec: float = 60.0,
) -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    state = _read_json(STATE_PATH)
    if not state:
        state = {
            "started_at": _utc_now(),
            "rebuild_restart_used": False,
            "baseline_games_db_count": _games_db_count(),
            "baseline_oracle_imports": _count_oracle_imports(
                POOL_LOG.read_text(encoding="utf-8", errors="replace") if POOL_LOG.is_file() else "",
                None,
            ),
            "expected_rebuild_pid": rebuild_pid,
            "expected_watcher_pid": watcher_pid,
            "expected_pool_pid": pool_pid,
            "manual_attention": [],
        }
        _write_json(STATE_PATH, state)

    while True:
        try:
            # Resolve PIDs from pid files (or defaults)
            rb_pid = rebuild_pid
            if state.get("rebuild_restart_new_pid"):
                rb_pid = int(state["rebuild_restart_new_pid"])
            else:
                pid_file = LOG_DIR / "safe_rebuild.pid"
                if pid_file.is_file():
                    try:
                        rb_pid = int(pid_file.read_text(encoding="ascii").strip())
                    except ValueError:
                        pass

            w_pid = watcher_pid
            wfile = LOG_DIR / "safe_rebuild_watcher.pid"
            if wfile.is_file():
                try:
                    w_pid = int(wfile.read_text(encoding="ascii").strip())
                except ValueError:
                    pass

            p_pid = pool_pid
            log_text = REBUILD_LOG.read_text(encoding="utf-8", errors="replace") if REBUILD_LOG.is_file() else ""
            progress = _parse_rebuild_progress(log_text)
            complete = _featurization_complete(log_text)
            rb_alive = _pid_alive(rb_pid)
            w_alive = _pid_alive(w_pid)
            p_alive = _pid_alive(p_pid)

            proc_check = {
                "rebuild_pid": rb_pid,
                "rebuild_alive": rb_alive,
                "rebuild_memory_mb": _process_memory_mb(rb_pid),
                "watcher_pid": w_pid,
                "watcher_alive": w_alive,
                "pool_pid": p_pid,
                "pool_alive": p_alive,
                "checked_at": _utc_now(),
            }
            state["last_process_check"] = proc_check

            if not p_alive:
                state.setdefault("manual_attention", [])
                msg = f"continuous pool PID {p_pid} not alive"
                if msg not in state["manual_attention"]:
                    state["manual_attention"].append(msg)

            if not w_alive and not complete:
                _record_failure(state, reason=f"watcher PID {w_pid} died before featurization completed")
                return 1

            if not rb_alive and not complete:
                restart = _restart_rebuild_once(state)
                state["last_rebuild_restart"] = restart
                if restart.get("restarted") and restart.get("new_pid"):
                    rebuild_pid = int(restart["new_pid"])
                    state["manual_attention"].append(
                        f"rebuild restarted once: old={rb_pid} new={restart['new_pid']}"
                    )
                else:
                    _record_failure(
                        state,
                        reason=f"rebuild PID {rb_pid} died before completion; restart={restart}",
                    )
                    return 1

            if not rb_alive and complete:
                # Expected after watcher intercept; watcher handles finalize
                pass

            games_count = _games_db_count()
            pool_log = POOL_LOG.read_text(encoding="utf-8", errors="replace") if POOL_LOG.is_file() else ""
            oracle_imports = _count_oracle_imports(pool_log, None)
            new_oracle = oracle_imports - int(state.get("baseline_oracle_imports", 0))

            final_report = _read_json(FINAL_REPORT)
            watcher_state = _read_json(WATCHER_STATE)
            gates = _activation_gates(final_report) if final_report else {}

            report = {
                "supervision_status": "RUNNING",
                "updated_at": _utc_now(),
                "supervisor_started_at": state.get("started_at"),
                "processes": proc_check,
                "rebuild_progress": progress,
                "featurization_complete": complete,
                "disk_free_gb": _disk_free_gb(_TRAINING / "data"),
                "training_paused": PAUSE_PATH.is_file(),
                "live_cache_rows": _live_cache_rows(),
                "temp_meta_present": (TEMP_CACHE / "meta.json").is_file(),
                "games_db_count": games_count,
                "new_games_since_supervisor_start": (
                    games_count - state["baseline_games_db_count"]
                    if games_count is not None and state.get("baseline_games_db_count") is not None
                    else None
                ),
                "oracle_imports_total": oracle_imports,
                "oracle_imports_since_start": new_oracle,
                "pool_alive": p_alive,
                "watcher_state": watcher_state,
                "finalize_report_status": final_report.get("status"),
                "activation_gates": gates,
                "rebuild_restart_used": state.get("rebuild_restart_used", False),
                "morning_summary": _build_morning_summary(state),
                "constraints": {
                    "no_cat_deploy": True,
                    "no_oracle_changes": True,
                    "no_weight_upload_unless_hash_changes": True,
                    "max_rebuild_restarts": 1,
                },
            }
            _write_json(REPORT_PATH, report)
            _write_json(STATE_PATH, state)

            if final_report.get("status") == "SUCCESS" and gates.get("all_gates"):
                report["supervision_status"] = "COMPLETE"
                report["morning_summary"] = _build_morning_summary(state)
                _write_json(REPORT_PATH, report)
                return 0

            if final_report.get("status") == "FAILED":
                _record_failure(state, reason=final_report.get("failure", "finalize failed"))
                return 1

            if watcher_state.get("status") in ("verify_failed", "missed_intercept", "failed"):
                _record_failure(state, reason=f"watcher status={watcher_state.get('status')}")
                return 1

        except Exception as exc:
            _record_failure(state, reason="supervisor exception", exc=traceback.format_exc())
            return 1

        time.sleep(poll_sec)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rebuild-pid", type=int, default=DEFAULT_REBUILD_PID)
    ap.add_argument("--watcher-pid", type=int, default=DEFAULT_WATCHER_PID)
    ap.add_argument("--pool-pid", type=int, default=DEFAULT_POOL_PID)
    ap.add_argument("--poll-sec", type=float, default=60.0)
    args = ap.parse_args()
    return supervise(
        rebuild_pid=args.rebuild_pid,
        watcher_pid=args.watcher_pid,
        pool_pid=args.pool_pid,
        poll_sec=args.poll_sec,
    )


if __name__ == "__main__":
    raise SystemExit(main())
