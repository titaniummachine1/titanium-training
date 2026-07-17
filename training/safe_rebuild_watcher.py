#!/usr/bin/env python3
"""Intercept v1 safe_rebuild activation and hand off to protocol v2 finalize.

Monitors the rebuild log and v1 parent PID. When featurization completes,
terminates the v1 process before it can run weak validation / swap, then runs
``safe_rebuild.py --finalize-v2 --allow-activation``.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_TRAINING = Path(__file__).resolve().parent
_REPO = _TRAINING.parent
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

from build_feature_cache import FV_LEN, check_fingerprint
from safe_rebuild import (
    LIVE_CACHE,
    LOG_DIR,
    OPENING_ENABLED_PATH,
    PAUSE_EPOCHS_PATH,
    PROTOCOL_VERSION,
    TEMP_CACHE,
    finalize_cache_v2,
    run_prefix_phase,
)
from titanium_training.paths import DATA_DIR

WATCHER_LOG = LOG_DIR / "safe_rebuild_watcher_internal.log"
WATCHER_STATE = LOG_DIR / "safe_rebuild_watcher_state.json"
REBUILD_LOG = LOG_DIR / "safe_rebuild.log"
FINAL_REPORT = LOG_DIR / "safe_rebuild_report.json"

# Markers emitted by v1 in-flight process (build_feature_cache + old run_cache_phase).
FEATURIZATION_DONE_MARKERS = (
    "Initial build elapsed",
    "Cache ready:",
)
# If v1 reaches these, interception failed.
V1_PAST_SAFE_POINT = (
    "CACHE VALIDATION FAILED",
    "CACHE NOT ACTIVATED",
    "=== Phase 2:",
    "=== SAFE REBUILD COMPLETE ===",
)

POLL_SEC = 3.0
DEFAULT_V1_PID = 0  # 0 = auto-detect running safe_rebuild.py process


def _find_rebuild_pid() -> int | None:
    """Return PID of active ``safe_rebuild.py`` build (not watcher/finalize-only)."""
    if sys.platform == "win32":
        ps = (
            "Get-CimInstance Win32_Process | "
            "Where-Object { $_.CommandLine -match 'safe_rebuild\\.py' "
            "-and $_.CommandLine -notmatch 'watcher' "
            "-and $_.CommandLine -notmatch 'finalize-v2' } | "
            "Select-Object -ExpandProperty ProcessId"
        )
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps],
                capture_output=True,
                text=True,
                timeout=30,
            )
            pids = [int(x.strip()) for x in out.stdout.splitlines() if x.strip().isdigit()]
            return pids[0] if pids else None
        except Exception:
            return None
    try:
        out = subprocess.run(
            ["ps", "-eo", "pid,args"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except Exception:
        return None
    for line in out.stdout.splitlines():
        if "safe_rebuild.py" not in line or "watcher" in line or "finalize-v2" in line:
            continue
        parts = line.strip().split(None, 1)
        if parts:
            try:
                return int(parts[0])
            except ValueError:
                continue
    return None
MIN_REBUILD_ROWS = 1_400_000
MAX_LIVE_ROWS = 50_000


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log(msg: str) -> None:
    line = f"[{_utc_now()}] {msg}"
    print(line, flush=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with WATCHER_LOG.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def _write_state(payload: dict) -> None:
    payload["updated_at"] = _utc_now()
    WATCHER_STATE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


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


def _read_log_tail(path: Path, max_bytes: int = 256_000) -> str:
    if not path.is_file():
        return ""
    data = path.read_bytes()
    if len(data) > max_bytes:
        data = data[-max_bytes:]
    return data.decode(errors="replace")


def _find_v1_child_pids(parent_pid: int) -> list[int]:
    """Return python.exe child PIDs of parent on Windows via wmic."""
    try:
        out = subprocess.run(
            [
                "wmic",
                "process",
                "where",
                f"(ParentProcessId={parent_pid})",
                "get",
                "ProcessId,Name",
                "/format:csv",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception:
        return []
    pids: list[int] = []
    for line in out.stdout.splitlines():
        if "python" not in line.lower():
            continue
        parts = [p.strip() for p in line.split(",") if p.strip().isdigit()]
        for p in parts:
            try:
                pids.append(int(p))
            except ValueError:
                pass
    return pids


def _terminate_tree(pid: int, *, grace_sec: float = 5.0) -> dict:
    """Terminate parent and python children."""
    killed: list[int] = []
    errors: list[str] = []

    child_pids = _find_v1_child_pids(pid)
    targets = child_pids + [pid]
    for tp in targets:
        if not _pid_alive(tp):
            continue
        try:
            subprocess.run(
                ["taskkill", "/PID", str(tp), "/T", "/F"],
                capture_output=True,
                timeout=30,
            )
            killed.append(tp)
        except Exception as exc:
            errors.append(f"taskkill {tp}: {exc}")

    deadline = time.monotonic() + grace_sec
    while time.monotonic() < deadline:
        if not _pid_alive(pid) and not any(_pid_alive(c) for c in child_pids):
            break
        time.sleep(0.25)

    remaining = [p for p in targets if _pid_alive(p)]
    return {"killed": killed, "errors": errors, "remaining_pids": remaining}


def _live_cache_rows() -> int | None:
    meta = LIVE_CACHE / "meta.json"
    if not meta.is_file():
        return None
    try:
        return int(json.loads(meta.read_text(encoding="utf-8")).get("n_total", -1))
    except Exception:
        return None


def _temp_cache_complete() -> tuple[bool, str, dict]:
    meta_path = TEMP_CACHE / "meta.json"
    if not meta_path.is_file():
        return False, "temp meta.json missing", {}
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"meta unreadable: {exc}", {}
    n_total = int(meta.get("n_total", 0))
    if n_total < MIN_REBUILD_ROWS:
        return False, f"n_total too low: {n_total}", meta
    pos_path = TEMP_CACHE / "positions.bin"
    expected_bytes = n_total * FV_LEN * 4
    if not pos_path.is_file():
        return False, "positions.bin missing", meta
    actual = pos_path.stat().st_size
    if actual < expected_bytes:
        return False, f"positions.bin size {actual} < expected {expected_bytes}", meta
    ok, reason = check_fingerprint(TEMP_CACHE)
    if not ok:
        return False, f"fingerprint: {reason}", meta
    # Prove memmap opens read-only
    try:
        import numpy as np

        mm = np.memmap(pos_path, dtype="float32", mode="r", shape=(n_total, FV_LEN))
        _ = mm[0, 0]
        del mm
    except Exception as exc:
        return False, f"memmap unreadable: {exc}", meta
    return True, "ok", meta


def _featurization_complete(log_text: str) -> bool:
    if any(bad in log_text for bad in V1_PAST_SAFE_POINT):
        return False
    if any(marker in log_text for marker in FEATURIZATION_DONE_MARKERS):
        return True
    ok, _reason, meta = _temp_cache_complete()
    if ok and "Pass 3:" in log_text and "train=" in log_text:
        return True
    return False


def _verify_pre_finalize(v1_pid: int) -> tuple[bool, list[str]]:
    errors: list[str] = []
    ok_temp, temp_reason, meta = _temp_cache_complete()
    if not ok_temp:
        errors.append(f"temp cache: {temp_reason}")
    live_rows = _live_cache_rows()
    if live_rows is None:
        errors.append("live cache meta missing")
    elif live_rows > MAX_LIVE_ROWS:
        errors.append(f"live cache already swapped? n_total={live_rows}")
    if _pid_alive(v1_pid):
        errors.append(f"v1 pid {v1_pid} still alive")
    remaining_children = _find_v1_child_pids(v1_pid)
    alive_children = [p for p in remaining_children if _pid_alive(p)]
    if alive_children:
        errors.append(f"v1 child pids alive: {alive_children}")
    return len(errors) == 0, errors


def _run_finalize_v2() -> dict:
    env = os.environ.copy()
    env["RUSTFLAGS"] = "-C target-cpu=native"
    env["PYTHONPATH"] = str(_TRAINING)
    env["PYTHONUNBUFFERED"] = "1"

    _log("Running v2 finalize (cache)...")
    cache_result = finalize_cache_v2(allow_activation=True)
    report: dict = {
        "protocol_version": PROTOCOL_VERSION,
        "watcher_intercepted_v1": True,
        "cache_finalize": cache_result,
        "started_at": _utc_now(),
    }

    if not cache_result.get("activated"):
        report["status"] = "FAILED"
        report["failure"] = "cache activation failed or validation failed"
        report["finished_at"] = _utc_now()
        FINAL_REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        _write_state(report)
        return report

    _log("Running v2 prefix phase...")
    prefix_result = run_prefix_phase(max_ply=16, skip_build=False, allow_activation=True)
    report["prefix"] = prefix_result

    training_ok = (
        cache_result.get("activated")
        and prefix_result.get("activated")
        and OPENING_ENABLED_PATH.is_file()
        and not PAUSE_EPOCHS_PATH.is_file()
        and cache_result.get("validation", {}).get("passed")
        and prefix_result.get("validation", {}).get("passed")
    )
    report["training_may_resume"] = training_ok
    report["status"] = "SUCCESS" if training_ok else "FAILED"
    if not training_ok:
        report["failure"] = "post-activation gates not all satisfied"
        if PAUSE_EPOCHS_PATH.is_file():
            _log("Keeping pause_training_epochs.json — gates not satisfied")
    report["finished_at"] = _utc_now()
    FINAL_REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _write_state(report)
    return report


def watch_and_intercept(*, v1_pid: int = DEFAULT_V1_PID) -> int:
    tracked_pid = v1_pid if v1_pid > 0 else (_find_rebuild_pid() or 0)
    _log(
        f"Watcher started rebuild_pid={tracked_pid or 'auto'}"
        f" poll={POLL_SEC}s log={REBUILD_LOG}"
    )
    _write_state({
        "status": "watching",
        "v1_pid": tracked_pid,
        "rebuild_pid": tracked_pid,
        "started_at": _utc_now(),
    })

    intercepted = False
    while True:
        if tracked_pid <= 0 or not _pid_alive(tracked_pid):
            redetected = _find_rebuild_pid()
            if redetected and redetected != tracked_pid:
                _log(f"Auto-detected rebuild pid {redetected}")
                tracked_pid = redetected
                _write_state({
                    "status": "watching",
                    "v1_pid": tracked_pid,
                    "rebuild_pid": tracked_pid,
                })

        log_text = _read_log_tail(REBUILD_LOG)
        alive = tracked_pid > 0 and _pid_alive(tracked_pid)

        if any(bad in log_text for bad in V1_PAST_SAFE_POINT if bad != "CACHE VALIDATION FAILED"):
            if "=== Phase 2:" in log_text or "=== SAFE REBUILD COMPLETE ===" in log_text:
                _log("ERROR: v1 process appears to have completed without interception")
                _write_state({"status": "missed_intercept", "log_tail": log_text[-2000:]})
                return 2

        if _featurization_complete(log_text) and alive and not intercepted:
            _log("Featurization complete marker detected — terminating rebuild before activation")
            kill_stats = _terminate_tree(tracked_pid)
            _log(f"taskkill result: {kill_stats}")
            time.sleep(1.0)
            ok, errs = _verify_pre_finalize(tracked_pid)
            if not ok:
                _log(f"Pre-finalize verification FAILED: {errs}")
                _write_state({"status": "verify_failed", "errors": errs, "kill_stats": kill_stats})
                report = {
                    "status": "FAILED",
                    "failure": "pre-finalize verification",
                    "errors": errs,
                    "kill_stats": kill_stats,
                }
                FINAL_REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
                return 1
            intercepted = True
            _write_state({"status": "intercepted", "kill_stats": kill_stats})
            try:
                report = _run_finalize_v2()
            except Exception as exc:
                _log(f"finalize-v2 exception: {exc}")
                report = {
                    "status": "FAILED",
                    "failure": str(exc),
                    "pause_kept": PAUSE_EPOCHS_PATH.is_file(),
                }
                FINAL_REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
                _write_state(report)
                return 1
            _log(f"Watcher finished status={report.get('status')}")
            return 0 if report.get("status") == "SUCCESS" else 1

        if not alive and not intercepted:
            if tracked_pid <= 0:
                time.sleep(POLL_SEC)
                continue
            ok_temp, reason, _meta = _temp_cache_complete()
            if ok_temp:
                _log(f"rebuild pid exited; temp cache complete ({reason}) — running finalize-v2")
                ok, errs = _verify_pre_finalize(-1)
                if not ok and any("still alive" in e for e in errs):
                    errs = [e for e in errs if "still alive" not in e and "child" not in e]
                    ok = len(errs) == 0
                if not ok:
                    _log(f"Pre-finalize verification FAILED: {errs}")
                    return 1
                intercepted = True
                report = _run_finalize_v2()
                return 0 if report.get("status") == "SUCCESS" else 1
            _log(f"rebuild pid exited but temp cache not ready: {reason}; continuing to watch")

        if not alive and intercepted:
            return 0

        time.sleep(POLL_SEC)


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--v1-pid",
        type=int,
        default=DEFAULT_V1_PID,
        help="Rebuild PID to watch (0=auto-detect safe_rebuild.py)",
    )
    args = ap.parse_args()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return watch_and_intercept(v1_pid=args.v1_pid)


if __name__ == "__main__":
    raise SystemExit(main())
