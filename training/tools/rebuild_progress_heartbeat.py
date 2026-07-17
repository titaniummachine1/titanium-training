#!/usr/bin/env python3
"""Append rebuild batch/CPU/I/O samples every minute while safe_rebuild is alive."""
from __future__ import annotations

import argparse
import json
import re
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
REBUILD_LOG = LOG_DIR / "safe_rebuild.log"
REBUILD_ERR = LOG_DIR / "safe_rebuild_err.log"
PID_FILE = LOG_DIR / "safe_rebuild.pid"
OUT_JSONL = LOG_DIR / "rebuild_progress.jsonl"

BATCH_RE = re.compile(
    r"batches=\s*(\d+)/(\d+)\s+approx_done=\s*([\d,]+)/([\d,]+)\s+written=([\d,]+)\s+failed=(\d+)\s+([\d.]+) pos/s\s+ETA\s+([\d.]+)s"
)
SINGLE_BATCH_RE = re.compile(
    r"(\d[\d,]*)/([\d,]+)\s+written=([\d,]+)\s+failed=([\d,]+)\s+([\d.]+) pos/s\s+ETA\s+([\d.]+)s"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_rebuild_pid(explicit: int | None) -> int | None:
    if explicit and explicit > 0:
        return explicit
    if PID_FILE.is_file():
        try:
            return int(PID_FILE.read_text(encoding="ascii").strip())
        except ValueError:
            return None
    return None


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
        import os

        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _parse_progress(log_text: str) -> dict:
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
                "log_line": line.strip(),
            }
        m2 = SINGLE_BATCH_RE.search(line)
        if m2:
            return {
                "approx_done": int(m2.group(1).replace(",", "")),
                "target_rows": int(m2.group(2).replace(",", "")),
                "rows_written": int(m2.group(3).replace(",", "")),
                "failed": int(m2.group(4).replace(",", "")),
                "throughput_pos_s": float(m2.group(5)),
                "eta_sec": float(m2.group(6)),
                "log_line": line.strip(),
            }
    return {}


def _win_process_stats(pid: int) -> dict:
    ps = (
        f"$p=Get-Process -Id {pid} -EA SilentlyContinue; "
        "if (-not $p) { exit 1 }; "
        "$kids=(Get-CimInstance Win32_Process -Filter \"ParentProcessId=$($p.Id)\" -EA SilentlyContinue); "
        "$ti=($kids | Where-Object { $_.Name -eq 'titanium.exe' }).Count; "
        "[pscustomobject]@{"
        "cpu=$p.CPU; ws_mb=[math]::Round($p.WorkingSet64/1MB,1); "
        "pm_mb=[math]::Round($p.PrivateMemorySize64/1MB,1); "
        "io_read_mb=[math]::Round($p.IOReadBytes/1MB,2); "
        "io_write_mb=[math]::Round($p.IOWriteBytes/1MB,2); "
        "threads=$p.Threads.Count; children=$kids.Count; titanium_children=$ti"
        "} | ConvertTo-Json -Compress"
    )
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return {}
        return json.loads(out.stdout.strip())
    except Exception:
        return {}


def _system_ram_pct() -> float | None:
    ps = (
        "$o=Get-CimInstance Win32_OperatingSystem; "
        "$t=[double]$o.TotalVisibleMemorySize; $f=[double]$o.FreePhysicalMemory; "
        "if ($t -le 0) { exit 1 }; [math]::Round(100.0*($t-$f)/$t,1)"
    )
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            capture_output=True,
            text=True,
            timeout=15,
        )
        val = out.stdout.strip()
        return float(val) if val else None
    except Exception:
        return None


def sample_once(pid: int, prev: dict | None) -> dict:
    log_text = REBUILD_LOG.read_text(encoding="utf-8", errors="replace") if REBUILD_LOG.is_file() else ""
    err_tail = ""
    if REBUILD_ERR.is_file():
        err_tail = REBUILD_ERR.read_text(encoding="utf-8", errors="replace")[-2000:]
    progress = _parse_progress(log_text)
    proc = _win_process_stats(pid)
    rec: dict = {
        "ts": _utc_now(),
        "rebuild_pid": pid,
        "alive": _pid_alive(pid),
        "progress": progress,
        "process": proc,
        "system_ram_pct": _system_ram_pct(),
        "rebuild_log_mtime": (
            datetime.fromtimestamp(REBUILD_LOG.stat().st_mtime, tz=timezone.utc).isoformat()
            if REBUILD_LOG.is_file()
            else None
        ),
        "stderr_tail": err_tail if err_tail.strip() else None,
    }
    if prev and proc:
        dt = 60.0
        try:
            rec["cpu_delta_sec"] = round(float(proc.get("cpu", 0)) - float(prev.get("cpu", 0)), 3)
            rec["ws_delta_mb"] = round(float(proc.get("ws_mb", 0)) - float(prev.get("ws_mb", 0)), 1)
            rec["io_read_delta_mb"] = round(
                float(proc.get("io_read_mb", 0)) - float(prev.get("io_read_mb", 0)), 2
            )
            rec["io_write_delta_mb"] = round(
                float(proc.get("io_write_mb", 0)) - float(prev.get("io_write_mb", 0)), 2
            )
            rec["sample_interval_sec"] = dt
        except (TypeError, ValueError):
            pass
        prev_prog = prev.get("_progress") or {}
        cur_batch = progress.get("batch_current")
        prev_batch = prev_prog.get("batch_current")
        if cur_batch is not None and prev_batch is not None:
            rec["batch_delta"] = cur_batch - prev_batch
    if progress:
        rec["_progress"] = progress  # internal carry-forward; stripped before write
    return rec


def append_sample(rec: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out = {k: v for k, v in rec.items() if not k.startswith("_")}
    with OUT_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(out, separators=(",", ":")) + "\n")


def run(*, pid: int | None, interval_sec: float, once: bool) -> int:
    rb_pid = _read_rebuild_pid(pid)
    if rb_pid is None:
        print("No rebuild PID", file=sys.stderr)
        return 1
    prev_proc: dict | None = None
    prev_carry: dict | None = None
    while True:
        if not _pid_alive(rb_pid):
            append_sample(
                {
                    "ts": _utc_now(),
                    "rebuild_pid": rb_pid,
                    "alive": False,
                    "note": "rebuild process exited",
                }
            )
            return 0
        rec = sample_once(rb_pid, prev_carry)
        if prev_proc and rec.get("process"):
            # attach deltas using stored previous process snapshot
            p = rec["process"]
            rec["cpu_delta_sec"] = round(float(p.get("cpu", 0)) - float(prev_proc.get("cpu", 0)), 3)
            rec["ws_delta_mb"] = round(float(p.get("ws_mb", 0)) - float(prev_proc.get("ws_mb", 0)), 1)
            rec["io_read_delta_mb"] = round(
                float(p.get("io_read_mb", 0)) - float(prev_proc.get("io_read_mb", 0)), 2
            )
            rec["io_write_delta_mb"] = round(
                float(p.get("io_write_mb", 0)) - float(prev_proc.get("io_write_mb", 0)), 2
            )
            rec["sample_interval_sec"] = interval_sec
            prev_prog = (prev_carry or {}).get("_progress") or {}
            cur_prog = rec.get("progress") or {}
            if "batch_current" in cur_prog and "batch_current" in prev_prog:
                rec["batch_delta"] = cur_prog["batch_current"] - prev_prog["batch_current"]
        append_sample(rec)
        prev_proc = dict(rec.get("process") or {})
        prev_carry = {"_progress": rec.get("progress") or {}}
        if once:
            return 0
        time.sleep(interval_sec)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pid", type=int, default=0, help="Rebuild PID (default: safe_rebuild.pid)")
    ap.add_argument("--interval-sec", type=float, default=60.0)
    ap.add_argument("--once", action="store_true", help="Single sample then exit")
    args = ap.parse_args()
    pid = args.pid if args.pid > 0 else None
    return run(pid=pid, interval_sec=args.interval_sec, once=args.once)


if __name__ == "__main__":
    raise SystemExit(main())
