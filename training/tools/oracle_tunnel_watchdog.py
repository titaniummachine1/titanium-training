#!/usr/bin/env python3
"""Keep the Oracle SSH tunnel (127.0.0.1:8765 -> Oracle) alive, but only while
training_coordinator is actually running -- no point holding the tunnel open
(or restarting it) when nothing on the laptop is going to use it.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

_TRAINING = Path(__file__).resolve().parents[1]
_REPO = _TRAINING.parent

LOG_DIR = _TRAINING / "data" / "overnight_logs"
TUNNEL_PID_FILE = LOG_DIR / "oracle_ssh_tunnel.pid"
TUNNEL_LOG = LOG_DIR / "oracle_ssh_tunnel.log"
COORDINATOR_PID_FILE = LOG_DIR / "training_coordinator.pid"
WATCHDOG_LOG = LOG_DIR / "oracle_tunnel_watchdog.log"

ORACLE_HOST = os.environ.get("ORACLE_HOST", "92.5.77.92")
ORACLE_USER = os.environ.get("ORACLE_USER", "ubuntu")
ORACLE_KEY_PATH = os.environ.get("ORACLE_KEY_PATH", str(Path.home() / ".ssh" / "oracle_titanium.key"))
LOCAL_PORT = int(os.environ.get("ORACLE_TUNNEL_LOCAL_PORT", "8765"))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log(msg: str) -> None:
    WATCHDOG_LOG.parent.mkdir(parents=True, exist_ok=True)
    with WATCHDOG_LOG.open("a", encoding="utf-8") as f:
        f.write(f"[{_utc_now()}] {msg}\n")


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
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _read_pid(path: Path) -> int | None:
    if not path.is_file():
        return None
    return _optional_pid(path.read_text(encoding="ascii").strip())


def _tunnel_forwarding(port: int, timeout: float = 5.0) -> bool:
    """A raw TCP connect only proves *something* is listening locally -- an
    ssh -L process can stay alive and keep that local socket open even after
    the actual upstream connection has died (a "zombie" tunnel: process
    running, port open, nothing forwarded). Confirmed live, 2026-07-06: the
    tunnel looked healthy by that shallow check for 3.5+ hours while the
    Oracle importer sat completely stalled. Do a real HTTP round-trip through
    the tunnel instead -- any HTTP response (even 401 unauthorized) proves the
    forward is actually relaying traffic; a connection error or timeout means
    it isn't, regardless of what the local socket looks like.
    """
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/status")
        urllib.request.urlopen(req, timeout=timeout)
        return True
    except urllib.error.HTTPError:
        return True  # server responded (e.g. 401) -- tunnel is genuinely up
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


def _coordinator_alive() -> bool:
    return _pid_alive(_read_pid(COORDINATOR_PID_FILE))


def _tunnel_healthy() -> bool:
    return _pid_alive(_read_pid(TUNNEL_PID_FILE)) and _tunnel_forwarding(LOCAL_PORT)


def _restart_tunnel() -> int | None:
    old_pid = _read_pid(TUNNEL_PID_FILE)
    if old_pid and _pid_alive(old_pid):
        subprocess.run(
            ["taskkill", "/PID", str(old_pid), "/F", "/T"],
            capture_output=True,
            timeout=30,
        )
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ssh",
        "-i", ORACLE_KEY_PATH,
        "-N",
        "-o", "ServerAliveInterval=15",
        "-o", "ServerAliveCountMax=3",
        "-o", "StrictHostKeyChecking=accept-new",
        "-L", f"{LOCAL_PORT}:127.0.0.1:8765",
        f"{ORACLE_USER}@{ORACLE_HOST}",
    ]
    out_f = open(TUNNEL_LOG, "a", encoding="utf-8", errors="replace")
    err_f = open(str(TUNNEL_LOG) + ".err", "a", encoding="utf-8", errors="replace")
    creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    proc = subprocess.Popen(
        cmd,
        cwd=str(_REPO),
        stdout=out_f,
        stderr=err_f,
        creationflags=creationflags,
    )
    TUNNEL_PID_FILE.write_text(str(proc.pid), encoding="ascii")
    return proc.pid


def watch(*, poll_sec: float) -> int:
    _log(f"oracle_tunnel_watchdog started (host={ORACLE_HOST} port={LOCAL_PORT})")
    was_coordinator_alive = False
    while True:
        try:
            coordinator_alive = _coordinator_alive()
            if coordinator_alive and not was_coordinator_alive:
                _log("training_coordinator is up — watchdog now actively maintaining tunnel")
            elif not coordinator_alive and was_coordinator_alive:
                _log("training_coordinator is down — watchdog going idle, leaving tunnel as-is")
            was_coordinator_alive = coordinator_alive

            if coordinator_alive and not _tunnel_healthy():
                new_pid = _restart_tunnel()
                _log(f"tunnel down/unreachable — restarted, new pid={new_pid}")
                time.sleep(3)
        except Exception as exc:
            _log(f"watchdog error: {exc}")
        time.sleep(poll_sec)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--poll-sec", type=float, default=30.0)
    args = ap.parse_args()
    return watch(poll_sec=args.poll_sec)


if __name__ == "__main__":
    raise SystemExit(main())
