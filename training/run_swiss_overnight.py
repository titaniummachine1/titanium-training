"""Random overnight tournament — global Elo ladder.

Four independent game slots run forever; each slot claims the next pairing as
soon as it finishes (no waiting for slow Ka games). One Node process + progress dock.

Anchor: ace-v13-ti-pure@5s = 1400 Elo (~Quoridor Pro default player).

Usage:
    python training/run_swiss_overnight.py
    python training/run_swiss_overnight.py --list
    python training/run_swiss_overnight.py --scoreboard
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))

from manifest import (  # noqa: E402
    load_manifest,
    save_manifest,
    ANCHOR_RATING,
    format_scoreboard,
    compute_global_ratings,
)
from swiss_tournament import (  # noqa: E402
    PARALLEL_MATCHUPS,
    Pairing,
    list_pairings,
)

OVERNIGHT_BATCH = ROOT / "site" / "overnight_batch.js"
COORD_SCRIPT = ROOT / "training" / "coordinator.py"
COORD_PORT = int(os.environ.get("COORDINATOR_PORT", "8765"))
BIN = ROOT / "engine" / "target" / "release" / "titanium.exe"
TOURNAMENT_DIR = ROOT / "training" / "data" / "tournament"
DATA_DIR = ROOT / "training" / "data"
COORD_PID_FILE = DATA_DIR / "coordinator.pid"

_coord_proc = None


def _kill_pid(pid: int) -> None:
    """Kill a process by PID, works on Windows and Unix."""
    try:
        if platform.system() == "Windows":
            subprocess.run(
                ["taskkill", "/F", "/PID", str(pid)],
                capture_output=True,
                timeout=5,
            )
        else:
            import signal
            os.kill(pid, signal.SIGTERM)
    except Exception:
        pass


def _kill_coordinator_by_port(port: int) -> None:
    """Kill any process currently listening on port (Windows + Unix fallback)."""
    if platform.system() == "Windows":
        try:
            subprocess.run(
                [
                    "powershell", "-NoProfile", "-NonInteractive", "-Command",
                    f"$pids = (Get-NetTCPConnection -LocalPort {port} -State Listen "
                    f"-ErrorAction SilentlyContinue).OwningProcess | Select-Object -Unique; "
                    f"foreach ($p in $pids) {{ Stop-Process -Id $p -Force -ErrorAction SilentlyContinue }}",
                ],
                capture_output=True,
                timeout=8,
            )
        except Exception:
            pass
    else:
        try:
            subprocess.run(["fuser", "-k", f"{port}/tcp"], capture_output=True, timeout=5)
        except Exception:
            pass
    time.sleep(0.4)  # Brief pause for OS to release the port


def _read_coord_pid() -> int | None:
    try:
        return int(COORD_PID_FILE.read_text().strip())
    except Exception:
        return None


def _write_coord_pid(pid: int) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    COORD_PID_FILE.write_text(str(pid))


def _clear_coord_pid() -> None:
    COORD_PID_FILE.unlink(missing_ok=True)


def coordinator_url() -> str:
    return os.environ.get("COORDINATOR_URL", f"http://127.0.0.1:{COORD_PORT}")


def start_coordinator() -> None:
    global _coord_proc
    url = coordinator_url()

    # --- Kill any coordinator tracked from a previous run ---
    stale_pid = _read_coord_pid()
    if stale_pid is not None:
        print(f"  Killing stale coordinator (pid {stale_pid})...", flush=True)
        _kill_pid(stale_pid)
        _clear_coord_pid()
        time.sleep(0.3)

    # --- Also sweep port for any untracked stale processes ---
    port_occupied = False
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=0.5):
            port_occupied = True
    except Exception:
        pass

    if port_occupied:
        print(f"  Port {COORD_PORT} still occupied — force-killing by port...", flush=True)
        _kill_coordinator_by_port(COORD_PORT)
        # Wait up to 3 s for port to free
        for _ in range(15):
            try:
                urllib.request.urlopen(f"{url}/health", timeout=0.3)
                time.sleep(0.2)
            except Exception:
                break

    # --- Start a fresh coordinator (stdout/stderr suppressed — not needed) ---
    _coord_proc = subprocess.Popen(
        [sys.executable, str(COORD_SCRIPT), "--port", str(COORD_PORT)],
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _write_coord_pid(_coord_proc.pid)

    for _ in range(40):
        try:
            urllib.request.urlopen(f"{url}/health", timeout=0.5)
            print(f"  coordinator {url} (pid {_coord_proc.pid}, single writer)", flush=True)
            return
        except Exception:
            time.sleep(0.1)
    raise RuntimeError(f"coordinator failed to start on {url}")


def stop_coordinator() -> None:
    global _coord_proc
    _clear_coord_pid()  # Always remove PID file on clean stop
    if _coord_proc is None or _coord_proc.poll() is not None:
        _coord_proc = None
        return
    _coord_proc.terminate()
    try:
        _coord_proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        _coord_proc.kill()
    _coord_proc = None


def run_pool(slots: int) -> int:
    """Long-lived Node pool — slots claim pairings independently via coordinator."""
    cmd = ["node", str(OVERNIGHT_BATCH), "--pool", "--slots", str(slots)]
    env = os.environ.copy()
    env.setdefault("COORDINATOR_URL", coordinator_url())
    print(f"  >> {' '.join(cmd)}", flush=True)
    if sys.stderr.isatty():
        return subprocess.run(cmd, cwd=str(ROOT), env=env).returncode
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
    proc.wait()
    return proc.returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true", help="Show all matchups and game counts")
    ap.add_argument("--scoreboard", action="store_true", help="Print scoreboard and exit")
    ap.add_argument("--parallel", type=int, default=PARALLEL_MATCHUPS,
                    help=f"Independent game slots (default {PARALLEL_MATCHUPS})")
    ap.add_argument("--batches", type=int, default=0,
                    help="(legacy) ignored in pool mode")
    args = ap.parse_args()

    if not BIN.exists():
        print(f"ERROR: build engine first: {BIN}")
        sys.exit(1)

    if args.list:
        print(f"\nELIGIBLE MATCHUPS  ({args.parallel} independent slots, 1 game at a time each)")
        print(f"{'#':<4} {'PAIRING':<36} {'GAMES':>6}  KIND")
        for i, (p, n, _tag) in enumerate(list_pairings(), 1):
            print(f"{i:<4} {p.label:<36} {n:>6}  {p.kind}")
        print(f"\nAnchor: ace-v13-ti-pure@5s = {int(ANCHOR_RATING)}")
        return

    if args.scoreboard:
        manifest = load_manifest()
        if not manifest.get("global_ratings"):
            manifest["global_ratings"] = compute_global_ratings(manifest.get("matchups", {}))
        print(format_scoreboard(manifest))
        return

    print("=" * 64)
    print("RANDOM OVERNIGHT TOURNAMENT  (continuous pool)")
    print(f"  {args.parallel} independent slots — next pairing when each game finishes")
    print(f"  Baseline anchor: ace-v13-ti-pure@5s = {int(ANCHOR_RATING)}")
    print("  Ponder on; Ka/Ishtar max 1 remote slot; coordinator serializes writes")
    print("=" * 64)
    print(format_scoreboard(load_manifest()))

    start_coordinator()
    TOURNAMENT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        rc = run_pool(args.parallel)
        if rc != 0:
            print(f"pool exited {rc}", flush=True)
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
        manifest = load_manifest()
        save_manifest(manifest)
        print(format_scoreboard(manifest))
    finally:
        stop_coordinator()


if __name__ == "__main__":
    main()
