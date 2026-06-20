"""Random overnight tournament — global Elo ladder.

Up to 8 independent game slots (7 when background NNUE is on — one thread for
eval-batch micro-train). Each slot claims the next pairing as soon as it finishes.
One Node process + progress dock.

Anchor: ace-v13-ti-pure@5s = 1200 Elo (Rust ti-pure; JS ace-v13 is bench-only).

Usage:
    python training/run_swiss_overnight.py
    python training/run_swiss_overnight.py --parallel 8
    python training/run_swiss_overnight.py --no-train          # 8 slots, no NNUE
    python training/run_swiss_overnight.py --list
    python training/run_swiss_overnight.py --scoreboard
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import queue
import re
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
POOL_STARTUP_LOG = ROOT / "training" / "data" / "pool_startup.log"


def _pool_log(msg: str) -> None:
    POOL_STARTUP_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(POOL_STARTUP_LOG, "a", encoding="utf-8") as f:
        f.write(msg.rstrip() + "\n")


def _pool_print(msg: str) -> None:
    """Startup chatter goes to log file; live UI is the Node progress dock."""
    _pool_log(msg)
    if os.environ.get("POOL_VERBOSE") == "1":
        print(msg, flush=True)

from tools.maintenance.manifest import (  # noqa: E402
    load_manifest,
    save_manifest,
    ANCHOR_ENTITY,
    ANCHOR_RATING,
    format_scoreboard,
    compute_global_ratings,
)
from titanium_training.training.guards import DEPLOY_EVERY_GAMES, HALFPW_WEIGHT_BYTES, net_weights_size_ok  # noqa: E402
from titanium_training.validation.engine_identity import assert_engine_ready  # noqa: E402
from tools.operations.swiss_tournament import (  # noqa: E402
    POOL_SLOTS_MAX,
    POOL_SLOTS_WITH_TRAIN,
    Pairing,
    list_pairings,
    pool_slots,
)

WEIGHTS = ROOT / "engine" / "src" / "titanium" / "net_weights.bin"


def preflight_weights() -> bool:
    if not WEIGHTS.exists():
        print(f"ERROR: missing {WEIGHTS} — run training/extend_field_planes.py")
        return False
    if not net_weights_size_ok(WEIGHTS):
        print(
            f"ERROR: {WEIGHTS.name} is {WEIGHTS.stat().st_size} B, "
            f"expected {HALFPW_WEIGHT_BYTES} B — run training/extend_field_planes.py"
        )
        return False
    return True

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


def start_coordinator(*, slots: int) -> None:
    global _coord_proc
    url = coordinator_url()

    # --- Kill any coordinator tracked from a previous run ---
    stale_pid = _read_coord_pid()
    if stale_pid is not None:
        _pool_print(f"  Killing stale coordinator (pid {stale_pid})...")
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
        _pool_print(f"  Port {COORD_PORT} still occupied — force-killing by port...")
        _kill_coordinator_by_port(COORD_PORT)
        # Wait up to 3 s for port to free
        for _ in range(15):
            try:
                urllib.request.urlopen(f"{url}/health", timeout=0.3)
                time.sleep(0.2)
            except Exception:
                break

    # --- Start a fresh coordinator (stdout/stderr suppressed — not needed) ---
    coord_env = os.environ.copy()
    coord_env["POOL_SLOTS"] = str(slots)
    _coord_proc = subprocess.Popen(
        [sys.executable, str(COORD_SCRIPT), "--port", str(COORD_PORT)],
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=coord_env,
    )
    _write_coord_pid(_coord_proc.pid)

    for _ in range(40):
        try:
            urllib.request.urlopen(f"{url}/health", timeout=0.5)
            _pool_print(f"  coordinator {url} (pid {_coord_proc.pid}, single writer)")
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


def run_pool(slots: int, *, enable_train: bool = True, max_games: int = 0) -> int:
    """Long-lived Node pool — slots claim pairings independently via coordinator."""
    cmd = ["node", str(OVERNIGHT_BATCH), "--pool", "--slots", str(slots)]
    if max_games > 0:
        cmd += ["--max-games", str(max_games)]
    env = os.environ.copy()
    env.setdefault("COORDINATOR_URL", coordinator_url())
    _pool_log(f"\n--- pool start {time.strftime('%Y-%m-%d %H:%M:%S')} ---")
    _pool_print(f"  >> {' '.join(cmd)}")

    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        # stderr must stay on the terminal — ProgressBoard uses alternate screen on stderr.
        text=True,
        bufsize=1,
    )

    pool_done_re = re.compile(r"^POOL_DONE db_id=(\d+)")
    train_q: queue.Queue[int | None] = queue.Queue()
    train_stop = threading.Event()

    def train_worker() -> None:
        os.environ["NNUE_POOL_QUIET"] = "1"
        from run_nnue_cycle import startup_train_catch_up, run_on_game

        training_blocked = startup_train_catch_up() != 0
        while not train_stop.is_set():
            try:
                item = train_q.get(timeout=1.0)
            except queue.Empty:
                continue
            if item is None:
                train_q.task_done()
                break
            if training_blocked:
                from titanium_training.training.guards import nnue_log
                nnue_log(f"game {item} deferred: an earlier training failure remains pending")
                train_q.task_done()
                continue
            try:
                rc = run_on_game(item)
                if rc != 0:
                    from titanium_training.training.guards import nnue_log
                    nnue_log(
                        f"game {item} failed training (rc={rc}); "
                        "pausing trainer so the cursor cannot skip it"
                    )
                    training_blocked = True
            except Exception as e:
                from titanium_training.training.guards import nnue_log
                nnue_log(f"error on game {item}: {e}")
                training_blocked = True
            train_q.task_done()

    train_thread = None
    if enable_train:
        train_thread = threading.Thread(target=train_worker, daemon=False)
        train_thread.start()

    assert proc.stdout is not None
    rc = 0
    try:
        for line in proc.stdout:
            stripped = line.strip()
            if enable_train:
                m = pool_done_re.match(stripped)
                if m:
                    gid = int(m.group(1))
                    train_q.put(gid)
        rc = proc.wait()
    finally:
        if enable_train:
            train_q.join()
            train_q.put(None)
            train_q.join()
            if train_thread is not None:
                train_thread.join(timeout=30)
        train_stop.set()
    return rc


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true", help="Show all matchups and game counts")
    ap.add_argument("--scoreboard", action="store_true", help="Print scoreboard and exit")
    ap.add_argument("--parallel", type=int, default=None,
                    help=f"Game slots (default {POOL_SLOTS_WITH_TRAIN} with train, "
                    f"{POOL_SLOTS_MAX} with --no-train; max {POOL_SLOTS_MAX})")
    ap.add_argument("--batches", type=int, default=0,
                    help="(legacy) ignored in pool mode")
    ap.add_argument("--no-train", action="store_true",
                    help="Disable background HalfPW NNUE training")
    ap.add_argument("--games", type=int, default=0,
                    help="Stop after this many game attempts (0 = continuous)")
    ap.add_argument("--local-only", action="store_true",
                    help="Use local ti-pure/self/frozen pairings; no remote opponents")
    args = ap.parse_args()

    if args.local_only:
        os.environ["POOL_LOCAL_ONLY"] = "1"

    if not BIN.exists():
        print(f"ERROR: build engine first: {BIN}")
        sys.exit(1)

    if not args.list and not args.scoreboard and not preflight_weights():
        sys.exit(1)
    if not args.list and not args.scoreboard:
        from tools.operations.pool_preflight import main as pool_preflight_main

        if pool_preflight_main() != 0:
            sys.exit(1)
        try:
            stamp = assert_engine_ready(write_if_missing=True, parity=True)
            _pool_print(f"  Engine stamp OK: {stamp['sha256'][:12]}  {BIN}")
        except Exception as e:
            print(f"ERROR: engine validation failed: {e}")
            sys.exit(1)

    enable_train = not args.no_train
    slots = pool_slots(train=enable_train, override=args.parallel)
    os.environ["POOL_SLOTS"] = str(slots)

    if args.list:
        print(f"\nELIGIBLE MATCHUPS  ({slots} independent slots, 1 game at a time each)")
        print(f"{'#':<4} {'PAIRING':<40} {'GAMES':>6}  {'ROLE':<6} KIND")
        for i, (p, n, role) in enumerate(list_pairings(), 1):
            print(f"{i:<4} {p.label:<40} {n:>6}  {role:<6} {p.kind}")
        print(f"\nAnchor: {ANCHOR_ENTITY} = {int(ANCHOR_RATING)}")
        return

    if args.scoreboard:
        from tools.maintenance.housekeeping import run_pool_housekeeping

        for msg in run_pool_housekeeping(reset_pool_counter=False):
            if msg.startswith("pruned"):
                print(f"  housekeeping: {msg}")
        manifest = load_manifest()
        if not manifest.get("global_ratings"):
            manifest["global_ratings"] = compute_global_ratings(manifest.get("matchups", {}))
        print(format_scoreboard(manifest))
        return

    POOL_STARTUP_LOG.write_text("", encoding="utf-8")
    _pool_print("=" * 64)
    _pool_print("RANDOM OVERNIGHT TOURNAMENT  (continuous pool)")
    _pool_print(f"  {slots} game slots — next pairing when each game finishes")
    if enable_train and slots < POOL_SLOTS_MAX:
        _pool_print(f"  ({POOL_SLOTS_MAX - slots} slot reserved for eval-batch micro-train; use --parallel 8 or --no-train for full {POOL_SLOTS_MAX})")
    _pool_print(f"  Baseline anchor: {ANCHOR_ENTITY} = {int(ANCHOR_RATING)}")
    _pool_print("  Adaptive: zero-ink first; Ka fallback only after a <=4/16 zero window")
    _pool_print("  Reserved: one adaptive remote + ti-pure@10s + v15 self@10s + frozen")
    if not args.no_train:
        _pool_print(
            f"  Background NNUE: micro-train; deploy every {DEPLOY_EVERY_GAMES} trains; "
            "targets=WDL/self-play outcomes only"
        )
    _pool_print("=" * 64)
    _pool_print(f"Live UI on this console — startup log: {POOL_STARTUP_LOG}")

    from tools.maintenance.housekeeping import run_pool_housekeeping

    for msg in run_pool_housekeeping(reset_pool_counter=True):
        _pool_print(f"  housekeeping: {msg}")

    manifest = load_manifest()
    t = manifest.setdefault("tournament", {})
    t["mode"] = "random-pool"
    t["parallel"] = slots
    save_manifest(manifest)

    start_coordinator(slots=slots)

    try:
        rc = run_pool(slots, enable_train=enable_train, max_games=max(0, args.games))
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
