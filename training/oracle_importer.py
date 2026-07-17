#!/usr/bin/env python3
"""Standalone Oracle result importer — decoupled from local game generation.

Polls the laptop Oracle tunnel, imports games into games.db + teacher parquet,
and acknowledges results.  A failure in local_game_pool must not stop imports.
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

_TRAINING = Path(__file__).resolve().parent
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

from oracle_laptop_client import OracleClientConfig, OracleImportThread, pull_once
from pool_lock import ORACLE_IMPORTER_LOCK_PATH, PoolInstanceLock, release_pool_lock

LOG_DIR = _TRAINING / "data" / "overnight_logs"
STATE_PATH = LOG_DIR / "oracle_importer_state.json"
PID_PATH = LOG_DIR / "oracle_importer.pid"
LOG_PATH = LOG_DIR / "oracle_importer.log"


def log(msg: str) -> None:
    line = f"[{datetime.now(timezone.utc).isoformat()}] {msg}"
    print(line, flush=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _save_state(payload: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(STATE_PATH)


def _record_import(item: dict) -> None:
    state = {}
    if STATE_PATH.is_file():
        try:
            state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            state = {}
    total = int(state.get("imports_total", 0) or 0) + 1
    payload = {
        "imports_total": total,
        "last_game_id": item.get("game_id"),
        "last_generation_id": item.get("generation_id"),
        "last_matchup_type": item.get("matchup_type"),
        "last_new_positions": int(item.get("new_positions", 0) or 0),
        "last_import_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_state(payload)
    log(
        f"[oracle] imported {item.get('game_id')} gen={item.get('generation_id')} "
        f"matchup={item.get('matchup_type')} new_pos={item.get('new_positions', 0)} "
        f"total={total}"
    )


def main() -> int:
    from prep_guard import guard_real_work

    guard_real_work("labeling", detail="oracle_importer")
    def _on_signal(signum, _frame):
        log(f"Signal {signum} — releasing importer lock and stopping...")
        release_pool_lock(ORACLE_IMPORTER_LOCK_PATH)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _on_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _on_signal)

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:8765")
    ap.add_argument("--token", required=True)
    ap.add_argument("--poll-sec", type=float, default=30.0)
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--once", action="store_true", help="Single pull then exit")
    args = ap.parse_args()

    cfg = OracleClientConfig(
        base_url=args.url,
        token=args.token,
        poll_sec=max(1.0, args.poll_sec),
        batch_limit=max(1, args.limit),
    )

    with PoolInstanceLock(lock_path=ORACLE_IMPORTER_LOCK_PATH) as lock_info:
        PID_PATH.write_text(str(lock_info.pid), encoding="ascii")
        log(
            f"Oracle importer lock acquired pid={lock_info.pid} "
            f"url={cfg.base_url} poll={cfg.poll_sec}s"
        )
        if args.once:
            result = pull_once(cfg)
            for item in result.get("imported", []):
                _record_import(item)
            if result.get("failed"):
                log(f"pull failures: {result['failed']}")
            return 0

        stop = threading.Event()
        importer = OracleImportThread(cfg, on_import=_record_import)
        importer.start()
        log("Oracle importer thread started")
        try:
            while not stop.is_set():
                time.sleep(1.0)
        except KeyboardInterrupt:
            log("Stopping Oracle importer...")
            importer.stop_event.set()
            importer.join(timeout=15)
        finally:
            release_pool_lock(ORACLE_IMPORTER_LOCK_PATH)
            log(f"Oracle importer stopped pid={lock_info.pid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
