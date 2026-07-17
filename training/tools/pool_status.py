#!/usr/bin/env python3
"""One-shot training pool health snapshot for supervision."""
from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_TRAINING = Path(__file__).resolve().parents[1]
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

LOG_DIR = _TRAINING / "data" / "overnight_logs"
STATE_PATH = LOG_DIR / "continuous_pool_state.json"
POOL_LOG = LOG_DIR / "continuous_pool.log"
LOCK_PATH = LOG_DIR / "continuous_pool.lock.json"
CACHE_DIR = _TRAINING / "data" / "feature_cache"
BEST = _TRAINING / "runs" / "value_oracle" / "net_weights_best.bin"
PREVIOUS = _TRAINING / "runs" / "value_oracle" / "net_weights_previous.bin"
GAMES_DB = _TRAINING / "data" / "canonical" / "games.db"


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _pool_pid() -> int | None:
    lock = _read_json(LOCK_PATH)
    pid = lock.get("pid")
    if not pid:
        return None
    try:
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if str(pid) in out.stdout:
            return int(pid)
    except Exception:
        pass
    return None


def _oracle_tunnel() -> str:
    token_path = Path.home() / "AppData" / "Local" / "titanium-oracle-api-token"
    if not token_path.is_file():
        return "no_token"
    token = token_path.read_text(encoding="ascii").strip()
    try:
        import urllib.request

        req = urllib.request.Request(
            "http://127.0.0.1:8765/status",
            headers={"Authorization": f"Bearer {token}"},
        )
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = json.loads(resp.read().decode())
            active = data.get("active_generation_id") or data.get("generation_id") or "?"
            workers = data.get("workers") or data.get("worker_count") or "?"
            return f"OK gen={active} workers={workers}"
    except Exception as exc:
        return f"DOWN ({exc})"


def _log_tail_stats() -> dict:
    if not POOL_LOG.is_file():
        return {}
    lines = POOL_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
    games = [ln for ln in lines if "] game " in ln]
    errors = [ln for ln in lines[-500:] if "error" in ln.lower() or "FAILED" in ln]
    epochs = [ln for ln in lines if ln.startswith("--- epoch")]
    last_game = games[-1] if games else None
    last_epoch = epochs[-1] if epochs else None
    # Rough games/min from last 20 game lines timestamps not available — use mtime delta
    return {
        "last_game_line": last_game,
        "last_epoch_line": last_epoch,
        "recent_errors": errors[-5:],
        "total_game_log_lines": len(games),
    }


def _games_db_counts() -> dict:
    if not GAMES_DB.is_file():
        return {}
    con = sqlite3.connect(str(GAMES_DB), timeout=5)
    try:
        total = con.execute("SELECT COUNT(*) FROM games").fetchone()[0]
        overnight = con.execute(
            "SELECT COUNT(*) FROM games WHERE source IN ('overnight_selfplay','overnight_mixed')"
        ).fetchone()[0]
        oracle = con.execute(
            "SELECT COUNT(*) FROM games WHERE source LIKE 'oracle%'"
        ).fetchone()[0]
        return {"total": total, "overnight": overnight, "oracle": oracle}
    finally:
        con.close()


def _npy_len(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        import numpy as np

        return int(len(np.load(path, mmap_mode="r", allow_pickle=True)))
    except Exception:
        return None


def main() -> int:
    state = _read_json(STATE_PATH)
    cache_meta = _read_json(CACHE_DIR / "meta.json")
    pool_ds = _TRAINING / "data" / "teacher_dataset_pool"
    pool_rows = 0
    pos_pq = pool_ds / "positions" / "part-00000.parquet"
    if pos_pq.is_file():
        import pyarrow.parquet as pq

        pool_rows = pq.read_table(pos_pq).num_rows

    epoch = int(state.get("epoch", 0))
    pos_since = int(state.get("positions_since_epoch", 0))
    trigger = 2048
    games_since = int(state.get("games_since_epoch", 0))
    n_cache = int(cache_meta.get("n_total", 0))
    n_train = int(cache_meta.get("n_train", n_cache))
    n_val = int(cache_meta.get("n_val", 0))
    train_idx_len = _npy_len(CACHE_DIR / "train_indices.npy")
    val_idx_len = _npy_len(CACHE_DIR / "val_indices.npy")
    usage = {}
    try:
        from position_usage import status as usage_status

        usage = usage_status(CACHE_DIR)
    except Exception:
        pass
    pid = _pool_pid()
    log_age_s = None
    if POOL_LOG.is_file():
        log_age_s = time.time() - POOL_LOG.stat().st_mtime

    best_sz = BEST.stat().st_size if BEST.is_file() else 0
    prev_sz = PREVIOUS.stat().st_size if PREVIOUS.is_file() else 0
    same_weights = False
    if BEST.is_file() and PREVIOUS.is_file():
        import hashlib

        def _sha(p: Path) -> str:
            h = hashlib.sha256()
            with p.open("rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            return h.hexdigest()[:16]

        same_weights = _sha(BEST) == _sha(PREVIOUS)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"=== Training pool status @ {now} ===")
    print(f"Pool process: {'pid=' + str(pid) if pid else 'NOT RUNNING'}")
    print(f"Log freshness: {log_age_s:.0f}s ago" if log_age_s is not None else "Log: missing")
    print(f"Epoch: {epoch}  |  new_positions: {pos_since}/{trigger}  |  games_with_new_data: {games_since}")
    if pos_since > 0:
        pct = min(100.0, 100.0 * pos_since / trigger)
        print(f"Next epoch progress: {pct:.1f}% ({max(0, trigger - pos_since)} positions remaining)")
    split_detail = f"{n_train:,} train / {n_val:,} val rows"
    if train_idx_len is not None and val_idx_len is not None:
        split_detail = f"{train_idx_len:,} train / {val_idx_len:,} val indices"
    print(f"Cache: {n_cache:,} total, {split_detail}  |  pool parquet: {pool_rows:,} positions")
    if n_cache > 0 and (n_val == 0 or val_idx_len == 0):
        print("WARN: feature cache has no validation rows; promotion diagnostics can overfit.")
    if usage:
        print(
            f"Position retirement: {usage.get('retired', 0):,} retired, "
            f"{usage.get('active_train', n_train):,} active train rows "
            f"(each row drops after {usage.get('max_usage', 5)} training epochs, not pool epochs)"
        )
    print(f"Weights: best={best_sz}B  previous={prev_sz}B  same_as_best={same_weights}")
    print(f"Oracle tunnel: {_oracle_tunnel()}")
    db = _games_db_counts()
    if db:
        print(f"Games DB: {db['total']:,} total ({db['overnight']:,} overnight, {db['oracle']:,} oracle)")
    tail = _log_tail_stats()
    if tail.get("last_epoch_line"):
        print(f"Last epoch: {tail['last_epoch_line']}")
    if tail.get("last_game_line"):
        print(f"Last game:  {tail['last_game_line']}")
    if tail.get("recent_errors"):
        print("Recent errors:")
        for e in tail["recent_errors"]:
            print(f"  {e}")
    return 0 if pid else 1


if __name__ == "__main__":
    raise SystemExit(main())
