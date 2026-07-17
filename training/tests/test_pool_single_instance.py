"""Single-instance lock for continuous_pool."""
from __future__ import annotations

import multiprocessing as mp
import time


def _hold_lock(seconds: float) -> None:
    import sys
    from pathlib import Path

    training = Path(__file__).resolve().parent
    if str(training) not in sys.path:
        sys.path.insert(0, str(training))
    from pool_lock import acquire_pool_lock

    acquire_pool_lock()
    time.sleep(seconds)


def test_second_pool_instance_rejected():
    from pool_lock import acquire_pool_lock, release_pool_lock

    release_pool_lock()
    proc = mp.Process(target=_hold_lock, args=(20.0,))
    proc.start()
    try:
        time.sleep(1.0)
        try:
            acquire_pool_lock()
            raise AssertionError("expected RuntimeError for second instance")
        except RuntimeError as exc:
            assert "already running" in str(exc).lower()
    finally:
        proc.terminate()
        proc.join(timeout=5)
        release_pool_lock()


def test_stale_lock_removed_when_pid_dead():
    import json
    from pool_lock import LOCK_PATH, acquire_pool_lock, release_pool_lock

    release_pool_lock()
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOCK_PATH.write_text(
        json.dumps({"pid": 99999999, "started_at": "stale", "repo": "x", "command_line": "x"}),
        encoding="utf-8",
    )
    try:
        acquire_pool_lock()
    finally:
        release_pool_lock()
