#!/usr/bin/env python3
"""Local self-play generator — decoupled from Oracle importing."""
from __future__ import annotations

import sys
from pathlib import Path

_TRAINING = Path(__file__).resolve().parent
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

from pool_lock import LOCAL_GAME_POOL_LOCK_PATH, PoolInstanceLock, release_pool_lock

LOG_DIR = _TRAINING / "data" / "overnight_logs"
PID_PATH = LOG_DIR / "local_game_pool.pid"


def main(argv: list[str] | None = None) -> int:
    from prep_guard import guard_real_work

    guard_real_work("local_self_play_pool")
    import signal

    from continuous_pool import ContinuousPool, build_pool_config, log, parse_pool_args

    def _on_signal(signum, _frame):
        log(f"Signal {signum} — releasing local pool lock and stopping...")
        release_pool_lock(LOCAL_GAME_POOL_LOCK_PATH)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _on_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _on_signal)

    args = parse_pool_args(argv)
    cfg = build_pool_config(args, no_oracle=True)
    pool = ContinuousPool(cfg)
    with PoolInstanceLock(lock_path=LOCAL_GAME_POOL_LOCK_PATH) as lock_info:
        PID_PATH.write_text(str(lock_info.pid), encoding="ascii")
        log(
            f"Local game pool lock acquired pid={lock_info.pid} lock_id={lock_info.pid}@"
            f"{lock_info.started_at} threads={cfg.threads}"
        )
        try:
            return pool.run()
        finally:
            log(f"Local game pool lock released pid={lock_info.pid}")
            release_pool_lock(LOCAL_GAME_POOL_LOCK_PATH)


if __name__ == "__main__":
    raise SystemExit(main())
