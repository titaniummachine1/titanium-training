"""File lock for cache featurization subprocesses (separate from game eval-batch lock)."""
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from pathlib import Path

from titanium_training.paths import TRAINING_ROOT

FEATURIZE_LOCK = TRAINING_ROOT / "data" / "featurize_batch.lock"
FEATURIZE_LOCK_TIMEOUT_SEC = float(os.environ.get("FEATURIZE_BATCH_LOCK_SEC", "600"))


@contextmanager
def featurize_batch_lock():
    FEATURIZE_LOCK.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + FEATURIZE_LOCK_TIMEOUT_SEC
    fd = None
    while time.time() < deadline:
        try:
            fd = open(FEATURIZE_LOCK, "x")
            fd.write(str(os.getpid()))
            fd.flush()
            break
        except FileExistsError:
            time.sleep(1.0)
    else:
        raise TimeoutError(
            f"featurize-batch lock busy after {FEATURIZE_LOCK_TIMEOUT_SEC:.0f}s"
        )
    try:
        yield
    finally:
        if fd is not None:
            fd.close()
        FEATURIZE_LOCK.unlink(missing_ok=True)
