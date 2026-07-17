#!/usr/bin/env python3
"""Benchmark persistent Ka NN inference without counting worker/model startup."""
from __future__ import annotations

import argparse
import sqlite3
import time

from ka_nn_collect_labels import GAMES_DB, LABELS_DB, KaWorker, sample_candidates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("directml", "cpu", "wasm", "js"), required=True)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--model-batch", type=int, default=64)
    parser.add_argument("--device-id", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    games = sqlite3.connect(GAMES_DB, timeout=120)
    labels = sqlite3.connect(LABELS_DB, timeout=120)
    try:
        candidates = sample_candidates(games, labels, limit=args.batch, seed=915_000)
    finally:
        games.close()
        labels.close()
    if not candidates:
        raise RuntimeError("no benchmark candidates")

    with KaWorker(
        backend=args.backend,
        batch_max=args.batch,
        model_batch=args.model_batch,
        device_id=args.device_id,
        threads=args.threads,
    ) as worker:
        worker.evaluate(candidates)  # warmup is intentionally excluded
        started = time.perf_counter()
        valid = 0
        rejected = 0
        for _ in range(args.iterations):
            response = worker.evaluate(candidates)
            valid += len(response.get("rows", []))
            rejected += len(response.get("rejected", []))
        elapsed = time.perf_counter() - started
    print(
        f"backend={args.backend} model_batch={args.model_batch} threads={args.threads} "
        f"requested={len(candidates) * args.iterations} valid={valid} rejected={rejected} "
        f"elapsed_sec={elapsed:.6f} valid_per_sec={valid / elapsed:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
