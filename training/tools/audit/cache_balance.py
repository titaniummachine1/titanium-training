#!/usr/bin/env python3
"""Cache balance / label distribution audit for feature-cache training."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _percentiles(arr: np.ndarray, ps: tuple[float, ...] = (50, 90, 99)) -> dict[str, float]:
    if len(arr) == 0:
        return {f"p{int(p)}": 0.0 for p in ps}
    return {f"p{int(p)}": float(np.percentile(arr, p)) for p in ps}


def write_cache_balance_report(
    cache_dir: Path,
    *,
    row_position_keys: list[Any],
    obs_counts: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    positions_mmap: np.ndarray | None = None,
    n_total: int | None = None,
) -> Path:
    """Write cache_balance.json summarizing targets and observation counts."""
    cache_dir = Path(cache_dir)
    n = len(row_position_keys)
    if positions_mmap is None and (cache_dir / "positions.bin").is_file():
        import gc; gc.collect()  # release any stale mmap handles on Windows
        if n_total is None:
            meta = json.loads((cache_dir / "meta.json").read_text(encoding="utf-8"))
            n_total = int(meta["n_total"])
        positions_mmap = np.memmap(
            cache_dir / "positions.bin",
            dtype="float32",
            mode="r",
            shape=(n_total, 547),
        )

    targets = np.array([float(positions_mmap[i][0]) for i in range(n)], dtype=np.float64) if positions_mmap is not None else np.array([])

    def _slice_stats(idxs: np.ndarray) -> dict[str, Any]:
        if len(idxs) == 0 or len(targets) == 0:
            return {"count": int(len(idxs))}
        t = targets[idxs]
        obs = obs_counts[idxs]
        win = float(np.mean(t > 0.55))
        loss = float(np.mean(t < 0.45))
        draw = float(1.0 - win - loss)
        return {
            "count": int(len(idxs)),
            "target_mean": float(np.mean(t)),
            "target_std": float(np.std(t)),
            "win_rate": win,
            "draw_rate": draw,
            "loss_rate": loss,
            "obs_per_position_mean": float(np.mean(obs)),
            "obs_per_position_percentiles": _percentiles(obs.astype(np.float64)),
        }

    multi = int(np.sum(obs_counts > 1))
    report = {
        "n_positions": n,
        "n_multi_label_positions": multi,
        "multi_label_fraction": multi / n if n else 0.0,
        "observation_count_summary": {
            "mean": float(np.mean(obs_counts)) if n else 0.0,
            "max": int(np.max(obs_counts)) if n else 0,
            **_percentiles(obs_counts.astype(np.float64)),
        },
        "train": _slice_stats(train_idx),
        "val": _slice_stats(val_idx),
        "label_aggregation": "mean_value_i16_per_position_key_at_cache_build",
    }
    out = cache_dir / "cache_balance.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"  balance report -> {out}", flush=True)
    return out
