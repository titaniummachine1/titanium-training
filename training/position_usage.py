"""Per-position training usage counters — retire after MAX_USAGE touches.

IMPORTANT: This does NOT delete positions from positions.bin or parquet datasets.
  usage_counts.npy  uint8[N]   — increments once per epoch when a row is trained
  (retired when count >= MAX_USAGE, excluded from train indices only)

At most MAX_RETIRED_FRAC of the corpus may be retired; further touches stay at
MAX_USAGE-1 so low-visit positions keep priority.

Read-only inspection; bump happens from trainer at end of each epoch.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

MAX_USAGE = 5
MAX_RETIRED_FRAC = 0.10


def usage_path(cache_dir: Path) -> Path:
    return cache_dir / "usage_counts.npy"


def meta_path(cache_dir: Path) -> Path:
    return cache_dir / "usage_meta.json"


def load_counts(cache_dir: Path, n_total: int) -> np.ndarray:
    p = usage_path(cache_dir)
    if not p.exists():
        return np.zeros(n_total, dtype=np.uint8)
    arr = np.load(p)
    if len(arr) != n_total:
        raise ValueError(f"usage_counts length {len(arr)} != n_total {n_total}")
    return arr


def save_counts(cache_dir: Path, counts: np.ndarray) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(usage_path(cache_dir), counts)


def retired_count(counts: np.ndarray) -> int:
    return int((counts >= MAX_USAGE).sum())


def retirement_allowed(counts: np.ndarray, n_total: int) -> bool:
    if n_total <= 0:
        return False
    return (retired_count(counts) + 1) / n_total <= MAX_RETIRED_FRAC


def active_indices(cache_dir: Path, split: str = "train") -> np.ndarray:
    """Train/val index array with retired rows removed."""
    meta = json.loads((cache_dir / "meta.json").read_text(encoding="utf-8"))
    n_total = int(meta["n_total"])
    indices = np.load(cache_dir / f"{split}_indices.npy")
    counts = load_counts(cache_dir, n_total)
    retired = counts[indices] >= MAX_USAGE
    active = indices[~retired]
    return active


def epoch_indices_low_visits_first(
    cache_dir: Path,
    indices: np.ndarray,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Shuffle with lowest usage_counts first (0 before 1, …, 4 before retired)."""
    meta = json.loads((cache_dir / "meta.json").read_text(encoding="utf-8"))
    n_total = int(meta["n_total"])
    counts = load_counts(cache_dir, n_total)
    idx = np.asarray(indices, dtype=np.int64)
    visits = counts[idx].astype(np.int32)
    rng = np.random.default_rng(seed)
    tie = rng.permutation(len(idx))
    order = np.lexsort((tie, visits))
    return idx[order].astype(np.int32)


def bump_epoch(cache_dir: Path, indices_used: np.ndarray) -> dict:
    """Increment usage for all cache row indices seen this epoch."""
    meta = json.loads((cache_dir / "meta.json").read_text(encoding="utf-8"))
    n_total = int(meta["n_total"])
    counts = load_counts(cache_dir, n_total)
    unique = np.unique(indices_used.astype(np.int64))
    capped_skips = 0
    for idx in unique:
        if not (0 <= idx < n_total):
            continue
        if counts[idx] >= MAX_USAGE:
            continue
        if counts[idx] == MAX_USAGE - 1 and not retirement_allowed(counts, n_total):
            capped_skips += 1
            continue
        if counts[idx] < 255:
            counts[idx] += 1
    save_counts(cache_dir, counts)
    retired = retired_count(counts)
    stats = {
        "touched": int(len(unique)),
        "retired_total": retired,
        "retired_frac": round(retired / max(n_total, 1), 4),
        "retirement_cap_skips": capped_skips,
        "active_train": int((counts[np.load(cache_dir / "train_indices.npy")] < MAX_USAGE).sum()),
        "max_retired_frac": MAX_RETIRED_FRAC,
    }
    meta_path(cache_dir).write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def status(cache_dir: Path) -> dict:
    meta = json.loads((cache_dir / "meta.json").read_text(encoding="utf-8"))
    n_total = int(meta["n_total"])
    counts = load_counts(cache_dir, n_total)
    train_idx = np.load(cache_dir / "train_indices.npy")
    retired = retired_count(counts)
    return {
        "n_total": n_total,
        "retired": retired,
        "retired_frac": round(retired / max(n_total, 1), 4),
        "active_train": int((counts[train_idx] < MAX_USAGE).sum()),
        "max_usage": MAX_USAGE,
        "max_retired_frac": MAX_RETIRED_FRAC,
    }
