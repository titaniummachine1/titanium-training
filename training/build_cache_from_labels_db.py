#!/usr/bin/env python3
"""Build a ws20 feature cache from a db_import labels.db.

This is for isolated segments such as generated self-play.  It keeps those
segments trainable with CachedDataset + position_usage retirement without
promoting them into the main teacher Parquet dataset.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

_TRAINING = Path(__file__).resolve().parent
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

from build_feature_cache import FV_LEN, FEATURE_SCHEMA, make_fingerprint, record_to_fv
from label_resolution import resolve_position_label_bundle
from label_weights import game_phase_from_record
from titanium_training.validation.engine_identity import load_expected_stamp


def main() -> int:
    from prep_guard import guard_real_work

    guard_real_work("dataset_finalization", detail="build_cache_from_labels_db.py")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels-db", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--val-fraction", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=20260625)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not args.labels_db.is_file():
        raise FileNotFoundError(args.labels_db)
    cache_dir = args.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    if any(cache_dir.iterdir()) and not args.force:
        raise RuntimeError(f"{cache_dir} is not empty; pass --force to overwrite")
    if args.force:
        for name in (
            "positions.bin",
            "row_position_keys.npy",
            "observation_counts.npy",
            "train_indices.npy",
            "val_indices.npy",
            "usage_counts.npy",
            "meta.json",
        ):
            (cache_dir / name).unlink(missing_ok=True)

    con = sqlite3.connect(str(args.labels_db))
    label_rows = con.execute(
        """
        SELECT l.pos_key, l.source, l.value_stm, l.n_samples, p.position_data
        FROM labels l
        JOIN positions p ON p.pos_key = l.pos_key
        ORDER BY l.pos_key, l.source
        """
    ).fetchall()
    con.close()
    if not label_rows:
        raise RuntimeError(f"no labels in {args.labels_db}")

    by_pos: dict[str, dict] = {}
    for pos_key, source, value_stm, n_samples, data in label_rows:
        entry = by_pos.setdefault(
            str(pos_key),
            {"labels": [], "data": data, "n_samples": 0},
        )
        entry["labels"].append((str(source), float(value_stm), int(n_samples or 1)))
        entry["n_samples"] += int(n_samples or 1)

    rows: list[tuple[str, bytes, float, int]] = []
    for pos_key, entry in by_pos.items():
        raw = entry["data"]
        try:
            rec = json.loads(bytes(raw).decode("utf-8") if isinstance(raw, bytes) else raw)
            phase = game_phase_from_record(rec)
        except (json.JSONDecodeError, UnicodeDecodeError):
            phase = "midgame"
        bundle = resolve_position_label_bundle(entry["labels"], game_phase=phase)
        if bundle is None:
            continue
        rows.append((pos_key, entry["data"], bundle.target, bundle.position_occurrence_count))

    vectors: list[np.ndarray] = []
    keys: list[str] = []
    obs: list[int] = []
    failed = 0
    for pos_key, data, value_stm, n_samples in rows:
        rec = json.loads(bytes(data).decode("utf-8") if isinstance(data, bytes) else data)
        target = (float(value_stm) + 1.0) / 2.0
        fv = record_to_fv(rec, target)
        if fv is None:
            failed += 1
            continue
        vectors.append(fv)
        keys.append(str(pos_key))
        obs.append(int(n_samples or 1))

    n_total = len(vectors)
    if n_total == 0:
        raise RuntimeError(f"all {len(rows)} rows failed featurization")

    mmap = np.memmap(cache_dir / "positions.bin", dtype="float32", mode="w+", shape=(n_total, FV_LEN))
    for i, fv in enumerate(vectors):
        mmap[i] = fv
    mmap.flush()
    del mmap

    rng = np.random.default_rng(args.seed)
    all_idx = np.arange(n_total, dtype=np.int32)
    rng.shuffle(all_idx)
    n_val = max(1, int(round(n_total * max(0.0, min(0.5, args.val_fraction))))) if n_total > 1 else 0
    val_idx = np.sort(all_idx[:n_val]).astype(np.int32)
    train_idx = np.sort(all_idx[n_val:]).astype(np.int32)

    np.save(cache_dir / "row_position_keys.npy", np.array(keys, dtype=object))
    np.save(cache_dir / "observation_counts.npy", np.array(obs, dtype=np.int32))
    np.save(cache_dir / "train_indices.npy", train_idx)
    np.save(cache_dir / "val_indices.npy", val_idx)
    np.save(cache_dir / "usage_counts.npy", np.zeros(n_total, dtype=np.uint8))

    stamp = load_expected_stamp() or {}
    meta = make_fingerprint(
        stamp,
        n_total,
        len(train_idx),
        len(val_idx),
        manifest_hash=f"labels-db:{args.labels_db.name}",
        cohort_prefixes=("isolated_selfplay",),
    )
    meta["schema"] = FEATURE_SCHEMA
    meta["source_labels_db"] = str(args.labels_db)
    meta["failed_rows"] = failed
    (cache_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "cache_dir": str(cache_dir),
        "n_total": n_total,
        "train": int(len(train_idx)),
        "val": int(len(val_idx)),
        "failed": failed,
        "fv_len": FV_LEN,
        "schema": FEATURE_SCHEMA,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
