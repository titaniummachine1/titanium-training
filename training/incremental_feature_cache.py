"""Append new teacher positions to an existing feature cache (no full rebuild)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_TRAINING = Path(__file__).resolve().parent
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

from build_feature_cache import (
    BATCH_SIZE,
    FV_LEN,
    check_fingerprint,
    eval_packed_batch_raw,
    make_fingerprint,
    record_to_fv,
)
from titanium_training.validation.engine_identity import load_expected_stamp
from titanium_training.data.teacher_value import iter_value_only_rows
from titanium_training.data.split import _split_bucket
from titanium_training.paths import REPO_ROOT
from cache_val_split import (
    SPLIT_SEED_DEFAULT,
    VAL_FRAC_DEFAULT,
    _pos_key_bytes,
    _pos_key_hex,
    load_val_manifest,
    save_val_manifest,
)


def _wipe_cache_files(cache_dir: Path) -> None:
    for name in (
        "meta.json",
        "positions.bin",
        "row_position_keys.npy",
        "observation_counts.npy",
        "usage_counts.npy",
        "train_indices.npy",
        "val_indices.npy",
        "val_manifest.json",
        "cache_balance.json",
    ):
        (cache_dir / name).unlink(missing_ok=True)


def init_empty_cache(cache_dir: Path, *, manifest_hash: str = "pool-incremental") -> None:
    """Valid zero-row cache shell — rows are appended per game."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    _wipe_cache_files(cache_dir)
    (cache_dir / "positions.bin").write_bytes(b"")
    np.save(cache_dir / "row_position_keys.npy", np.array([], dtype=object))
    np.save(cache_dir / "observation_counts.npy", np.array([], dtype=np.int32))
    np.save(cache_dir / "usage_counts.npy", np.array([], dtype=np.uint8))
    np.save(cache_dir / "train_indices.npy", np.array([], dtype=np.int32))
    np.save(cache_dir / "val_indices.npy", np.array([], dtype=np.int32))
    stamp = load_expected_stamp() or {}
    meta = make_fingerprint(stamp, 0, 0, 0, manifest_hash=manifest_hash)
    meta["split_algorithm"] = "pool_incremental_per_game"
    meta["incremental_appends"] = 0
    (cache_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def ensure_cache_ready(cache_dir: Path, pool_dataset: Path) -> dict[str, Any]:
    """Recover invalid cache without destroying teacher_dataset_good.

    Retirement (usage_counts) never deletes rows — it only skips them at train time.
    This function must not wipe a multi-million-row cache when the 1.4M teacher
    parquet corpus still exists; rebuild from that corpus instead.
    """
    ok, reason = check_fingerprint(cache_dir)
    if ok:
        return {"ok": True, "action": "ready"}

    from titanium_training.paths import ACTIVE_TEACHER_DATASET

    good = ACTIVE_TEACHER_DATASET
    good_manifest = good / "manifest.json"
    if good_manifest.is_file():
        try:
            good_n = int(json.loads(good_manifest.read_text(encoding="utf-8"))["counts"]["positions"])
        except Exception:
            good_n = 0
        if good_n > 100_000:
            return {
                "ok": False,
                "action": "needs_rebuild_from_good",
                "reason": reason,
                "teacher_dataset": str(good),
                "teacher_positions": good_n,
                "message": (
                    f"Cache invalid ({reason}) but teacher_dataset_good has {good_n:,} positions "
                    f"intact. Run build_feature_cache from good — do NOT init_empty_cache."
                ),
            }

    init_empty_cache(cache_dir, manifest_hash=f"pool-recover:{reason}")
    stats = append_new_positions(cache_dir, pool_dataset)
    return {
        "ok": True,
        "action": "recovered_pool_only",
        "reason": reason,
        "bulk_import": stats,
    }


def _iter_dataset_rows(dataset_dir: Path):
    """Yield value rows from teacher parquet; works with or without manifest.json."""
    manifest_path = dataset_dir / "manifest.json"
    if manifest_path.is_file():
        yield from iter_value_only_rows(dataset_dir, root=REPO_ROOT)
        return

    labels_path = dataset_dir / "labels" / "part-00000.parquet"
    positions_path = dataset_dir / "positions" / "part-00000.parquet"
    if not labels_path.is_file() or not positions_path.is_file():
        return

    import pyarrow.parquet as pq

    labels = pq.read_table(labels_path, columns=["position_key", "value_i16", "source_cohort"])
    positions = pq.read_table(positions_path, columns=["position_key", "packed_state", "side_to_move"])
    pos_by_key = {positions.column("position_key")[i].as_py(): i for i in range(positions.num_rows)}
    for i in range(labels.num_rows):
        value_i16 = labels.column("value_i16")[i].as_py()
        if value_i16 is None:
            continue
        pos_key = labels.column("position_key")[i].as_py()
        pos_i = pos_by_key.get(pos_key)
        if pos_i is None:
            yield {"_missing_position": True, "position_key": pos_key}
            continue
        yield {
            "position_key": pos_key,
            "packed_state": positions.column("packed_state")[pos_i].as_py(),
            "side_to_move": int(positions.column("side_to_move")[pos_i].as_py()),
            "value_i16": int(value_i16),
            "source_cohort": str(labels.column("source_cohort")[i].as_py() or ""),
        }


def _collect_new_specs(dataset_dir: Path, existing_keys: set[Any]) -> list[tuple[Any, bytes, float, int, int]]:
    """Return (position_key, packed, target_p0, side_to_move, obs_n) for keys not in cache."""
    pos_data: dict[Any, list] = {}
    for r in _iter_dataset_rows(dataset_dir):
        if r.get("_missing_position"):
            continue
        key = r["position_key"]
        if key in existing_keys:
            continue
        packed = bytes(r["packed_state"])
        if len(packed) != 24:
            continue
        if key in pos_data:
            pos_data[key][2].append(int(r["value_i16"]))
            continue
        pos_data[key] = [packed, int(r["side_to_move"]), [int(r["value_i16"])]]

    specs: list[tuple[Any, bytes, float, int, int]] = []
    for key, d in pos_data.items():
        target = (sum(d[2]) / len(d[2]) / 100.0 + 1.0) / 2.0
        specs.append((key, d[0], target, d[1], len(d[2])))
    return specs


def _rows_to_specs(rows: list[dict]) -> list[tuple[Any, bytes, float, int, int]]:
    specs: list[tuple[Any, bytes, float, int, int]] = []
    for rec in rows:
        target = (int(rec["value_i16"]) / 100.0 + 1.0) / 2.0
        specs.append((rec["pk"], rec["packed"], target, int(rec["side_to_move"]), 1))
    return specs


def _assign_incremental_split(
    cache_dir: Path,
    row_keys: list[Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Assign all cache rows to train/val with a stable position-key split.

    The incremental cache appends rows one game at a time, so it cannot rely on
    build_feature_cache.py to create a held-out set.  Persist validation keys in
    val_manifest.json and extend that set deterministically for new keys.
    """
    existing = load_val_manifest(cache_dir) or {}
    val_fraction = float(existing.get("val_fraction", VAL_FRAC_DEFAULT))
    seed = int(existing.get("split_seed", SPLIT_SEED_DEFAULT))
    val_fraction = max(0.0, min(0.5, val_fraction))

    val_keys = set(existing.get("val_position_keys_hex") or [])
    if val_fraction > 0:
        for key in row_keys:
            hx = _pos_key_hex(key)
            if hx in val_keys:
                continue
            if _split_bucket(_pos_key_bytes(key), seed) < val_fraction:
                val_keys.add(hx)

    if not val_keys and row_keys and val_fraction > 0:
        # Tiny caches can miss a 5% hash bucket. Keep at least one row held out.
        best_key = min(row_keys, key=lambda key: _split_bucket(_pos_key_bytes(key), seed ^ 0xA5A5))
        val_keys.add(_pos_key_hex(best_key))

    train_rows: list[int] = []
    val_rows: list[int] = []
    for i, key in enumerate(row_keys):
        if _pos_key_hex(key) in val_keys:
            val_rows.append(i)
        else:
            train_rows.append(i)

    rng = np.random.default_rng(seed)
    train_arr = rng.permutation(np.array(train_rows, dtype=np.int32))
    val_arr = np.array(val_rows, dtype=np.int32)

    manifest = dict(existing)
    manifest.update(
        {
            "split_algorithm": "incremental_position_key_hash",
            "split_seed": seed,
            "val_fraction": val_fraction,
            "val_position_keys_hex": sorted(val_keys),
            "n_val_position_keys": len(val_keys),
        }
    )
    save_val_manifest(cache_dir, manifest)
    return train_arr, val_arr, manifest


def _append_specs(cache_dir: Path, specs: list[tuple[Any, bytes, float, int, int]]) -> dict[str, Any]:
    if not specs:
        return {"ok": True, "appended": 0, "needs_full_rebuild": False}

    ok, reason = check_fingerprint(cache_dir)
    if not ok:
        return {"ok": False, "appended": 0, "reason": reason, "needs_full_rebuild": False}

    meta_path = cache_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    n_old = int(meta["n_total"])
    keys_path = cache_dir / "row_position_keys.npy"
    if not keys_path.is_file():
        return {"ok": False, "appended": 0, "reason": "row_position_keys.npy missing", "needs_full_rebuild": False}

    existing_keys = set(np.load(keys_path, allow_pickle=True))
    new_specs = [s for s in specs if s[0] not in existing_keys]
    if not new_specs:
        return {"ok": True, "appended": 0, "n_total": n_old, "needs_full_rebuild": False}

    pos_path = cache_dir / "positions.bin"
    n_new = len(new_specs)
    n_total = n_old + n_new

    if n_old > 0:
        old_mmap = np.memmap(pos_path, dtype="float32", mode="r", shape=(n_old, FV_LEN))
        with open(pos_path, "r+b") as f:
            f.truncate(n_total * FV_LEN * 4)
        mmap = np.memmap(pos_path, dtype="float32", mode="r+", shape=(n_total, FV_LEN))
        mmap[:n_old] = old_mmap[:]
        del old_mmap
    else:
        with open(pos_path, "wb") as f:
            f.truncate(n_total * FV_LEN * 4)
        mmap = np.memmap(pos_path, dtype="float32", mode="r+", shape=(n_total, FV_LEN))

    import gc
    gc.collect()

    if usage_path := cache_dir / "usage_counts.npy":
        if usage_path.is_file() and n_old > 0:
            _uc = np.load(usage_path)
            usage = _uc if len(_uc) == n_old else np.zeros(n_old, dtype=np.uint8)
        else:
            usage = np.zeros(n_old, dtype=np.uint8)

    obs_counts = np.load(cache_dir / "observation_counts.npy") if n_old > 0 else np.array([], dtype=np.int32)
    row_keys = list(np.load(keys_path, allow_pickle=True))
    write_at = n_old
    appended_obs: list[int] = []
    n_failed = 0

    for start in range(0, n_new, BATCH_SIZE):
        chunk = new_specs[start : start + BATCH_SIZE]
        batch_rows = [(packed, target_p0, side_to_move) for _k, packed, target_p0, side_to_move, _n in chunk]
        results = eval_packed_batch_raw(batch_rows)
        for (key, packed, target_p0, side_to_move, obs_n), rec in zip(chunk, results):
            if rec is None:
                n_failed += 1
                continue
            me = int(rec.get("turn", side_to_move))
            target = target_p0 if me == 0 else (1.0 - target_p0)
            fv = record_to_fv(rec, target)
            if fv is None:
                n_failed += 1
                continue
            mmap[write_at] = fv
            row_keys.append(key)
            appended_obs.append(int(obs_n))
            write_at += 1

    mmap.flush()
    del mmap
    gc.collect()

    n_written = write_at - n_old
    if n_written == 0:
        if n_old == 0:
            with open(pos_path, "wb") as f:
                f.truncate(0)
        return {"ok": False, "appended": 0, "reason": "featurize failed for all new rows", "failed": n_failed}

    n_total = n_old + n_written
    obs_new = np.array(appended_obs, dtype=np.int32)
    np.save(cache_dir / "observation_counts.npy", np.concatenate([obs_counts, obs_new]))
    np.save(cache_dir / "usage_counts.npy", np.concatenate([usage, np.zeros(n_written, dtype=np.uint8)]))
    np.save(keys_path, np.array(row_keys, dtype=object))

    train_idx, val_idx, val_manifest = _assign_incremental_split(cache_dir, row_keys)
    np.save(cache_dir / "train_indices.npy", train_idx)
    np.save(cache_dir / "val_indices.npy", val_idx)

    stamp = load_expected_stamp() or {}
    fp = make_fingerprint(stamp, n_total, len(train_idx), len(val_idx), manifest_hash=meta.get("manifest_hash", "pool-incremental"))
    fp["split_algorithm"] = val_manifest.get("split_algorithm", "incremental_position_key_hash")
    fp["val_manifest"] = val_manifest
    fp["incremental_appends"] = int(meta.get("incremental_appends", 0)) + n_written
    meta_path.write_text(json.dumps(fp, indent=2) + "\n", encoding="utf-8")

    return {
        "ok": True,
        "appended": n_written,
        "failed": n_failed,
        "n_total": n_total,
        "needs_full_rebuild": False,
    }


def append_game_to_cache(cache_dir: Path, moves: list[str], outcome_p0: int) -> dict[str, Any]:
    """Featurize one finished game and append new positions immediately."""
    from sync_overnight_to_teacher import positions_from_game

    rows = positions_from_game(moves, outcome_p0)
    if not rows:
        return {"ok": True, "appended": 0}
    return _append_specs(cache_dir, _rows_to_specs(rows))


def append_db_game_to_cache(cache_dir: Path, game_id: str) -> dict[str, Any]:
    """Load a persisted game from games.db and append its positions."""
    import sqlite3
    from db_import import GAMES_DB_PATH

    if not GAMES_DB_PATH.is_file():
        return {"ok": False, "appended": 0, "reason": "games.db missing"}
    con = sqlite3.connect(str(GAMES_DB_PATH), timeout=30)
    row = con.execute("SELECT outcome_p0 FROM games WHERE game_id=?", (game_id,)).fetchone()
    if not row:
        con.close()
        return {"ok": False, "appended": 0, "reason": f"game not found: {game_id}"}
    moves = [
        r[0]
        for r in con.execute(
            "SELECT move_alg FROM game_moves WHERE game_id=? ORDER BY move_num",
            (game_id,),
        )
    ]
    con.close()
    if not moves:
        return {"ok": False, "appended": 0, "reason": "no moves"}
    return append_game_to_cache(cache_dir, moves, int(row[0]))


def append_new_positions(cache_dir: Path, dataset_dir: Path) -> dict:
    """
    Featurize teacher rows whose position_key is absent from the cache and append.

    Returns stats dict.  Never triggers a full rebuild of teacher_dataset_good.
    """
    ok, reason = check_fingerprint(cache_dir)
    if not ok:
        return {"ok": False, "appended": 0, "reason": reason, "needs_full_rebuild": False}

    meta_path = cache_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    n_old = int(meta["n_total"])
    keys_path = cache_dir / "row_position_keys.npy"
    if not keys_path.is_file():
        return {"ok": False, "appended": 0, "reason": "row_position_keys.npy missing", "needs_full_rebuild": False}

    existing_keys = set(np.load(keys_path, allow_pickle=True))
    new_specs = _collect_new_specs(dataset_dir, existing_keys)
    if not new_specs:
        return {"ok": True, "appended": 0, "n_total": n_old, "needs_full_rebuild": False}

    stats = _append_specs(cache_dir, new_specs)
    stats["needs_full_rebuild"] = False
    return stats
