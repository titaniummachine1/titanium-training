#!/usr/bin/env python3
"""Safe sequential rebuild of feature cache and opening prefix index (protocol v2).

Phase 1a: Build feature cache from teacher_dataset_good into temp dir (long-running).
Phase 1b: Finalize — audit gaps, canonical delta, strong validation, atomic swap.
Phase 2:  Prefix index from games.db with exact game counts and delta.

Activation requires protocol v2 validation. Use --finalize-v2 after featurization.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import shutil
import sqlite3
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

_TRAINING = Path(__file__).resolve().parent
_REPO = _TRAINING.parent
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

from build_feature_cache import (
    DEFAULT_FEATURIZE_WORKERS,
    FV_LEN,
    build,
    check_fingerprint,
    eval_packed_batch_raw,
    record_to_fv,
)
from db_import import GAMES_DB_PATH, LABELS_DB_PATH
from incremental_feature_cache import append_db_game_to_cache, append_new_positions
from opening_prefix_index import (
    DEFAULT_INDEX_PATH,
    OpeningPrefixIndex,
    canonical_move_prefix,
    mirror_move_alg,
)
from sync_overnight_to_teacher import load_synced_ids, pool_teacher_dir
from teacher_scan_audit import audit_teacher_dataset_scan
from titanium_training.paths import ACTIVE_TEACHER_DATASET, DATA_DIR, REPO_ROOT

PROTOCOL_VERSION = 2
LOG_DIR = DATA_DIR / "overnight_logs"
PAUSE_EPOCHS_PATH = LOG_DIR / "pause_training_epochs.json"
REBUILD_STATE_PATH = LOG_DIR / "safe_rebuild_state.json"
OPENING_ENABLED_PATH = LOG_DIR / "opening_exploration_enabled.json"
FINALIZE_V2_REQUIRED_PATH = LOG_DIR / "REQUIRE_FINALIZE_V2.json"

LIVE_CACHE = DATA_DIR / "feature_cache"
TEMP_CACHE = DATA_DIR / "feature_cache_rebuild"
SMOKE_CACHE = DATA_DIR / "feature_cache_smoke"
LIVE_PREFIX = DEFAULT_INDEX_PATH
TEMP_PREFIX = LIVE_PREFIX.parent / "opening_prefix_index_rebuild.db"
TEACHER_GOOD = ACTIVE_TEACHER_DATASET

# v2: expect every labeled position; orphans without labels are documented, not errors.
POSITION_TOLERANCE = 0.0
MIN_PARITY_SAMPLES = 384


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def set_pause_epochs(*, reason: str, phase: str) -> None:
    _write_json(
        PAUSE_EPOCHS_PATH,
        {"paused": True, "reason": reason, "phase": phase, "since": _utc_now()},
    )


def clear_pause_epochs() -> None:
    if PAUSE_EPOCHS_PATH.is_file():
        PAUSE_EPOCHS_PATH.unlink(missing_ok=True)


def load_rebuild_state() -> dict:
    if REBUILD_STATE_PATH.is_file():
        return json.loads(REBUILD_STATE_PATH.read_text(encoding="utf-8"))
    return {}


def save_rebuild_state(state: dict) -> None:
    state["updated_at"] = _utc_now()
    state["protocol_version"] = PROTOCOL_VERSION
    _write_json(REBUILD_STATE_PATH, state)


def teacher_good_hwm() -> dict[str, Any]:
    manifest_path = TEACHER_GOOD / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pos_path = TEACHER_GOOD / "positions" / "part-00000.parquet"
    return {
        "dataset": str(TEACHER_GOOD),
        "position_count": int(manifest["counts"]["positions"]),
        "label_count": int(manifest["counts"]["labels"]),
        "manifest_mtime": pos_path.stat().st_mtime if pos_path.is_file() else None,
        "recorded_at": _utc_now(),
    }


def pool_teacher_hwm() -> dict[str, Any]:
    pool = pool_teacher_dir()
    labels_path = pool / "labels" / "part-00000.parquet"
    n_labels = 0
    if labels_path.is_file():
        import pyarrow.parquet as pq

        n_labels = pq.read_metadata(labels_path).num_rows
    return {
        "dataset": str(pool),
        "label_rows": n_labels,
        "synced_game_ids": len(load_synced_ids()),
        "recorded_at": _utc_now(),
    }


def games_db_hwm(games_db: Path = GAMES_DB_PATH) -> dict[str, Any]:
    con = sqlite3.connect(str(games_db), timeout=120)
    game_ids = [
        r[0]
        for r in con.execute(
            "SELECT game_id FROM games ORDER BY imported_at, game_id"
        ).fetchall()
    ]
    game_count = len(game_ids)
    max_imported = con.execute("SELECT MAX(imported_at) FROM games").fetchone()[0]
    con.close()
    return {
        "games_db": str(games_db),
        "game_count": game_count,
        "game_ids": game_ids,
        "max_imported_at": max_imported,
        "recorded_at": _utc_now(),
    }


def labels_db_hwm(labels_db: Path = LABELS_DB_PATH) -> dict[str, Any]:
    if not labels_db.is_file():
        return {"labels_db": str(labels_db), "position_count": 0, "recorded_at": _utc_now()}
    con = sqlite3.connect(str(labels_db), timeout=120)
    n = int(con.execute("SELECT COUNT(*) FROM positions").fetchone()[0])
    con.close()
    return {"labels_db": str(labels_db), "position_count": n, "recorded_at": _utc_now()}


def _backup_path(base: Path, suffix: str) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return base.parent / f"{base.name}_{suffix}_{ts}"


def _close_cache_handles(cache_dir: Path) -> None:
    gc.collect()
    time.sleep(0.25)


def atomic_swap_dir_safe(src: Path, live: Path, backup: Path, *, max_retries: int = 8) -> dict[str, Any]:
    """Windows-safe directory swap with retries and rollback on failure."""
    _close_cache_handles(live)
    _close_cache_handles(src)
    backup.parent.mkdir(parents=True, exist_ok=True)
    last_err: str | None = None
    for attempt in range(1, max_retries + 1):
        try:
            if backup.exists():
                shutil.rmtree(backup)
            if live.exists():
                shutil.move(str(live), str(backup))
            if not src.exists():
                raise FileNotFoundError(f"temp cache missing: {src}")
            shutil.move(str(src), str(live))
            return {"ok": True, "attempt": attempt, "backup": str(backup), "live": str(live)}
        except OSError as exc:
            last_err = str(exc)
            gc.collect()
            time.sleep(0.75 * attempt)
            if live.exists() is False and backup.exists():
                try:
                    shutil.move(str(backup), str(live))
                except OSError:
                    pass
    return {"ok": False, "error": last_err, "backup": str(backup), "live": str(live)}


def atomic_swap_file_safe(src: Path, live: Path, backup: Path, *, max_retries: int = 8) -> dict[str, Any]:
    gc.collect()
    backup.parent.mkdir(parents=True, exist_ok=True)
    last_err: str | None = None
    for attempt in range(1, max_retries + 1):
        try:
            if live.is_file():
                shutil.copy2(live, backup)
                live.unlink()
            shutil.move(str(src), str(live))
            return {"ok": True, "attempt": attempt, "backup": str(backup), "live": str(live)}
        except OSError as exc:
            last_err = str(exc)
            gc.collect()
            time.sleep(0.75 * attempt)
            if not live.is_file() and backup.is_file():
                try:
                    shutil.copy2(backup, live)
                except OSError:
                    pass
    return {"ok": False, "error": last_err}


def append_canonical_delta(
    cache_dir: Path,
    games_hwm: dict[str, Any],
    pool_hwm: dict[str, Any],
) -> dict[str, Any]:
    """Append every position from games committed after HWM + pool teacher safety net.

    Both paths deduplicate by canonical position_key via row_position_keys.npy;
    pool rows already present from games.db append are skipped (not double-counted).
    """
    meta_path = cache_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    n_total_before = int(meta.get("n_total", 0))
    keys_path = cache_dir / "row_position_keys.npy"
    keys_before_games: set[Any] = set()
    if keys_path.is_file() and n_total_before > 0:
        keys_before_games = set(np.load(keys_path, allow_pickle=True))

    hwm_ids = set(games_hwm.get("game_ids") or [])
    con = sqlite3.connect(str(games_hwm["games_db"]), timeout=120)
    all_rows = con.execute(
        "SELECT game_id, source FROM games ORDER BY imported_at, game_id"
    ).fetchall()
    con.close()
    new_game_ids = [gid for gid, _src in all_rows if gid not in hwm_ids]

    per_game: list[dict[str, Any]] = []
    positions_from_games = 0
    game_failures = 0
    for gid in new_game_ids:
        stats = append_db_game_to_cache(cache_dir, gid)
        per_game.append({"game_id": gid, **stats})
        if stats.get("ok"):
            positions_from_games += int(stats.get("appended", 0) or 0)
        else:
            game_failures += 1

    meta_mid = json.loads(meta_path.read_text(encoding="utf-8"))
    n_total_after_games = int(meta_mid.get("n_total", 0))
    keys_after_games = set(np.load(keys_path, allow_pickle=True)) if keys_path.is_file() else set()
    keys_added_by_games = len(keys_after_games - keys_before_games)

    pool_path = Path(pool_hwm["dataset"])
    pool_would_dup = 0
    from incremental_feature_cache import _collect_new_specs

    pool_candidate_specs = _collect_new_specs(pool_path, keys_after_games)

    pool_stats = append_new_positions(cache_dir, pool_path)
    pool_appended = int(pool_stats.get("appended", 0) or 0)

    meta_after = json.loads(meta_path.read_text(encoding="utf-8"))
    n_total_after = int(meta_after.get("n_total", 0))
    row_delta_integrity = n_total_after == n_total_before + keys_added_by_games + pool_appended

    pool_hwm_after = pool_teacher_hwm()

    return {
        "games_hwm": games_hwm,
        "pool_hwm": pool_hwm,
        "new_game_ids": new_game_ids,
        "new_game_count": len(new_game_ids),
        "n_total_before": n_total_before,
        "n_total_after_games": n_total_after_games,
        "n_total_after": n_total_after,
        "positions_appended_from_games": positions_from_games,
        "keys_added_by_games": keys_added_by_games,
        "pool_candidate_new_keys": len(pool_candidate_specs),
        "pool_appended": pool_appended,
        "pool_skipped_already_in_cache": max(0, len(pool_candidate_specs) - pool_appended),
        "game_failures": game_failures,
        "row_delta_integrity_ok": row_delta_integrity,
        "per_game_sample": per_game[:20],
        "pool_append": pool_stats,
        "pool_hwm_after": pool_hwm_after,
    }


def _teacher_key_specs() -> tuple[dict[Any, dict], dict[Any, list[str]]]:
    """position_key -> spec; cohort -> keys (for stratified sampling)."""
    from titanium_training.data.teacher_value import iter_value_only_rows

    key_specs: dict[Any, dict] = {}
    cohort_keys: dict[Any, list[str]] = defaultdict(list)
    pos_data: dict[Any, list] = {}
    for r in iter_value_only_rows(TEACHER_GOOD, root=REPO_ROOT):
        if r.get("_missing_position"):
            continue
        key = r["position_key"]
        pos_data.setdefault(key, [bytes(r["packed_state"]), int(r["side_to_move"]), []])
        pos_data[key][2].append(int(r["value_i16"]))
    for key, d in pos_data.items():
        target_p0 = (sum(d[2]) / len(d[2]) / 100.0 + 1.0) / 2.0
        key_specs[key] = {
            "packed": d[0],
            "target_p0": target_p0,
            "side_to_move": d[1],
            "obs_n": len(d[2]),
        }
    return key_specs, cohort_keys


def validate_feature_cache_v2(
    cache_dir: Path,
    *,
    scan_audit: dict[str, Any],
    sample_n: int = MIN_PARITY_SAMPLES,
    build_rows_before_delta: int | None = None,
) -> dict[str, Any]:
    ok, reason = check_fingerprint(cache_dir)
    meta = json.loads((cache_dir / "meta.json").read_text(encoding="utf-8"))
    n_total = int(meta.get("n_total", 0))
    fv_len = int(meta.get("fv_len", 0))
    keys = np.load(cache_dir / "row_position_keys.npy", allow_pickle=True)
    mmap = np.memmap(cache_dir / "positions.bin", dtype="float32", mode="r", shape=(n_total, fv_len))
    obs = np.load(cache_dir / "observation_counts.npy")
    usage = np.load(cache_dir / "usage_counts.npy", allow_pickle=False) if (cache_dir / "usage_counts.npy").is_file() else None

    expected_unique = int(scan_audit["cache_build_would_emit"])
    count_ok = n_total >= expected_unique
    build_rows_ok = (
        build_rows_before_delta is None or int(build_rows_before_delta) >= expected_unique
    )
    fv_ok = fv_len == FV_LEN
    meta_align_ok = len(keys) == n_total == mmap.shape[0] == len(obs)
    usage_ok = usage is not None and len(usage) == n_total

    key_specs, _cohort = _teacher_key_specs()
    key_index = {keys[i]: i for i in range(n_total)}

    must_check: set[int] = {0, 1, n_total - 1, n_total // 2}
    rng = random.Random(42)
    pool = [k for k in key_specs if k in key_index]
    rng.shuffle(pool)
    for k in pool:
        if len(must_check) >= sample_n:
            break
        must_check.add(key_index[k])

    sample_errors: list[str] = []
    checked = 0
    for row_i in sorted(must_check):
        if row_i < 0 or row_i >= n_total:
            continue
        key = keys[row_i]
        spec = key_specs.get(key)
        if spec is None:
            sample_errors.append(f"row={row_i} key not in teacher specs")
            continue
        packed, target_p0, stm, obs_n = (
            spec["packed"],
            spec["target_p0"],
            spec["side_to_move"],
            spec["obs_n"],
        )
        if int(obs[row_i]) != int(obs_n):
            sample_errors.append(f"row={row_i} obs_count cache={obs[row_i]} teacher={obs_n}")
        recs = eval_packed_batch_raw([(packed, target_p0, stm)], timeout_sec=120)
        rec = recs[0]
        if rec is None:
            sample_errors.append(f"row={row_i} eval failed")
            continue
        me = int(rec.get("turn", stm))
        target = target_p0 if me == 0 else (1.0 - target_p0)
        fv = record_to_fv(rec, target)
        if fv is None:
            sample_errors.append(f"row={row_i} record_to_fv failed")
            continue
        if not np.allclose(mmap[row_i], fv, rtol=0, atol=1e-4):
            sample_errors.append(
                f"row={row_i} fv max_diff={float(np.max(np.abs(mmap[row_i] - fv)))}"
            )
        if abs(float(mmap[row_i, 0]) - float(target)) > 1e-4:
            sample_errors.append(f"row={row_i} target mismatch")
        checked += 1

    del mmap
    gc.collect()

    scan_ok = bool(scan_audit.get("fully_explained"))
    sample_ok = len(sample_errors) == 0
    passed = (
        ok
        and count_ok
        and build_rows_ok
        and fv_ok
        and meta_align_ok
        and usage_ok
        and scan_ok
        and sample_ok
        and checked >= min(sample_n, n_total, len(pool))
    )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "passed": passed,
        "fingerprint_ok": ok,
        "fingerprint_reason": reason,
        "n_total": n_total,
        "expected_unique_labeled": expected_unique,
        "build_rows_before_delta": build_rows_before_delta,
        "build_rows_ok": build_rows_ok,
        "fv_len": fv_len,
        "count_ok": count_ok,
        "meta_align_ok": meta_align_ok,
        "usage_ok": usage_ok,
        "scan_audit": scan_audit,
        "sample_checked": checked,
        "sample_target": sample_n,
        "sample_ok": sample_ok,
        "sample_errors": sample_errors[:25],
        "note_policy": "Value cache has no policy offsets; policy lives in teacher_dataset_good sidecars.",
    }


def validate_prefix_index_v2(
    idx: OpeningPrefixIndex,
    build_stats: dict[str, Any],
    *,
    games_db: Path = GAMES_DB_PATH,
) -> dict[str, Any]:
    n = idx.total_prefixes()
    mirror_ok = canonical_move_prefix(["e2", "e8", "d2", "f8"]) == canonical_move_prefix(
        ["e2", "e8", "f2", "d8"]
    )
    assert mirror_move_alg("a3h") == "i3h"
    top = idx.frequency_distribution(max_ply=16, limit=5)

    game_store_count = None
    game_store_path = DATA_DIR / "canonical" / "game_store.db"
    if game_store_path.is_file():
        con = sqlite3.connect(str(game_store_path), timeout=30)
        game_store_count = int(con.execute("SELECT COUNT(*) FROM games").fetchone()[0])
        con.close()

    indexed = int(build_stats.get("games_indexed", 0))
    total_db = int(build_stats.get("games_total_in_db", 0))
    skipped = int(build_stats.get("games_skipped_no_moves", 0)) + int(
        build_stats.get("games_skipped_empty_moves", 0)
    )
    passed = (
        n >= 1000
        and mirror_ok
        and indexed > 0
        and indexed + skipped == total_db
    )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "passed": passed,
        "prefix_count": n,
        "mirror_ok": mirror_ok,
        "games_total_in_db": total_db,
        "games_indexed": indexed,
        "games_skipped": skipped,
        "game_store_db_games": game_store_count,
        "top_prefixes": top,
        "note": "Prefix index sources canonical games.db (overnight/oracle ingestion), not game_store.db.",
    }


def smoke_validate_cache(cache_dir: Path, *, sample_n: int = 32) -> dict[str, Any]:
    """Lightweight validation for cache-smoke: fingerprint, memmap, parity samples."""
    ok, reason = check_fingerprint(cache_dir)
    if not ok:
        return {"passed": False, "reason": reason}
    meta = json.loads((cache_dir / "meta.json").read_text(encoding="utf-8"))
    n_total = int(meta["n_total"])
    fv_len = int(meta["fv_len"])
    keys = np.load(cache_dir / "row_position_keys.npy", allow_pickle=True)
    mmap = np.memmap(cache_dir / "positions.bin", dtype="float32", mode="r", shape=(n_total, fv_len))
    del mmap
    mmap = np.memmap(cache_dir / "positions.bin", dtype="float32", mode="r", shape=(n_total, fv_len))
    key_specs, _ = _teacher_key_specs()
    key_index = {keys[i]: i for i in range(n_total)}
    must = sorted({0, n_total - 1, n_total // 2} | set(range(min(sample_n, n_total))))
    errors: list[str] = []
    for row_i in must:
        if row_i >= n_total:
            continue
        key = keys[row_i]
        spec = key_specs.get(key)
        if spec is None:
            errors.append(f"row={row_i} key missing from teacher specs")
            continue
        recs = eval_packed_batch_raw([(spec["packed"], spec["target_p0"], spec["side_to_move"])], timeout_sec=120)
        rec = recs[0]
        if rec is None:
            errors.append(f"row={row_i} eval failed")
            continue
        cached_target = float(mmap[row_i, 0])
        fv = record_to_fv(rec, cached_target)
        if fv is None or not np.allclose(mmap[row_i], fv, rtol=0, atol=1e-4):
            errors.append(
                f"row={row_i} parity mismatch max_diff="
                f"{float(np.max(np.abs(mmap[row_i] - fv))) if fv is not None else -1}"
            )
    del mmap
    return {"passed": not errors, "errors": errors, "n_total": n_total, "parity_checked": len(must)}


def run_cache_smoke(*, workers: int, max_positions: int, eval_timeout_sec: int = 900) -> dict[str, Any]:
    print(f"=== cache-smoke: {max_positions} positions ===", flush=True)
    if SMOKE_CACHE.exists():
        shutil.rmtree(SMOKE_CACHE)
    SMOKE_CACHE.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    build(
        SMOKE_CACHE,
        TEACHER_GOOD,
        workers=workers,
        max_positions=max_positions,
        eval_timeout_sec=eval_timeout_sec,
        resume=False,
    )
    elapsed = time.perf_counter() - t0
    validation = smoke_validate_cache(SMOKE_CACHE)
    result = {
        "ok": validation["passed"],
        "elapsed_sec": round(elapsed, 1),
        "cache_dir": str(SMOKE_CACHE),
        "validation": validation,
    }
    if not validation["passed"]:
        raise RuntimeError(f"cache-smoke validation failed: {validation}")
    print("cache-smoke PASSED", json.dumps(result, indent=2), flush=True)
    return result


def run_cache_build(*, workers: int, eval_timeout_sec: int = 900, resume: bool = True) -> dict[str, Any]:
    print("=== Phase 1a: feature cache featurization ===", flush=True)
    set_pause_epochs(reason="feature_cache_rebuild", phase="cache_build")
    _write_json(
        FINALIZE_V2_REQUIRED_PATH,
        {
            "required": True,
            "protocol_version": PROTOCOL_VERSION,
            "message": "Do not activate until: python training/safe_rebuild.py --finalize-v2",
            "since": _utc_now(),
        },
    )
    state = load_rebuild_state()
    good_hwm = teacher_good_hwm()
    state["cache_phase"] = {
        "started_at": _utc_now(),
        "rebuild_pid": os.getpid(),
        "teacher_good_hwm": good_hwm,
        "games_hwm": games_db_hwm(),
        "labels_hwm": labels_db_hwm(),
        "pool_teacher_hwm": pool_teacher_hwm(),
        "temp_cache": str(TEMP_CACHE),
        "live_cache": str(LIVE_CACHE),
        "build_only": True,
    }
    save_rebuild_state(state)

    if TEMP_CACHE.exists() and not resume:
        shutil.rmtree(TEMP_CACHE)
    TEMP_CACHE.mkdir(parents=True, exist_ok=True)

    print(f"Building from {TEACHER_GOOD} -> {TEMP_CACHE} (resume={resume})", flush=True)
    t0 = time.perf_counter()
    build(TEMP_CACHE, TEACHER_GOOD, workers=workers, eval_timeout_sec=eval_timeout_sec, resume=resume)
    elapsed = time.perf_counter() - t0
    state["cache_phase"]["build_elapsed_sec"] = round(elapsed, 1)
    state["cache_phase"]["build_completed_at"] = _utc_now()
    save_rebuild_state(state)
    print(f"Featurization complete in {elapsed:.1f}s — run --finalize-v2 before activation", flush=True)
    return {"ok": True, "elapsed_sec": elapsed, "temp_cache": str(TEMP_CACHE)}


def finalize_cache_v2(*, allow_activation: bool) -> dict[str, Any]:
    print("=== Phase 1b: cache finalize (audit + delta + validation) ===", flush=True)
    if not (TEMP_CACHE / "meta.json").is_file():
        raise FileNotFoundError(f"temp cache incomplete: {TEMP_CACHE / 'meta.json'}")

    meta = json.loads((TEMP_CACHE / "meta.json").read_text(encoding="utf-8"))
    n_total = int(meta.get("n_total", 0))
    build_rows_before_delta = n_total
    if not (TEMP_CACHE / "usage_counts.npy").is_file() and n_total > 0:
        np.save(TEMP_CACHE / "usage_counts.npy", np.zeros(n_total, dtype=np.uint8))

    state = load_rebuild_state()
    cache_phase = state.get("cache_phase") or {}
    games_hwm = cache_phase.get("games_hwm") or games_db_hwm()
    pool_hwm = cache_phase.get("pool_teacher_hwm") or pool_teacher_hwm()

    scan_audit = audit_teacher_dataset_scan(TEACHER_GOOD)
    _write_json(LOG_DIR / "teacher_scan_audit.json", scan_audit)
    print("Teacher scan audit:", json.dumps(scan_audit["reconciliation"], indent=2), flush=True)

    if not scan_audit.get("fully_explained"):
        raise RuntimeError(f"Unexplained position gap: {scan_audit.get('unexplained_gap')}")

    delta = append_canonical_delta(TEMP_CACHE, games_hwm, pool_hwm)
    if not delta.get("row_delta_integrity_ok"):
        raise RuntimeError(f"delta row count integrity failed: {delta}")
    validation = validate_feature_cache_v2(
        TEMP_CACHE,
        scan_audit=scan_audit,
        build_rows_before_delta=build_rows_before_delta,
    )

    backup = _backup_path(LIVE_CACHE, "pre_rebuild")
    activated = False
    swap_result: dict[str, Any] | None = None
    if allow_activation and validation["passed"]:
        _close_cache_handles(LIVE_CACHE)
        swap_result = atomic_swap_dir_safe(TEMP_CACHE, LIVE_CACHE, backup)
        activated = bool(swap_result.get("ok"))
        if not activated:
            validation["passed"] = False
            validation["swap_error"] = swap_result

    cache_phase.update(
        {
            "build_rows_before_delta": build_rows_before_delta,
            "scan_audit": scan_audit,
            "delta": delta,
            "validation": validation,
            "swap_result": swap_result,
            "activated": activated,
            "backup_cache": str(backup) if activated else None,
            "finalize_completed_at": _utc_now(),
        }
    )
    state["cache_phase"] = cache_phase
    save_rebuild_state(state)

    if activated:
        if FINALIZE_V2_REQUIRED_PATH.is_file():
            FINALIZE_V2_REQUIRED_PATH.unlink(missing_ok=True)

    return {
        "activated": activated,
        "validation": validation,
        "delta": delta,
        "scan_audit": scan_audit,
        "paths": {
            "temp": str(TEMP_CACHE),
            "live": str(LIVE_CACHE),
            "backup": str(backup) if activated else None,
        },
    }


def run_prefix_phase(*, max_ply: int = 16, skip_build: bool = False, allow_activation: bool = True) -> dict[str, Any]:
    print("=== Phase 2: opening prefix index rebuild ===", flush=True)
    set_pause_epochs(reason="prefix_index_rebuild", phase="prefix")
    state = load_rebuild_state()
    games_hwm = games_db_hwm()
    state["prefix_phase"] = {
        "started_at": _utc_now(),
        "games_db_hwm": games_hwm,
        "temp_index": str(TEMP_PREFIX),
        "live_index": str(LIVE_PREFIX),
        "max_ply": max_ply,
    }
    save_rebuild_state(state)

    if TEMP_PREFIX.is_file():
        TEMP_PREFIX.unlink()
    idx = OpeningPrefixIndex(TEMP_PREFIX)
    build_stats: dict[str, Any] = {}
    try:
        if not skip_build:
            print(f"Building prefix index from {GAMES_DB_PATH}", flush=True)
            build_stats = idx.build_from_games_db(max_ply=max_ply, batch_log=1000)
            print(json.dumps(build_stats, indent=2), flush=True)
        else:
            build_stats = {"games_total_in_db": games_hwm["game_count"], "games_indexed": 0}

        hwm_ids = set(games_hwm.get("game_ids") or [])
        con = sqlite3.connect(str(GAMES_DB_PATH), timeout=120)
        delta_rows = [
            r
            for r in con.execute(
                "SELECT game_id, outcome_p0, source FROM games ORDER BY imported_at, game_id"
            ).fetchall()
            if r[0] not in hwm_ids
        ]
        con.close()
        delta_registered = 0
        for game_id, outcome_p0, source in delta_rows:
            con = sqlite3.connect(str(GAMES_DB_PATH), timeout=120)
            moves = [
                r[0]
                for r in con.execute(
                    "SELECT move_alg FROM game_moves WHERE game_id=? ORDER BY move_num",
                    (game_id,),
                )
            ]
            con.close()
            if moves:
                idx.register_game(moves, int(outcome_p0), source=str(source), max_ply=max_ply)
                delta_registered += 1

        build_stats["delta_games"] = len(delta_rows)
        build_stats["delta_registered"] = delta_registered
        build_stats["games_hwm_after"] = games_db_hwm()
        validation = validate_prefix_index_v2(idx, build_stats)
    finally:
        idx.close()

    backup = _backup_path(LIVE_PREFIX, "pre_rebuild")
    activated = False
    swap_result: dict[str, Any] | None = None
    if allow_activation and validation["passed"]:
        swap_result = atomic_swap_file_safe(TEMP_PREFIX, LIVE_PREFIX, backup)
        activated = bool(swap_result.get("ok"))
        if activated:
            _write_json(
                OPENING_ENABLED_PATH,
                {
                    "enabled": True,
                    "since": _utc_now(),
                    "prefix_count": validation["prefix_count"],
                    "protocol_version": PROTOCOL_VERSION,
                },
            )

    state["prefix_phase"].update(
        {
            "build_stats": build_stats,
            "validation": validation,
            "swap_result": swap_result,
            "activated": activated,
            "backup_index": str(backup) if activated else None,
            "completed_at": _utc_now(),
        }
    )
    save_rebuild_state(state)
    clear_pause_epochs()
    return {
        "activated": activated,
        "validation": validation,
        "build_stats": build_stats,
        "opening_exploration_enabled": activated,
        "paths": {
            "temp": str(TEMP_PREFIX),
            "live": str(LIVE_PREFIX),
            "backup": str(backup) if activated else None,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=("cache", "prefix", "all", "build-only", "cache-smoke"), default="build-only")
    ap.add_argument("--max-positions", type=int, default=None, help="Limit positions (cache-smoke)")
    ap.add_argument("--finalize-v2", action="store_true", help="Audit, delta, validate, optionally activate cache")
    ap.add_argument("--allow-activation", action="store_true", help="Permit atomic swap when validation passes")
    ap.add_argument("--workers", type=int, default=max(1, min(DEFAULT_FEATURIZE_WORKERS, (__import__("os").cpu_count() or 4) // 2)))
    ap.add_argument("--eval-timeout-sec", type=int, default=900,
                    help="Per-batch eval-packed-batch timeout (default 900s)")
    ap.add_argument("--skip-prefix-build", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    if args.report_only:
        print(json.dumps(load_rebuild_state(), indent=2))
        audit_path = LOG_DIR / "teacher_scan_audit.json"
        if audit_path.is_file():
            print("\n--- teacher_scan_audit ---")
            print(audit_path.read_text(encoding="utf-8"))
        return 0

    report: dict[str, Any] = {"protocol_version": PROTOCOL_VERSION, "started_at": _utc_now()}
    try:
        if args.finalize_v2:
            report["cache_finalize"] = finalize_cache_v2(allow_activation=args.allow_activation)
            if args.allow_activation and not report["cache_finalize"]["activated"]:
                print("CACHE NOT ACTIVATED", flush=True)
                return 1
            if args.phase in ("prefix", "all") or args.allow_activation:
                report["prefix"] = run_prefix_phase(
                    max_ply=args.max_ply,
                    skip_build=args.skip_prefix_build,
                    allow_activation=args.allow_activation,
                )
                if args.allow_activation and not report["prefix"]["activated"]:
                    return 1
            return 0

        if args.phase == "cache-smoke":
            n = int(args.max_positions or 8192)
            report["cache_smoke"] = run_cache_smoke(
                workers=args.workers,
                max_positions=n,
                eval_timeout_sec=args.eval_timeout_sec,
            )
            report["finished_at"] = _utc_now()
            _write_json(LOG_DIR / "safe_rebuild_report.json", report)
            return 0

        if args.phase in ("cache", "all", "build-only"):
            report["cache_build"] = run_cache_build(
                workers=args.workers,
                eval_timeout_sec=args.eval_timeout_sec,
                resume=True,
            )
            print(
                "\nFeaturization done. Before activation run:\n"
                "  python training/safe_rebuild.py --finalize-v2 --allow-activation\n"
                "  python training/safe_rebuild.py --finalize-v2 --allow-activation  # includes prefix if --phase all via second command with prefix",
                flush=True,
            )
            if args.phase == "build-only":
                report["finished_at"] = _utc_now()
                _write_json(LOG_DIR / "safe_rebuild_report.json", report)
                return 0

        if args.phase == "prefix":
            report["prefix"] = run_prefix_phase(
                max_ply=args.max_ply,
                skip_build=args.skip_prefix_build,
                allow_activation=args.allow_activation,
            )
    except Exception as exc:
        import traceback

        report["status"] = "FAILED"
        report["failure"] = str(exc)
        report["traceback"] = traceback.format_exc()
        report["rebuild_pid"] = os.getpid()
        report["pause_kept"] = PAUSE_EPOCHS_PATH.is_file()
        report["finished_at"] = _utc_now()
        _write_json(LOG_DIR / "safe_rebuild_report.json", report)
        raise

    report["finished_at"] = _utc_now()
    _write_json(LOG_DIR / "safe_rebuild_report.json", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
