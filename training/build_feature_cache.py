#!/usr/bin/env python3
"""
Build full-corpus feature cache for value-NNUE training.

Featurizes all teacher-dataset positions via titanium eval-packed-batch
and stores compact float32 feature vectors in a versioned disk cache.
Training then streams from the cache each epoch without re-calling the engine.

Cache layout:
  <cache_dir>/meta.json          -- version fingerprint + shape
  <cache_dir>/positions.bin      -- float32 memmap, shape (N, FV_LEN=545)
  <cache_dir>/train_indices.npy  -- int32 (N_train,), shuffled
  <cache_dir>/val_indices.npy    -- int32 (N_val,)

Feature vector offsets (FV_LEN = 545):
  [0]        target               win-prob, side-to-move perspective
  [1]        d_me
  [2]        d_opp
  [3]        w_me
  [4]        w_opp
  [5]        legal_wall_norm      legal_wall_count / 128
  [6]        width_opp            corridor width (raw count)
  [7]        cross_me_norm        legal walls crossing my path / 128
  [8]        cross_opp_norm
  [9..72]    hw[64]               horizontal wall bitmask (P1-mirrored)
  [73..136]  vw[64]               vertical wall bitmask (P1-mirrored)
  [137..217] route_me[81]
  [218..298] route_opp[81]
  [299..379] route_near_me[81]
  [380..460] route_near_opp[81]
  [461..541] route_contested[81]
  [542]      bucket               pawn bucket 0-8
  [543]      pawn_me              pawn cell 0-80
  [544]      pawn_opp

Usage:
    python training/build_feature_cache.py
    python training/build_feature_cache.py --cache-dir training/data/feature_cache
    python training/build_feature_cache.py --force   # rebuild even if current
"""
from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Bootstrap sys.path
# ---------------------------------------------------------------------------
_TRAINING = Path(__file__).resolve().parent
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

from titanium_training.data.eval_packed import FEATURE_SCHEMA
from titanium_training.models.field_planes import (
    GOAL_INV_P0, GOAL_INV_P1,
    compact_route_vectors,
    rec_field,
)
from titanium_training.paths import ENGINE_BIN, REPO_ROOT, ACTIVE_TEACHER_DATASET
from titanium_training.data.teacher_value import iter_value_only_rows
from titanium_training.validation.engine_identity import load_expected_stamp

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FV_LEN    = 545
VAL_FRAC  = 0.05
CACHE_SEED = 42
BATCH_SIZE = 4096  # large batch amortizes per-subprocess LUT startup cost
PACKED_RECORD = struct.Struct("<I24s")

NET_MIRC = [(8 - i // 9) * 9 + i % 9 for i in range(81)]
NET_MIRS = [(7 - i // 8) * 8 + i % 8 for i in range(64)]
NET_BKT  = [(i // 9 // 3) * 3 + (i % 9) // 3 for i in range(81)]

# ---------------------------------------------------------------------------
# Engine call
# ---------------------------------------------------------------------------

def eval_packed_batch_raw(items: list[tuple[bytes, float, int]]) -> list[dict | None]:
    """
    Call titanium eval-packed-batch.  items = [(packed_state, target, side_to_move)].
    Returns list of result dicts (or None on failure).
    """
    payload = bytearray()
    for i, (ps, _, _) in enumerate(items):
        payload.extend(PACKED_RECORD.pack(i, ps))
    proc = subprocess.run(
        [str(ENGINE_BIN), "eval-packed-batch"],
        input=bytes(payload),
        capture_output=True,
        cwd=str(REPO_ROOT),
        timeout=300,
    )
    if proc.returncode != 0:
        return [None] * len(items)
    lines = [ln for ln in proc.stdout.decode(errors="replace").splitlines() if ln.strip()]
    results: list[dict | None] = [None] * len(items)
    for ln in lines:
        try:
            rec = json.loads(ln)
            idx = int(rec.get("row", -1))
            if 0 <= idx < len(items) and rec.get("ok"):
                results[idx] = rec
        except Exception:
            pass
    return results

# ---------------------------------------------------------------------------
# Record → feature vector
# ---------------------------------------------------------------------------

def record_to_fv(rec: dict, target: float) -> np.ndarray | None:
    """Convert one eval-packed-batch result to a flat float32 feature vector."""
    try:
        me  = int(rec["turn"])

        d_me  = float(rec["d0"] if me == 0 else rec["d1"])
        d_opp = float(rec["d1"] if me == 0 else rec["d0"])
        w_me  = float(rec["wl0"] if me == 0 else rec["wl1"])
        w_opp = float(rec["wl1"] if me == 0 else rec["wl0"])

        # ws[15]: opponent corridor width
        gi0 = rec_field(rec, GOAL_INV_P0)
        gi1 = rec_field(rec, GOAL_INV_P1)
        d0i = int(round(d_me))  if me == 0 else int(round(d_opp))
        d1i = int(round(d_opp)) if me == 0 else int(round(d_me))
        cw0 = sum(1 for v in gi0 if v == d0i)
        cw1 = sum(1 for v in gi1 if v == d1i)
        width_opp = float(cw1 if me == 0 else cw0)

        legal_wall_norm = float(rec["legal_wall_count"]) / 128.0
        if me == 0:
            cross_me  = float(rec.get("legal_path_cross_p0", 0)) / 128.0
            cross_opp = float(rec.get("legal_path_cross_p1", 0)) / 128.0
        else:
            cross_me  = float(rec.get("legal_path_cross_p1", 0)) / 128.0
            cross_opp = float(rec.get("legal_path_cross_p0", 0)) / 128.0

        hw = rec["hw"]; vw = rec["vw"]
        if me == 0:
            wall_hw = [float(hw[s]) for s in range(64)]
            wall_vw = [float(vw[s]) for s in range(64)]
            bucket   = NET_BKT[rec["pawn0"]]
            pawn_me  = rec["pawn0"]
            pawn_opp = rec["pawn1"]
        else:
            wall_hw = [float(hw[NET_MIRS[s]]) for s in range(64)]
            wall_vw = [float(vw[NET_MIRS[s]]) for s in range(64)]
            pawn_me  = NET_MIRC[rec["pawn1"]]
            pawn_opp = NET_MIRC[rec["pawn0"]]
            bucket   = NET_BKT[pawn_me]

        rm, ro, nm, no, rc = compact_route_vectors(rec, NET_MIRC)

        fv = np.empty(FV_LEN, dtype=np.float32)
        fv[0]         = target
        fv[1]         = d_me
        fv[2]         = d_opp
        fv[3]         = w_me
        fv[4]         = w_opp
        fv[5]         = legal_wall_norm
        fv[6]         = width_opp
        fv[7]         = cross_me
        fv[8]         = cross_opp
        fv[9:73]      = wall_hw
        fv[73:137]    = wall_vw
        fv[137:218]   = rm
        fv[218:299]   = ro
        fv[299:380]   = nm
        fv[380:461]   = no
        fv[461:542]   = rc
        fv[542]       = float(bucket)
        fv[543]       = float(pawn_me)
        fv[544]       = float(pawn_opp)
        return fv
    except Exception:
        return None

# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------

def make_fingerprint(engine_stamp: dict, n_total: int, n_train: int, n_val: int) -> dict:
    return {
        "fv_len":         FV_LEN,
        "schema":         FEATURE_SCHEMA,
        "engine_sha256":  engine_stamp.get("sha256", "unknown"),
        "manifest_hash":  "31a422f25a8c701ebfa72410f59fab9dff52c2717e30985a3f8e6929be007d02",
        "n_total":        n_total,
        "n_train":        n_train,
        "n_val":          n_val,
        "val_frac":       VAL_FRAC,
        "seed":           CACHE_SEED,
    }


def check_fingerprint(cache_dir: Path) -> tuple[bool, str]:
    meta_path = cache_dir / "meta.json"
    if not meta_path.exists():
        return False, "meta.json missing"
    try:
        stored = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:
        return False, f"meta.json unreadable: {e}"
    stamp = load_expected_stamp() or {}
    checks = {
        "fv_len":        (stored.get("fv_len"),        FV_LEN),
        "schema":        (stored.get("schema"),         FEATURE_SCHEMA),
        "engine_sha256": (stored.get("engine_sha256"),  stamp.get("sha256", "unknown")),
        "manifest_hash": (stored.get("manifest_hash"),  "31a422f25a8c701ebfa72410f59fab9dff52c2717e30985a3f8e6929be007d02"),
    }
    for k, (got, want) in checks.items():
        if got != want:
            return False, f"{k} mismatch: cache={got!r} current={want!r}"
    pos_path = cache_dir / "positions.bin"
    ti_path  = cache_dir / "train_indices.npy"
    vi_path  = cache_dir / "val_indices.npy"
    for p in (pos_path, ti_path, vi_path):
        if not p.exists():
            return False, f"{p.name} missing"
    return True, "ok"

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build(cache_dir: Path, dataset_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    stamp = load_expected_stamp() or {}

    print(f"Building feature cache -> {cache_dir}")
    print(f"  schema  : {FEATURE_SCHEMA}")
    print(f"  engine  : {stamp.get('sha256','?')[:16]}...")
    print(f"  dataset : {dataset_dir}")
    print(f"  fv_len  : {FV_LEN}")
    print()

    # Pass 1: scan all labels, deduplicate by position_key, average multiple labels.
    # The dataset has 2.28M label rows for 1.4M unique positions.  Averaging reduces
    # the cache to unique positions only while preserving the expected value signal.
    print("Pass 1: scanning Parquet, deduplicating by position_key...", flush=True)
    pos_data: dict[Any, list] = {}  # key -> [packed_state, side_to_move, [value_i16, ...]]
    n_labels = 0
    for r in iter_value_only_rows(dataset_dir, root=REPO_ROOT):
        if r.get("_missing_position"):
            continue
        packed = bytes(r["packed_state"])
        if len(packed) != 24:
            continue
        key = r["position_key"]
        if key not in pos_data:
            pos_data[key] = [packed, int(r["side_to_move"]), []]
        pos_data[key][2].append(int(r["value_i16"]))
        n_labels += 1
    rows = [
        (d[0], (sum(d[2]) / len(d[2]) / 100.0 + 1.0) / 2.0, d[1])
        for d in pos_data.values()
    ]
    del pos_data
    print(f"  {n_labels:,} labels -> {len(rows):,} unique positions (avg {n_labels/len(rows):.2f} labels/pos)", flush=True)

    # Pass 2: featurize in batches, write directly to memmap
    N_MAX = len(rows)
    pos_path = cache_dir / "positions.bin"
    print(f"\nPass 2: featurizing {N_MAX:,} positions in batches of {BATCH_SIZE}...", flush=True)

    mmap = np.memmap(pos_path, dtype="float32", mode="w+", shape=(N_MAX, FV_LEN))
    n_written  = 0
    n_failed   = 0
    t0 = time.perf_counter()

    for start in range(0, N_MAX, BATCH_SIZE):
        chunk    = rows[start : start + BATCH_SIZE]
        results  = eval_packed_batch_raw(chunk)

        for (packed, target_p0, side_to_move), rec in zip(chunk, results):
            if rec is None:
                n_failed += 1
                continue
            # Convert target to side-to-move perspective (matches __getitem__)
            me = int(rec.get("turn", side_to_move))
            target = target_p0 if me == 0 else (1.0 - target_p0)
            fv = record_to_fv(rec, target)
            if fv is None:
                n_failed += 1
                continue
            mmap[n_written] = fv
            n_written += 1

        if (start // BATCH_SIZE) % 100 == 0:
            elapsed = time.perf_counter() - t0
            done    = start + len(chunk)
            rate    = done / elapsed if elapsed > 0 else 0
            eta     = (N_MAX - done) / rate if rate > 0 else 0
            print(f"  {done:>8,}/{N_MAX:,}  written={n_written:,}  "
                  f"failed={n_failed:,}  {rate:.0f} pos/s  ETA {eta:.0f}s",
                  flush=True)

    mmap.flush()
    del mmap
    elapsed = time.perf_counter() - t0
    print(f"\n  Done: {n_written:,} written, {n_failed:,} failed in {elapsed:.1f}s", flush=True)
    coverage = 100.0 * n_written / N_MAX if N_MAX else 0
    print(f"  Coverage: {coverage:.2f}%", flush=True)

    # Pass 3: shuffle and split train/val
    print("\nPass 3: shuffle + train/val split...", flush=True)
    rng = np.random.default_rng(CACHE_SEED)
    perm = rng.permutation(n_written).astype(np.int32)
    n_val   = max(1, int(n_written * VAL_FRAC))
    n_train = n_written - n_val
    train_idx = perm[:n_train]
    val_idx   = perm[n_train:]
    np.save(cache_dir / "train_indices.npy", train_idx)
    np.save(cache_dir / "val_indices.npy",   val_idx)
    print(f"  train={n_train:,}  val={n_val:,}", flush=True)

    # Write meta
    fp = make_fingerprint(stamp, n_written, n_train, n_val)
    (cache_dir / "meta.json").write_text(json.dumps(fp, indent=2) + "\n", encoding="utf-8")
    print(f"\nCache ready: {cache_dir}", flush=True)
    print(json.dumps(fp, indent=2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default="training/data/feature_cache",
                    help="Cache output directory")
    ap.add_argument("--dataset",   default=str(ACTIVE_TEACHER_DATASET),
                    help="Teacher dataset directory (default: active)")
    ap.add_argument("--force",     action="store_true",
                    help="Rebuild even if cache fingerprint matches")
    args = ap.parse_args()

    cache_dir   = Path(args.cache_dir)
    dataset_dir = Path(args.dataset)

    if not args.force:
        ok, reason = check_fingerprint(cache_dir)
        if ok:
            print(f"Cache is up-to-date ({cache_dir}). Use --force to rebuild.")
            return 0
        print(f"Cache invalid: {reason}. Rebuilding...")

    build(cache_dir, dataset_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
