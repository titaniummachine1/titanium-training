#!/usr/bin/env python3
"""Fail-closed integrity + parity audit for the versioned 952-feature cache.

Blocks training on any failure. Covers:
  - fv_len exactly 952
  - schema exactly halfpw-sparse-route5-catv5-normalized5-ws20-v1
  - no 628-feature contamination / size mismatch
  - packed sidecar + stm + positions + index metadata agreement
  - train/val disjoint and lineage-covering
  - packed-derived phase counts nonzero and not all-midgame
  - bit-for-bit fresh engine parity on a deterministic sample
  - P2 180-degree orientation on sampled side-to-move=1 rows
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_TRAINING = Path(__file__).resolve().parent
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

from build_feature_cache import FV_LEN, eval_packed_batch_raw, record_to_fv
from label_weights import game_phase_from_packed
from titanium_training.data.eval_packed import FEATURE_SCHEMA

EXPECTED_SCHEMA = "halfpw-sparse-route5-catv5-normalized5-ws20-v1"
EXPECTED_ENGINE_SHA = "dceb8f9de28215747c66491cb71b77ae0299e935b28fdf3be3763eb127ce0ea0"
LEGACY_FV_LEN = 628
MIRC = [(8 - i // 9) * 9 + (8 - i % 9) for i in range(81)]


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}")
    return 1


def audit_integrity(cache_dir: Path) -> tuple[dict, int]:
    meta_path = cache_dir / "meta.json"
    if not meta_path.is_file():
        return {}, _fail("meta.json missing — rebuild incomplete")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    n = int(meta.get("n_total", -1))
    n_train = int(meta.get("n_train", -1))
    n_val = int(meta.get("n_val", -1))

    if meta.get("fv_len") != FV_LEN:
        return meta, _fail(f"fv_len {meta.get('fv_len')} != {FV_LEN}")
    if meta.get("schema") != EXPECTED_SCHEMA or FEATURE_SCHEMA != EXPECTED_SCHEMA:
        return meta, _fail(
            f"schema {meta.get('schema')!r} / FEATURE_SCHEMA={FEATURE_SCHEMA!r} "
            f"!= {EXPECTED_SCHEMA!r}"
        )
    if meta.get("engine_sha256") != EXPECTED_ENGINE_SHA:
        return meta, _fail(
            f"engine_sha256 {meta.get('engine_sha256')!r} != {EXPECTED_ENGINE_SHA!r}"
        )
    if n <= 0 or n_train <= 0 or n_val <= 0:
        return meta, _fail(f"bad counts n_total={n} n_train={n_train} n_val={n_val}")
    if n_train + n_val != n:
        return meta, _fail(f"n_train+n_val={n_train + n_val} != n_total={n}")

    required = {
        "positions.bin": n * FV_LEN * 4,
        "row_packed_states.bin": n * 24,
        "row_side_to_move.npy": None,
        "train_indices.npy": None,
        "val_indices.npy": None,
    }
    for name, want_bytes in required.items():
        path = cache_dir / name
        if not path.is_file():
            return meta, _fail(f"missing {name}")
        if want_bytes is not None and path.stat().st_size != want_bytes:
            # Detect legacy 628-width contamination explicitly.
            legacy = n * LEGACY_FV_LEN * 4
            extra = f" (matches legacy 628-width size {legacy})" if path.stat().st_size == legacy else ""
            return meta, _fail(f"{name} size {path.stat().st_size} != {want_bytes}{extra}")

    # Refuse any co-located stale 628 meta markers.
    stale = cache_dir / "STALE_LABEL_PERSPECTIVE.json"
    if stale.is_file():
        return meta, _fail("STALE_LABEL_PERSPECTIVE.json present — refuse mixed cache dir")

    train = np.load(cache_dir / "train_indices.npy")
    val = np.load(cache_dir / "val_indices.npy")
    stm = np.load(cache_dir / "row_side_to_move.npy")
    if train.dtype.kind not in "iu" or val.dtype.kind not in "iu":
        return meta, _fail("train/val indices must be integer")
    if train.shape != (n_train,) or val.shape != (n_val,):
        return meta, _fail(
            f"index shapes train={train.shape}/{n_train} val={val.shape}/{n_val}"
        )
    if stm.shape != (n,):
        return meta, _fail(f"stm shape {stm.shape} != ({n},)")

    train_set = set(int(x) for x in train.tolist())
    val_set = set(int(x) for x in val.tolist())
    if len(train_set) != n_train:
        return meta, _fail(f"train indices have duplicates: unique={len(train_set)}")
    if len(val_set) != n_val:
        return meta, _fail(f"val indices have duplicates: unique={len(val_set)}")
    overlap = train_set & val_set
    if overlap:
        return meta, _fail(f"train/val overlap ({len(overlap)} rows) — lineage unsafe")
    union = train_set | val_set
    if union != set(range(n)):
        missing = n - len(union)
        return meta, _fail(f"train∪val does not cover [0,n): missing_or_oob≈{missing}")

    # Packed-derived phase coverage over a large deterministic slice.
    packed = (cache_dir / "row_packed_states.bin").read_bytes()
    phase_sample = min(n, 50_000)
    rng = np.random.default_rng(7)
    phase_idxs = rng.choice(n, size=phase_sample, replace=False)
    phases = Counter(
        game_phase_from_packed(packed[i * 24 : (i + 1) * 24]) for i in phase_idxs
    )
    if phases.get("midgame", 0) == phase_sample:
        return meta, _fail("all sampled phases are midgame — packed phase hardcode suspected")
    for name in ("opening", "midgame", "endgame"):
        if phases.get(name, 0) <= 0:
            return meta, _fail(f"phase {name} count is zero in {phase_sample}-row sample")

    meta["_phase_sample"] = dict(phases)
    meta["_phase_sample_n"] = phase_sample
    print(
        "OK integrity: "
        f"n={n:,} train={n_train:,} val={n_val:,} "
        f"schema={EXPECTED_SCHEMA} fv_len={FV_LEN} "
        f"phases({phase_sample})={dict(phases)}"
    )
    return meta, 0


def audit_parity(cache_dir: Path, meta: dict, *, sample: int, seed: int) -> int:
    n = int(meta["n_total"])
    mmap = np.memmap(
        cache_dir / "positions.bin", dtype="float32", mode="r", shape=(n, FV_LEN)
    )
    stm = np.load(cache_dir / "row_side_to_move.npy")
    packed_blob = (cache_dir / "row_packed_states.bin").read_bytes()

    rng = np.random.default_rng(seed)
    idxs = np.sort(rng.choice(n, size=min(sample, n), replace=False))
    items = [
        (packed_blob[i * 24 : (i + 1) * 24], float(mmap[i, 0]), int(stm[i]))
        for i in idxs
    ]
    results = eval_packed_batch_raw(items)
    mismatches = 0
    checked = 0
    p2_checked = 0
    for i, (_packed, _tgt, stm_i), rec in zip(idxs, items, results):
        if rec is None:
            return _fail(f"engine returned None for row {i}")
        fv = record_to_fv(rec, float(mmap[i, 0]))
        if fv is None:
            return _fail(f"record_to_fv failed for row {i}")
        if fv.shape != (FV_LEN,):
            return _fail(f"fresh fv shape {fv.shape} != ({FV_LEN},) at row {i}")
        cached = np.asarray(mmap[i], dtype=np.float32)
        if not np.isfinite(cached).all():
            return _fail(f"non-finite cached features at row {i}")
        if not np.array_equal(cached, fv):
            diff = np.where(cached != fv)[0]
            print(
                f"FAIL: row {i} mismatch at {len(diff)} offsets; "
                f"first={int(diff[0])} cached={cached[diff[0]]} fresh={fv[diff[0]]}"
            )
            mismatches += 1
            if mismatches >= 5:
                break
            continue
        checked += 1
        me = int(rec.get("turn", stm_i))
        if me == 1:
            p2_checked += 1
            expected_me = MIRC[int(rec["pawn1"])]
            if int(round(float(cached[950]))) != expected_me:
                return _fail(
                    f"P2 orientation row {i}: cached pawn_me={cached[950]} "
                    f"expected MIRC[pawn1]={expected_me}"
                )

    if mismatches:
        return _fail(f"{mismatches} mismatched rows in parity sample")
    print(
        f"OK parity: {checked} rows bit-identical; P2 checks={p2_checked}; "
        f"sample={len(idxs)}"
    )
    return 0


def audit(cache_dir: Path, *, sample: int, seed: int) -> int:
    meta, rc = audit_integrity(cache_dir)
    if rc != 0:
        print("BLOCK: integrity failed — do not train")
        return rc
    rc = audit_parity(cache_dir, meta, sample=sample, seed=seed)
    if rc != 0:
        print("BLOCK: parity failed — do not train")
        return rc
    report = {
        "ok": True,
        "cache_dir": str(cache_dir),
        "schema": EXPECTED_SCHEMA,
        "fv_len": FV_LEN,
        "n_total": meta["n_total"],
        "n_train": meta["n_train"],
        "n_val": meta["n_val"],
        "engine_sha256": meta.get("engine_sha256"),
        "phase_sample": meta.get("_phase_sample"),
        "parity_sample": sample,
    }
    print(json.dumps(report, indent=2))
    print("PASS: cache integrity + parity — training may proceed on this cache only")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--sample", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    return audit(Path(args.cache_dir), sample=args.sample, seed=args.seed)


if __name__ == "__main__":
    raise SystemExit(main())
