#!/usr/bin/env python3
"""
Import wallz.gg replays into a wallz feature cache.

Pipeline:
  1. Read training/data/aditional imports/replays.jsonl.gz
  2. Convert each game's moves to engine algebraic notation
  3. Feed all per-game position prefixes to `titanium eval-batch` (one line = one prefix)
  4. Convert each JSON response to a feature vector via record_to_fv (same logic as cache builder)
  5. Deduplicate and write wallz_positions.bin + wallz_meta.json in the same format as
     the main feature cache — ready to merge into positions.bin after training finishes.

Usage:
    python training/import_wallz.py [--dry-run] [--limit N] [--lines-per-call N]
    python training/import_wallz.py --merge   # merge wallz cache into main feature cache
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterator

import numpy as np

# ---------------------------------------------------------------------------
# Paths & imports
# ---------------------------------------------------------------------------

_TRAINING = Path(__file__).resolve().parent
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

from titanium_training.paths import ENGINE_BIN, REPO_ROOT
from build_feature_cache import (
    record_to_fv, FV_LEN, NET_MIRC, NET_MIRS, NET_BKT,
    make_fingerprint, check_fingerprint,
    CACHE_SEED, VAL_FRAC,
)

REPLAYS_GZ   = _TRAINING / "data" / "aditional imports" / "replays.jsonl.gz"
MAIN_CACHE   = _TRAINING / "data" / "feature_cache"
WALLZ_CACHE  = _TRAINING / "data" / "wallz_cache"

_COL_LETTER = "abcdefghi"


# ---------------------------------------------------------------------------
# Move conversion
# ---------------------------------------------------------------------------

def wallz_move_to_engine(move: dict) -> str | None:
    t = move.get("type")
    if t == "pawn":
        to = move.get("to", {})
        x, y = int(to["x"]), int(to["y"])
        if not (0 <= x <= 8 and 0 <= y <= 8):
            return None
        return f"{_COL_LETTER[x]}{y + 1}"
    elif t == "wall":
        w = move.get("wall", {})
        x, y = int(w["x"]), int(w["y"])
        o = w.get("o", "")
        if o not in ("h", "v"):
            return None
        if not (0 <= x <= 7 and 0 <= y <= 7):
            return None
        return f"{_COL_LETTER[x]}{y + 1}{o}"
    return None


def game_to_engine_moves(payload: dict) -> list[str] | None:
    result = []
    for entry in payload.get("moves", []):
        m = wallz_move_to_engine(entry.get("move", {}))
        if m is None:
            return None
        result.append(m)
    return result


# ---------------------------------------------------------------------------
# Read wallz games
# ---------------------------------------------------------------------------

def iter_wallz(path: Path, limit: int | None = None) -> Iterator[dict]:
    count = 0
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if limit and count >= limit:
                break
            try:
                obj = json.loads(line)
                p = obj.get("payload", {})
                if not p.get("winner") or not p.get("moves"):
                    continue
                yield p
                count += 1
            except Exception:
                continue


# ---------------------------------------------------------------------------
# Position dedup key: (pawn0, pawn1, wl0, wl1, turn) + wall hash
# ---------------------------------------------------------------------------

def position_key(rec: dict) -> bytes:
    pawn0 = rec.get("pawn0", -1)
    pawn1 = rec.get("pawn1", -1)
    wl0   = int(rec.get("wl0", 0))
    wl1   = int(rec.get("wl1", 0))
    turn  = int(rec.get("turn", 0))
    hw    = tuple(int(x) for x in rec.get("hw", []))
    vw    = tuple(int(x) for x in rec.get("vw", []))
    raw   = f"{pawn0},{pawn1},{wl0},{wl1},{turn},{hw},{vw}".encode()
    return hashlib.md5(raw).digest()


# ---------------------------------------------------------------------------
# Batch eval via eval-batch
# ---------------------------------------------------------------------------

def eval_batch_lines(lines: list[str]) -> list[dict | None]:
    """Send lines to titanium eval-batch (one move-sequence per line), return JSON dicts."""
    if not lines:
        return []
    stdin_text = "\n".join(lines) + "\n"
    try:
        proc = subprocess.run(
            [str(ENGINE_BIN), "eval-batch"],
            input=stdin_text.encode(),
            capture_output=True,
            cwd=str(REPO_ROOT),
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        return [None] * len(lines)
    if proc.returncode != 0:
        return [None] * len(lines)
    out_lines = [l for l in proc.stdout.decode(errors="replace").splitlines() if l.strip()]
    if len(out_lines) != len(lines):
        # mismatch — return None for all (can happen if engine crashes mid-batch)
        return [None] * len(lines)
    results = []
    for ln in out_lines:
        try:
            results.append(json.loads(ln))
        except Exception:
            results.append(None)
    return results


# ---------------------------------------------------------------------------
# Build wallz feature cache
# ---------------------------------------------------------------------------

def build_wallz_cache(
    limit: int | None = None,
    lines_per_call: int = 8192,
    workers: int = 4,
    dry_run: bool = False,
) -> int:
    from titanium_training.validation.engine_identity import load_expected_stamp
    stamp = load_expected_stamp() or {}

    print(f"Engine: {stamp.get('sha256', '?')[:16]}...")
    print(f"Input : {REPLAYS_GZ}")
    print(f"Output: {WALLZ_CACHE}")
    print()

    # 1. Read and convert games
    print("Reading wallz games...", flush=True)
    raw_games: list[dict] = list(iter_wallz(REPLAYS_GZ, limit))
    print(f"  {len(raw_games):,} games loaded")

    converted: list[tuple[list[str], int]] = []  # (moves, outcome_for_p0: ±100)
    n_bad = 0
    for p in raw_games:
        moves = game_to_engine_moves(p)
        if moves is None:
            n_bad += 1
            continue
        outcome_p0 = 100 if p["winner"] == "p1" else -100
        converted.append((moves, outcome_p0))
    print(f"  {len(converted):,} valid ({n_bad} bad)")

    if dry_run:
        for moves, outcome in converted[:3]:
            print(f"  {'P0' if outcome>0 else 'P1'} wins  {' '.join(moves[:6])}...")
        return 0

    # 2. Build list of (eval-batch line, outcome_p0, ply) for every prefix
    #    For a game with N moves, prefixes are: "", "e2", "e2 e8", ..., "e2 e8 ... last"
    #    The side-to-move at ply k is k % 2 (0 = P0, 1 = P1).
    print("\nBuilding prefix list...", flush=True)
    prefix_lines:   list[str]  = []
    prefix_outcomes: list[int] = []  # outcome_p0

    for moves, outcome_p0 in converted:
        for k in range(len(moves) + 1):
            prefix_lines.append(" ".join(moves[:k]))
            prefix_outcomes.append(outcome_p0)

    print(f"  {len(prefix_lines):,} total prefixes")

    # 3. Deduplicate prefix lines before sending to engine (exact string match)
    #    For duplicate strings, keep first occurrence and remember all outcomes (average).
    print("Pre-deduplicating prefixes...", flush=True)
    uniq_line_to_idx: dict[str, int] = {}
    uniq_lines: list[str] = []
    uniq_outcome_sums:  list[float] = []
    uniq_outcome_count: list[int]   = []
    for line, outcome_p0 in zip(prefix_lines, prefix_outcomes):
        if line not in uniq_line_to_idx:
            uniq_line_to_idx[line] = len(uniq_lines)
            uniq_lines.append(line)
            uniq_outcome_sums.append(float(outcome_p0))
            uniq_outcome_count.append(1)
        else:
            idx = uniq_line_to_idx[line]
            uniq_outcome_sums[idx] += float(outcome_p0)
            uniq_outcome_count[idx] += 1
    print(f"  {len(uniq_lines):,} unique prefixes (from {len(prefix_lines):,})")

    # 4. Parallel eval-batch calls
    print(f"\nRunning eval-batch: {workers} workers, {lines_per_call:,} lines/chunk...",
          flush=True)
    n_ok = 0; n_fail = 0
    eval_results: list[dict | None] = [None] * len(uniq_lines)

    # Build list of (start_idx, chunk) work items
    work = [
        (start, uniq_lines[start : start + lines_per_call])
        for start in range(0, len(uniq_lines), lines_per_call)
    ]
    n_chunks = len(work)
    n_done   = 0
    t0 = time.perf_counter()

    def _eval_chunk(item):
        start, chunk = item
        return start, eval_batch_lines(chunk)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_eval_chunk, w): w for w in work}
        for fut in as_completed(futures):
            start, recs = fut.result()
            for i, rec in enumerate(recs):
                eval_results[start + i] = rec
                if rec is not None:
                    n_ok += 1
                else:
                    n_fail += 1
            n_done += 1
            if n_done % max(1, n_chunks // 20) == 0 or n_done == n_chunks:
                elapsed = time.perf_counter() - t0
                rate    = (start + len(recs)) / elapsed if elapsed > 0 else 0
                pct     = 100 * n_done / n_chunks
                eta     = (len(uniq_lines) - n_ok - n_fail) / rate if rate > 0 else 0
                print(f"  chunk {n_done:>4}/{n_chunks}  {pct:.0f}%  "
                      f"ok={n_ok:,}  fail={n_fail}  {rate:.0f} pos/s  ETA {eta:.0f}s",
                      flush=True)

    elapsed = time.perf_counter() - t0
    print(f"  Done: {n_ok:,} ok / {n_fail} fail  in {elapsed:.1f}s  "
          f"({n_ok/elapsed:.0f} pos/s)", flush=True)

    # 5. Convert to feature vectors, deduplicate by position content
    print("\nConverting to feature vectors...", flush=True)
    seen_keys: set[bytes]  = set()
    fv_list:   list[np.ndarray] = []
    n_fv_fail = 0

    for i, rec in enumerate(eval_results):
        if rec is None:
            n_fv_fail += 1
            continue
        avg_outcome_p0 = uniq_outcome_sums[i] / uniq_outcome_count[i]
        # Convert to 0..1 win-prob (same convention as build_feature_cache.py):
        #   +100 -> 1.0 (P0 certain win), -100 -> 0.0 (P0 certain loss)
        target_p0 = (avg_outcome_p0 / 100.0 + 1.0) / 2.0
        stm = int(rec.get("turn", 0))
        # Flip to side-to-move perspective (same as cache builder line 291)
        target = target_p0 if stm == 0 else (1.0 - target_p0)
        pk = position_key(rec)
        if pk in seen_keys:
            continue
        seen_keys.add(pk)
        fv = record_to_fv(rec, target)
        if fv is None:
            n_fv_fail += 1
            continue
        fv_list.append(fv)

    print(f"  {len(fv_list):,} unique FVs  ({n_fv_fail} failures)")

    if not fv_list:
        print("No feature vectors — aborting.")
        return 1

    # 6. Write cache
    WALLZ_CACHE.mkdir(parents=True, exist_ok=True)
    n_total = len(fv_list)
    rng     = np.random.default_rng(CACHE_SEED)
    idx     = np.arange(n_total, dtype=np.int32)
    rng.shuffle(idx)
    n_val   = max(1, int(n_total * VAL_FRAC))
    n_train = n_total - n_val
    train_idx = idx[:n_train]
    val_idx   = idx[n_train:]

    pos_path = WALLZ_CACHE / "positions.bin"
    print(f"\nWriting {pos_path} ({n_total:,} × {FV_LEN} float32)...", flush=True)
    data = np.stack(fv_list, axis=0).astype(np.float32)
    mmap = np.memmap(pos_path, dtype="float32", mode="w+", shape=(n_total, FV_LEN))
    mmap[:] = data
    mmap.flush()

    np.save(WALLZ_CACHE / "train_indices.npy", train_idx)
    np.save(WALLZ_CACHE / "val_indices.npy",   val_idx)

    meta = make_fingerprint(stamp, n_total, n_train, n_val)
    meta["n_total"]  = n_total
    meta["n_train"]  = n_train
    meta["n_val"]    = n_val
    meta["source"]   = "wallz"
    meta["n_games"]  = len(converted)
    (WALLZ_CACHE / "meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(f"  train={n_train:,}  val={n_val:,}")
    print(f"  Written to {WALLZ_CACHE}")
    print()
    print("Next: python training/import_wallz.py --merge")
    return 0


# ---------------------------------------------------------------------------
# Merge wallz cache into main cache
# ---------------------------------------------------------------------------

def merge_into_main() -> int:
    """Append wallz positions to main feature cache and rebuild indices."""
    print(f"Merging {WALLZ_CACHE} -> {MAIN_CACHE}", flush=True)

    ok, reason = check_fingerprint(WALLZ_CACHE)
    if not ok:
        print(f"Wallz cache invalid: {reason}")
        return 1

    # Load main cache meta
    main_meta = json.loads((MAIN_CACHE / "meta.json").read_text(encoding="utf-8"))
    wallz_meta = json.loads((WALLZ_CACHE / "meta.json").read_text(encoding="utf-8"))

    n_main   = main_meta["n_total"]
    n_wallz  = wallz_meta["n_total"]
    n_merged = n_main + n_wallz

    print(f"  Main   : {n_main:,} positions")
    print(f"  Wallz  : {n_wallz:,} positions")
    print(f"  Merged : {n_merged:,} positions")

    # Load main cache
    main_pos  = np.memmap(MAIN_CACHE / "positions.bin", dtype="float32",
                           mode="r", shape=(n_main, FV_LEN))
    wallz_pos = np.memmap(WALLZ_CACHE / "positions.bin", dtype="float32",
                           mode="r", shape=(n_wallz, FV_LEN))

    # Write merged positions.bin (back up old one first)
    main_pos_path = MAIN_CACHE / "positions.bin"
    backup_path   = MAIN_CACHE / "positions_pre_wallz.bin"
    if not backup_path.exists():
        print(f"  Backing up -> {backup_path.name}")
        import shutil
        shutil.copy2(main_pos_path, backup_path)

    print(f"  Writing merged positions.bin...", flush=True)
    merged = np.memmap(main_pos_path, dtype="float32", mode="w+", shape=(n_merged, FV_LEN))
    merged[:n_main]  = main_pos[:]
    merged[n_main:]  = wallz_pos[:]
    merged.flush()

    # Rebuild shuffled indices
    rng = np.random.default_rng(CACHE_SEED)
    idx = np.arange(n_merged, dtype=np.int32)
    rng.shuffle(idx)
    n_val   = max(1, int(n_merged * VAL_FRAC))
    n_train = n_merged - n_val
    np.save(MAIN_CACHE / "train_indices.npy", idx[:n_train])
    np.save(MAIN_CACHE / "val_indices.npy",   idx[n_train:])

    # Update meta.json (invalidates fingerprint so trainer will refuse stale cache)
    from titanium_training.validation.engine_identity import load_expected_stamp
    stamp = load_expected_stamp() or {}
    meta  = make_fingerprint(stamp, n_merged, n_train, n_val)
    meta["n_total"] = n_merged
    meta["n_train"] = n_train
    meta["n_val"]   = n_val
    meta["sources"] = [f"teacher_dataset ({n_main:,})", f"wallz ({n_wallz:,})"]
    (MAIN_CACHE / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"  train={n_train:,}  val={n_val:,}")
    print("  Merge complete. Restart trainer with same --cache-dir.")
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run",        action="store_true")
    ap.add_argument("--limit",          type=int, default=None)
    ap.add_argument("--lines-per-call", type=int, default=8192,
                    help="Prefixes per eval-batch subprocess call (default 8192)")
    ap.add_argument("--workers",        type=int, default=4,
                    help="Parallel eval-batch workers (default 4; leave 1 core for trainer)")
    ap.add_argument("--merge",          action="store_true",
                    help="Merge completed wallz cache into main feature cache")
    args = ap.parse_args()

    if args.merge:
        return merge_into_main()
    return build_wallz_cache(
        limit=args.limit,
        lines_per_call=args.lines_per_call,
        workers=args.workers,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
