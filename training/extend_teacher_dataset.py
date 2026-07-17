#!/usr/bin/env python3
"""
Build a non-active experimental teacher dataset with new positions from:
  - zero.ink JSONL game files  (soft NN value labels — highest quality)
  - self-play games in games.db (outcome labels from titanium vs titanium)

Creates training/data/teacher_dataset_experimental_extended/ with:
  - Old positions + new positions  (dedup by packed_state)
  - Old labels + new labels
  - Same policy files (no new policies)
  - New manifest.json with updated hash

This script does not define the active training dataset. Active training uses
training/data/teacher_dataset_good via titanium_training.paths.

Usage:
  python training/extend_teacher_dataset.py           # zeroink-only experimental extension
  python training/extend_teacher_dataset.py --dry-run # count new positions, don't write
  python training/extend_teacher_dataset.py --zeroink-only  # skip self-play
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO     = Path(__file__).resolve().parent.parent
_TRAINING = _REPO / "training"
sys.path.insert(0, str(_TRAINING))

import pyarrow as pa
import pyarrow.parquet as pq

from titanium_training.store.state import PositionState

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

OLD_DIR  = _TRAINING / "data" / "teacher_dataset"
NEW_DIR  = _TRAINING / "data" / "teacher_dataset_experimental_extended"
ZI_DIR   = _TRAINING / "data" / "zeroink_games"
GAMES_DB = _TRAINING / "data" / "canonical" / "games.db"
LABELS_DB = _TRAINING / "data" / "canonical" / "labels.db"
PATHS_PY = _TRAINING / "titanium_training" / "paths.py"

# ---------------------------------------------------------------------------
# Key derivation (must match teacher_dataset/build.py)
# ---------------------------------------------------------------------------

def position_key(canonical_hash: bytes, packed_state: bytes) -> bytes:
    return hashlib.blake2b(canonical_hash + packed_state, digest_size=16).digest()


def make_position_key_from_state(ps: PositionState) -> tuple[bytes, bytes, bytes]:
    """Returns (packed_state_bytes, canonical_hash_bytes, position_key_bytes)."""
    packed  = ps.packed_state()
    canon   = ps.canonical_hash()
    pk      = position_key(canon, packed)
    return packed, canon, pk

# ---------------------------------------------------------------------------
# Value conversion
# ---------------------------------------------------------------------------

def winprob_to_value_i16(win_prob: float) -> int:
    """zero.ink win-prob (0..1, current-player) → value_i16 (-100..100)."""
    stm_val = win_prob * 2.0 - 1.0          # -1..+1, positive = good for current player
    return max(-100, min(100, int(round(stm_val * 100))))


def float_stm_to_value_i16(value_stm: float) -> int:
    """labels.db value_stm (-1..+1, current-player) → value_i16."""
    return max(-100, min(100, int(round(value_stm * 100))))

# ---------------------------------------------------------------------------
# Zero.ink wall bitmask conversion
# Zero.ink {x, y} slot: slot = y * 8 + x  (matches notation_to_wall_slot)
# ---------------------------------------------------------------------------

def walls_to_mask(wall_list: list[dict]) -> int:
    mask = 0
    for w in wall_list:
        slot = int(w["y"]) * 8 + int(w["x"])
        mask |= (1 << slot)
    return mask


def zeroink_state_to_position(state: dict) -> PositionState:
    return PositionState(
        player0_cell    = int(state["player0Cell"]),
        player1_cell    = int(state["player1Cell"]),
        player0_walls   = int(state["player0Walls"]),
        player1_walls   = int(state["player1Walls"]),
        horizontal_walls = walls_to_mask(state.get("horizontalWalls", [])),
        vertical_walls   = walls_to_mask(state.get("verticalWalls", [])),
        side_to_move    = int(state["currentPlayer"]),
    )

# ---------------------------------------------------------------------------
# New position sources
# ---------------------------------------------------------------------------

def iter_zeroink_positions() -> dict:
    """Yield {packed, canon, pk, side_to_move, value_i16, source_cohort} per position."""
    for jsonl in sorted(ZI_DIR.glob("*.jsonl")):
        for line in jsonl.open(encoding="utf-8"):
            try:
                rec = json.loads(line)
            except Exception:
                continue
            state_raw = rec.get("state")
            value     = rec.get("value")
            if state_raw is None or value is None:
                continue
            try:
                ps = zeroink_state_to_position(state_raw)
                ps.validate(require_paths=False)   # skip path check for speed
            except Exception:
                continue
            packed, canon, pk = make_position_key_from_state(ps)
            yield {
                "packed": packed,
                "canon":  canon,
                "pk":     pk,
                "side_to_move": int(state_raw["currentPlayer"]),
                "value_i16":    winprob_to_value_i16(float(value)),
                "source_cohort": "quoridor-zero.ink",
            }


def iter_selfplay_positions() -> dict:
    """
    Yield positions from self-play games in games.db/labels.db.
    Reconstructs board state by replaying move sequences.
    Only uses 'selfplay_train' source (clean self-play, not verify games).
    """
    if not GAMES_DB.exists() or not LABELS_DB.exists():
        return

    gcon = sqlite3.connect(str(GAMES_DB), timeout=30)
    lcon = sqlite3.connect(str(LABELS_DB), timeout=30)

    # Load value labels keyed by pos_key
    val_map: dict[str, float] = {}
    for pos_key_hex, value_stm in lcon.execute(
        "SELECT pos_key, value_stm FROM labels WHERE source IN ('selfplay_train','selfplay_verify')"
    ):
        # Use the latest value if duplicated
        val_map[pos_key_hex] = float(value_stm)

    if not val_map:
        gcon.close(); lcon.close()
        return

    # Load all self-play games (not verify games)
    games = gcon.execute(
        "SELECT game_id FROM games WHERE source='selfplay_train'"
    ).fetchall()

    for (game_id,) in games:
        moves_rows = gcon.execute(
            "SELECT move_num, pos_key, move_alg FROM game_moves WHERE game_id=? ORDER BY move_num",
            (game_id,)
        ).fetchall()
        if not moves_rows:
            continue

        state = PositionState.initial()
        for move_num, pos_key_hex, move_alg in moves_rows:
            if pos_key_hex not in val_map:
                # advance state anyway
                try:
                    state = state.with_move(move_alg)
                except Exception:
                    break
                continue
            try:
                packed, canon, pk = make_position_key_from_state(state)
            except Exception:
                try:
                    state = state.with_move(move_alg)
                except Exception:
                    break
                continue
            yield {
                "packed": packed,
                "canon":  canon,
                "pk":     pk,
                "side_to_move": state.side_to_move,
                "value_i16":    float_stm_to_value_i16(val_map[pos_key_hex]),
                "source_cohort": "titanium-selfplay",
            }
            try:
                state = state.with_move(move_alg)
            except Exception:
                break

    gcon.close(); lcon.close()


# ---------------------------------------------------------------------------
# Build extended dataset
# ---------------------------------------------------------------------------

def build_extended(*, dry_run: bool = False, zeroink_only: bool = True) -> None:
    # ------ Load existing dataset ------
    print("Reading existing teacher_dataset ...", flush=True)
    old_pos_tbl = pq.read_table(OLD_DIR / "positions" / "part-00000.parquet")
    old_lbl_tbl = pq.read_table(OLD_DIR / "labels"    / "part-00000.parquet")

    # Build set of existing packed_states for dedup
    existing_packed: set[bytes] = set()
    for i in range(old_pos_tbl.num_rows):
        existing_packed.add(bytes(old_pos_tbl.column("packed_state")[i].as_py()))
    print(f"  Existing: {old_pos_tbl.num_rows:,} positions, {old_lbl_tbl.num_rows:,} labels")

    # ------ Collect new data ------
    new_pos_rows: list[dict] = []   # position table rows
    new_lbl_rows: list[dict] = []   # label table rows
    seen_packed:  set[bytes] = set()

    def add_if_new(rec: dict) -> bool:
        packed = rec["packed"]
        if packed in existing_packed or packed in seen_packed:
            return False
        seen_packed.add(packed)
        new_pos_rows.append({
            "position_key":     rec["pk"],
            "canonical_hash":   rec["canon"],
            "packed_state":     packed,
            "side_to_move":     rec["side_to_move"],
            "source_flags":     0,
            "total_observations": 1,
        })
        # Use a stable 8-byte label_set_id from pk
        label_set_id = rec["pk"][:8]
        new_lbl_rows.append({
            "position_key":     rec["pk"],
            "label_set_id":     label_set_id,
            "target_kind":      4,           # same as titanium-native value-only labels
            "value_i16":        rec["value_i16"],
            "best_move_u8":     None,
            "policy_record_id": None,
            "has_policy":       False,
            "observation_count": 1,
            "source_cohort":    rec["source_cohort"],
        })
        return True

    print("Collecting zero.ink positions ...", flush=True)
    zi_count = 0
    for rec in iter_zeroink_positions():
        if add_if_new(rec):
            zi_count += 1
    print(f"  New zero.ink positions: {zi_count:,}")

    sp_count = 0
    if not zeroink_only:
        print("WARNING: including Titanium self-play; do not use this output as active NNUE training data.")
        print("Collecting self-play positions ...", flush=True)
        for rec in iter_selfplay_positions():
            if add_if_new(rec):
                sp_count += 1
        print(f"  New self-play positions: {sp_count:,}")

    total_new = len(new_pos_rows)
    print(f"Total new unique positions: {total_new:,}")

    if dry_run:
        print("(dry-run — not writing)")
        return

    if total_new == 0:
        print("Nothing new to add.")
        return

    # ------ Build new Parquet tables ------
    print("Building new Parquet tables ...", flush=True)

    # Positions schema — infer from existing + append new
    new_pos_arrays = {
        "position_key":      pa.array([r["position_key"]     for r in new_pos_rows], type=old_pos_tbl.schema.field("position_key").type),
        "canonical_hash":    pa.array([r["canonical_hash"]   for r in new_pos_rows], type=old_pos_tbl.schema.field("canonical_hash").type),
        "packed_state":      pa.array([r["packed_state"]     for r in new_pos_rows], type=old_pos_tbl.schema.field("packed_state").type),
        "side_to_move":      pa.array([r["side_to_move"]     for r in new_pos_rows], type=old_pos_tbl.schema.field("side_to_move").type),
        "source_flags":      pa.array([r["source_flags"]     for r in new_pos_rows], type=old_pos_tbl.schema.field("source_flags").type),
        "total_observations":pa.array([r["total_observations"] for r in new_pos_rows], type=old_pos_tbl.schema.field("total_observations").type),
    }
    new_pos_tbl = pa.table(new_pos_arrays, schema=old_pos_tbl.schema)
    combined_pos = pa.concat_tables([old_pos_tbl, new_pos_tbl])

    # Labels schema
    new_lbl_arrays = {
        "position_key":     pa.array([r["position_key"]     for r in new_lbl_rows], type=old_lbl_tbl.schema.field("position_key").type),
        "label_set_id":     pa.array([r["label_set_id"]     for r in new_lbl_rows], type=old_lbl_tbl.schema.field("label_set_id").type),
        "target_kind":      pa.array([r["target_kind"]      for r in new_lbl_rows], type=old_lbl_tbl.schema.field("target_kind").type),
        "value_i16":        pa.array([r["value_i16"]        for r in new_lbl_rows], type=old_lbl_tbl.schema.field("value_i16").type),
        "best_move_u8":     pa.array([r["best_move_u8"]     for r in new_lbl_rows], type=old_lbl_tbl.schema.field("best_move_u8").type),
        "policy_record_id": pa.array([r["policy_record_id"] for r in new_lbl_rows], type=old_lbl_tbl.schema.field("policy_record_id").type),
        "has_policy":       pa.array([r["has_policy"]       for r in new_lbl_rows], type=old_lbl_tbl.schema.field("has_policy").type),
        "observation_count":pa.array([r["observation_count"] for r in new_lbl_rows], type=old_lbl_tbl.schema.field("observation_count").type),
        "source_cohort":    pa.array([r["source_cohort"]    for r in new_lbl_rows], type=old_lbl_tbl.schema.field("source_cohort").type),
    }
    new_lbl_tbl = pa.table(new_lbl_arrays, schema=old_lbl_tbl.schema)
    combined_lbl = pa.concat_tables([old_lbl_tbl, new_lbl_tbl])

    # ------ Write output ------
    print(f"Writing to {NEW_DIR} ...", flush=True)
    NEW_DIR.mkdir(parents=True, exist_ok=True)
    (NEW_DIR / "positions").mkdir(exist_ok=True)
    (NEW_DIR / "labels").mkdir(exist_ok=True)

    pq.write_table(combined_pos, NEW_DIR / "positions" / "part-00000.parquet",
                   compression="zstd")
    pq.write_table(combined_lbl, NEW_DIR / "labels"    / "part-00000.parquet",
                   compression="zstd")

    # Copy observations (unchanged) and policies (no new policy data)
    import shutil
    shutil.copytree(OLD_DIR / "observations", NEW_DIR / "observations", dirs_exist_ok=True)
    shutil.copytree(OLD_DIR / "policies",     NEW_DIR / "policies",     dirs_exist_ok=True)

    # ------ Build new manifest ------
    print("Computing manifest ...", flush=True)
    old_manifest = json.loads((OLD_DIR / "manifest.json").read_text(encoding="utf-8"))

    def file_bytes(path: Path) -> int:
        return path.stat().st_size

    new_manifest = {
        **old_manifest,
        "created_at":     datetime.now(timezone.utc).isoformat(),
        "source_sqlite":  str(GAMES_DB),
        "teacher_dataset_status": "extended",
        "immutable":      False,
        "promotion_allowed": False,
        "parent_dataset": str(OLD_DIR),
        "extension": {
            "new_zeroink_positions": zi_count,
            "new_selfplay_positions": sp_count,
            "total_new_positions": total_new,
        },
        "counts": {
            "positions":      combined_pos.num_rows,
            "labels":         combined_lbl.num_rows,
            "observations":   old_manifest["counts"]["observations"],
            "unique_policies":old_manifest["counts"]["unique_policies"],
            "has_policy_labels": old_manifest["counts"]["has_policy_labels"],
            "policy_quarantined": 0,
        },
        "bytes": {
            "positions":  file_bytes(NEW_DIR / "positions" / "part-00000.parquet"),
            "labels":     file_bytes(NEW_DIR / "labels"    / "part-00000.parquet"),
            "observations": old_manifest["bytes"]["observations"],
            "policy_bin": old_manifest["bytes"]["policy_bin"],
            "policy_idx": old_manifest["bytes"]["policy_idx"],
        },
        "parts": {
            "positions":   ["training/data/teacher_dataset_extended/positions/part-00000.parquet"],
            "labels":      ["training/data/teacher_dataset_extended/labels/part-00000.parquet"],
            "observations":["training/data/teacher_dataset_extended/observations/part-00000.parquet"],
            "policies":    [
                "training/data/teacher_dataset_extended/policies/policy-00000.bin",
                "training/data/teacher_dataset_extended/policies/policy-00000.idx",
            ],
        },
    }
    # Remove old promotion gates (they don't apply to the extended dataset)
    new_manifest.pop("promotion_gates", None)
    new_manifest.pop("parity_audit", None)
    new_manifest.pop("gate_evidence_bundle", None)

    # Compute new manifest hash (same algorithm as promotion_gates.compute_manifest_hash)
    payload = {k: v for k, v in new_manifest.items() if k != "manifest_hash"}
    new_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    new_manifest["manifest_hash"] = new_hash

    (NEW_DIR / "manifest.json").write_text(
        json.dumps(new_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"New manifest hash: {new_hash}")

    print()
    print("=" * 64)
    print(f"Extended dataset ready at: {NEW_DIR}")
    print(f"  Positions: {old_pos_tbl.num_rows:,} -> {combined_pos.num_rows:,} (+{total_new:,})")
    print(f"  Labels:    {old_lbl_tbl.num_rows:,} -> {combined_lbl.num_rows:,} (+{total_new:,})")
    print()
    print("This output is experimental and is not selected by training automatically.")
    print("Active training data remains training/data/teacher_dataset_good.")
    print("=" * 64)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run",      action="store_true",
                    help="Count new positions without writing anything")
    ap.add_argument("--include-titanium-selfplay", action="store_true",
                    help="Also add Titanium self-play. Do not use this for active NNUE value training.")
    args = ap.parse_args()
    build_extended(dry_run=args.dry_run, zeroink_only=not args.include_titanium_selfplay)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
