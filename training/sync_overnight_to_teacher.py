#!/usr/bin/env python3
"""Archive-only overnight self-play sync helper.

Collapsed Titanium self-play is not active NNUE value-training data. Do not use
this helper to append into `teacher_dataset_good`.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

_TRAINING = Path(__file__).resolve().parent
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

import pyarrow as pa
import pyarrow.parquet as pq

from extend_teacher_dataset import float_stm_to_value_i16, make_position_key_from_state
from pool_lock import TeacherSyncLock
from titanium_training.paths import ACTIVE_TEACHER_DATASET, DATA_DIR
from titanium_training.store.state import PositionState

from db_import import GAMES_DB_PATH, LABELS_DB_PATH

OVERNIGHT_SOURCES = (
    "overnight_selfplay",
    "overnight_mixed",
    "pool_generation_selfplay",
    "pool_generation_mixed",
)
STATE_PATH = _TRAINING / "data" / "overnight_logs" / "teacher_sync_state.json"
EXPERIMENTAL_DATASET = DATA_DIR / "teacher_dataset_experimental_extended"
POOL_DATASET = DATA_DIR / "teacher_dataset_pool"


def active_teacher_dir() -> Path:
    """Return the active dataset, refusing to auto-select overnight self-play."""
    return ACTIVE_TEACHER_DATASET


def pool_teacher_dir() -> Path:
    """Parquet append target for overnight / oracle self-play (never teacher_dataset_good)."""
    override = os.environ.get("TITANIUM_POOL_TEACHER_DATASET")
    path = Path(override) if override else POOL_DATASET
    _bootstrap_teacher_if_missing(path)
    return path


def load_synced_ids() -> set[str]:
    if not STATE_PATH.is_file():
        return set()
    try:
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        return set(data.get("synced_game_ids", []))
    except Exception:
        return set()


def save_synced_ids(ids: set[str]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(
        json.dumps(
            {
                "synced_game_ids": sorted(ids),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def iter_overnight_games() -> list[tuple[str, list[str], int]]:
    if not GAMES_DB_PATH.is_file():
        return []
    con = sqlite3.connect(str(GAMES_DB_PATH), timeout=30)
    placeholders = ",".join("?" * len(OVERNIGHT_SOURCES))
    rows = con.execute(
        f"SELECT game_id, outcome_p0 FROM games WHERE source IN ({placeholders}) ORDER BY imported_at",
        OVERNIGHT_SOURCES,
    ).fetchall()
    out: list[tuple[str, list[str], int]] = []
    for game_id, outcome_p0 in rows:
        moves = [
            r[0]
            for r in con.execute(
                "SELECT move_alg FROM game_moves WHERE game_id=? ORDER BY move_num",
                (game_id,),
            )
        ]
        if moves:
            out.append((game_id, moves, int(outcome_p0)))
    con.close()
    return out


def positions_from_game(moves: list[str], outcome_p0: int) -> list[dict]:
    rows: list[dict] = []
    state = PositionState.initial()
    for move_alg in moves:
        try:
            packed, canon, pk = make_position_key_from_state(state)
        except Exception:
            try:
                state = state.with_move(move_alg)
            except Exception:
                break
            continue
        stm = state.side_to_move
        outcome_stm = float(outcome_p0) if stm == 0 else float(-outcome_p0)
        rows.append(
            {
                "packed": packed,
                "canon": canon,
                "pk": pk,
                "side_to_move": stm,
                "value_i16": float_stm_to_value_i16(outcome_stm),
            }
        )
        try:
            state = state.with_move(move_alg)
        except Exception:
            break
    return rows


_POS_SCHEMA = pa.schema([
    pa.field("position_key", pa.binary()),
    pa.field("canonical_hash", pa.binary()),
    pa.field("packed_state", pa.binary()),
    pa.field("side_to_move", pa.int32()),
    pa.field("source_flags", pa.int32()),
    pa.field("total_observations", pa.int32()),
])

_LBL_SCHEMA = pa.schema([
    pa.field("position_key", pa.binary()),
    pa.field("label_set_id", pa.binary()),
    pa.field("target_kind", pa.int32()),
    pa.field("value_i16", pa.int16()),
    pa.field("best_move_u8", pa.uint8()),
    pa.field("policy_record_id", pa.string()),
    pa.field("has_policy", pa.bool_()),
    pa.field("observation_count", pa.int32()),
    pa.field("source_cohort", pa.string()),
])


def _bootstrap_teacher_if_missing(dataset_dir: Path) -> None:
    """Create empty schema-correct parquet stubs if the teacher dataset is absent.

    This allows the pool to recover from accidental deletion or corruption
    without losing the ability to append new games.
    """
    pos_path = dataset_dir / "positions" / "part-00000.parquet"
    lbl_path = dataset_dir / "labels" / "part-00000.parquet"
    if pos_path.is_file() and lbl_path.is_file():
        return
    pos_path.parent.mkdir(parents=True, exist_ok=True)
    lbl_path.parent.mkdir(parents=True, exist_ok=True)
    if not pos_path.is_file():
        empty_pos = pa.table({f.name: pa.array([], type=f.type) for f in _POS_SCHEMA}, schema=_POS_SCHEMA)
        pq.write_table(empty_pos, str(pos_path), compression="zstd")
    if not lbl_path.is_file():
        empty_lbl = pa.table({f.name: pa.array([], type=f.type) for f in _LBL_SCHEMA}, schema=_LBL_SCHEMA)
        pq.write_table(empty_lbl, str(lbl_path), compression="zstd")


def append_to_teacher(new_rows: list[dict], dataset_dir: Path) -> int:
    if dataset_dir.name == "teacher_dataset_good":
        raise RuntimeError("refusing to append overnight/Titanium self-play to teacher_dataset_good")
    pos_path = dataset_dir / "positions" / "part-00000.parquet"
    lbl_path = dataset_dir / "labels" / "part-00000.parquet"
    _bootstrap_teacher_if_missing(dataset_dir)

    old_pos = pq.read_table(pos_path)
    old_lbl = pq.read_table(lbl_path)
    existing: set[bytes] = {bytes(old_pos.column("packed_state")[i].as_py()) for i in range(old_pos.num_rows)}

    pos_add: list[dict] = []
    lbl_add: list[dict] = []
    seen: set[bytes] = set()
    for rec in new_rows:
        packed = rec["packed"]
        if packed in existing or packed in seen:
            continue
        seen.add(packed)
        pos_add.append(
            {
                "position_key": rec["pk"],
                "canonical_hash": rec["canon"],
                "packed_state": packed,
                "side_to_move": rec["side_to_move"],
                "source_flags": 0,
                "total_observations": 1,
            }
        )
        lbl_add.append(
            {
                "position_key": rec["pk"],
                "label_set_id": rec["pk"][:8],
                "target_kind": 4,
                "value_i16": rec["value_i16"],
                "best_move_u8": None,
                "policy_record_id": None,
                "has_policy": False,
                "observation_count": 1,
                "source_cohort": "titanium-overnight",
            }
        )

    if not pos_add:
        return 0

    new_pos_tbl = pa.table(
        {
            "position_key": pa.array([r["position_key"] for r in pos_add], type=old_pos.schema.field("position_key").type),
            "canonical_hash": pa.array([r["canonical_hash"] for r in pos_add], type=old_pos.schema.field("canonical_hash").type),
            "packed_state": pa.array([r["packed_state"] for r in pos_add], type=old_pos.schema.field("packed_state").type),
            "side_to_move": pa.array([r["side_to_move"] for r in pos_add], type=old_pos.schema.field("side_to_move").type),
            "source_flags": pa.array([r["source_flags"] for r in pos_add], type=old_pos.schema.field("source_flags").type),
            "total_observations": pa.array([r["total_observations"] for r in pos_add], type=old_pos.schema.field("total_observations").type),
        },
        schema=old_pos.schema,
    )
    new_lbl_tbl = pa.table(
        {
            "position_key": pa.array([r["position_key"] for r in lbl_add], type=old_lbl.schema.field("position_key").type),
            "label_set_id": pa.array([r["label_set_id"] for r in lbl_add], type=old_lbl.schema.field("label_set_id").type),
            "target_kind": pa.array([r["target_kind"] for r in lbl_add], type=old_lbl.schema.field("target_kind").type),
            "value_i16": pa.array([r["value_i16"] for r in lbl_add], type=old_lbl.schema.field("value_i16").type),
            "best_move_u8": pa.array([r["best_move_u8"] for r in lbl_add], type=old_lbl.schema.field("best_move_u8").type),
            "policy_record_id": pa.array([r["policy_record_id"] for r in lbl_add], type=old_lbl.schema.field("policy_record_id").type),
            "has_policy": pa.array([r["has_policy"] for r in lbl_add], type=old_lbl.schema.field("has_policy").type),
            "observation_count": pa.array([r["observation_count"] for r in lbl_add], type=old_lbl.schema.field("observation_count").type),
            "source_cohort": pa.array([r["source_cohort"] for r in lbl_add], type=old_lbl.schema.field("source_cohort").type),
        },
        schema=old_lbl.schema,
    )

    pq.write_table(pa.concat_tables([old_pos, new_pos_tbl]), pos_path, compression="zstd")
    pq.write_table(pa.concat_tables([old_lbl, new_lbl_tbl]), lbl_path, compression="zstd")

    manifest_path = dataset_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        manifest.setdefault("counts", {})
        manifest["counts"]["positions"] = old_pos.num_rows + len(pos_add)
        manifest["counts"]["labels"] = old_lbl.num_rows + len(lbl_add)
        manifest["overnight_append"] = manifest.get("overnight_append", 0) + len(pos_add)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return len(pos_add)


def sync_stragglers(*, dataset_dir: Path | None = None, limit: int = 16) -> dict:
    """Sync a small number of unsynced games (consistency flush, not bulk migration)."""
    dataset_dir = dataset_dir or active_teacher_dir()
    synced = load_synced_ids()
    pending = [g for g in iter_overnight_games() if g[0] not in synced][:limit]
    if not pending:
        return {"games": 0, "new_positions": 0, "unsynced_remaining": 0}

    total_added = 0
    with TeacherSyncLock():
        for game_id, moves, outcome_p0 in pending:
            rows = positions_from_game(moves, outcome_p0)
            if rows:
                total_added += append_to_teacher(rows, dataset_dir)
            synced.add(game_id)
    save_synced_ids(synced)

    remaining = sum(1 for g in iter_overnight_games() if g[0] not in synced)
    return {
        "games": len(pending),
        "new_positions": total_added,
        "unsynced_remaining": remaining,
    }


def sync_single_game(
    game_id: str,
    *,
    dataset_dir: Path | None = None,
    teacher_lock: object | None = None,
) -> dict:
    """Append one overnight game (already in games.db) to teacher parquet if not synced."""
    dataset_dir = dataset_dir or active_teacher_dir()
    synced = load_synced_ids()
    if game_id in synced:
        return {"game_id": game_id, "new_positions": 0, "skipped": True, "counted": False}

    if not GAMES_DB_PATH.is_file():
        raise RuntimeError("games.db missing")

    con = sqlite3.connect(str(GAMES_DB_PATH), timeout=30)
    row = con.execute(
        "SELECT outcome_p0 FROM games WHERE game_id=?",
        (game_id,),
    ).fetchone()
    if not row:
        con.close()
        raise RuntimeError(f"game not in db: {game_id}")

    moves = [
        r[0]
        for r in con.execute(
            "SELECT move_alg FROM game_moves WHERE game_id=? ORDER BY move_num",
            (game_id,),
        )
    ]
    con.close()
    if not moves:
        raise RuntimeError(f"no moves for game: {game_id}")

    rows = positions_from_game(moves, int(row[0]))
    if not rows:
        synced.add(game_id)
        save_synced_ids(synced)
        return {"game_id": game_id, "new_positions": 0, "counted": True}

    # TeacherSyncLock is the CROSS-PROCESS guard -- continuous_pool.py's own
    # threading.Lock (teacher_lock, when passed) only serializes its own
    # worker threads and does nothing against oracle_importer.py, which runs
    # as a separate OS process and calls this same function on the same
    # parquet files. Both processes must go through TeacherSyncLock, or the
    # parquet write can interleave and truncate (see git history: "Page was
    # smaller than expected" corruption from exactly this race).
    with TeacherSyncLock():
        if teacher_lock is not None:
            with teacher_lock:
                added = append_to_teacher(rows, dataset_dir)
        else:
            added = append_to_teacher(rows, dataset_dir)

    synced.add(game_id)
    save_synced_ids(synced)
    return {"game_id": game_id, "new_positions": added, "counted": True}


def sync(*, dataset_dir: Path | None = None) -> dict:
    dataset_dir = dataset_dir or active_teacher_dir()
    synced = load_synced_ids()
    pending = [g for g in iter_overnight_games() if g[0] not in synced]
    if not pending:
        return {"games": 0, "new_positions": 0, "synced_total": len(synced)}

    all_rows: list[dict] = []
    for game_id, moves, outcome_p0 in pending:
        all_rows.extend(positions_from_game(moves, outcome_p0))
        synced.add(game_id)

    with TeacherSyncLock():
        added = append_to_teacher(all_rows, dataset_dir)
    save_synced_ids(synced)
    return {"games": len(pending), "new_positions": added, "synced_total": len(synced)}


def main() -> int:
    from prep_guard import guard_real_work

    guard_real_work("dataset_finalization", detail="sync_overnight_to_teacher.py")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=None)
    args = ap.parse_args()
    stats = sync(dataset_dir=args.dataset or active_teacher_dir())
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
