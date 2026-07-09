"""Database-backed streaming training batches.

The canonical labels database is the source of truth.  This module selects
bounded lists of eligible position IDs, loads one chunk from SQLite, featurizes
that chunk, yields trainer minibatches, then releases the chunk.

No feature cache, epoch memmap, or full-corpus in-memory dictionary is required.
"""
from __future__ import annotations

import json
import math
import sqlite3
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset

_TRAINING = Path(__file__).resolve().parent
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

from build_feature_cache import FV_LEN, record_to_fv
from label_perspective import (
    LABEL_PERSPECTIVE_CONVENTION,
    json_row_target_prob,
    packed_row_target_prob,
    value_i16_to_dataset_stm,
)
from label_resolution import resolve_position_label_bundle
from label_weights import game_phase_from_record
from position_usage_db import ensure_schema, open_labels_db
from titanium_training.data.eval_packed import eval_packed_batch_allow_errors
from titanium_training.models.field_planes import (
    CAT_HEAT,
    ROUTE_CONTESTED,
    ROUTE_ME,
    ROUTE_NEAR_ME,
    ROUTE_NEAR_OPP,
    ROUTE_OPP,
)

DEFAULT_LABELS_DB = _TRAINING / "data" / "canonical" / "labels.db"
FEATURIZE_CHUNK_DEFAULT = 4096
PREFETCH_BATCHES_DEFAULT = 2
MAX_LOADER_MEMORY_BYTES = 2 * 1024**3
OPENING_SANITY_PREFIX = ("e2", "e8", "e3", "e7")


@dataclass(frozen=True)
class DbCounts:
    labeled_positions: int
    eligible_positions: int
    usage_tracked: int


@dataclass(frozen=True)
class LabeledPosition:
    position_id: str
    packed_state: bytes
    value_target: float
    sample_weight: float = 1.0
    storage_kind: str = "json"
    source_tier: str = "unknown"
    game_phase: str = "midgame"
    policy_target: object | None = None
    dataset_side_to_move: int | None = None
    value_dataset_stm: float | None = None


@dataclass
class DbTrainingBatch:
    position_ids: list[str]
    features: np.ndarray
    value_targets: np.ndarray
    sample_weights: np.ndarray
    source_tiers: list[str]
    game_phases: list[str]
    policy_targets: object | None


def _main_db_path(con: sqlite3.Connection) -> Path | None:
    for _seq, name, file_name in con.execute("PRAGMA database_list").fetchall():
        if name == "main" and file_name:
            return Path(str(file_name))
    return None


def _prepare_opening_sanity_filter(con: sqlite3.Connection) -> bool:
    """Attach sibling games.db and cache positions from sane four-ply openings.

    Collapsed self-play often starts with wall spam or malformed protocol tokens.
    A game must survive two pawn moves per side on the center file:
    white e2, black e8, white e3, black e7.  If a labels DB has no sibling
    games.db, leave sampling unchanged (teacher rows and isolated tests).
    """
    if con.execute(
        "SELECT 1 FROM sqlite_temp_master WHERE type='table' AND name='sane_opening_positions'"
    ).fetchone():
        return True

    labels_path = _main_db_path(con)
    if labels_path is None:
        return False
    games_path = labels_path.with_name("games.db")
    if not games_path.is_file():
        return False

    attached = {
        str(row[1])
        for row in con.execute("PRAGMA database_list").fetchall()
    }
    if "opening_games" not in attached:
        con.execute("ATTACH DATABASE ? AS opening_games", (str(games_path),))
    has_moves = con.execute(
        """
        SELECT 1
        FROM opening_games.sqlite_master
        WHERE type='table' AND name='game_moves'
        """
    ).fetchone()
    if not has_moves:
        return False

    con.executescript(
        """
        CREATE TEMP TABLE IF NOT EXISTS sane_opening_games(
            game_id TEXT PRIMARY KEY
        );
        CREATE TEMP TABLE IF NOT EXISTS sane_opening_positions(
            pos_key TEXT PRIMARY KEY
        );
        DELETE FROM sane_opening_games;
        DELETE FROM sane_opening_positions;
        """
    )
    con.execute(
        """
        INSERT OR IGNORE INTO sane_opening_games(game_id)
        SELECT game_id
        FROM opening_games.game_moves
        WHERE move_num BETWEEN 0 AND 3
        GROUP BY game_id
        HAVING COUNT(DISTINCT move_num) = 4
           AND SUM(CASE WHEN move_num = 0 AND move_alg = ? THEN 1 ELSE 0 END) = 1
           AND SUM(CASE WHEN move_num = 1 AND move_alg = ? THEN 1 ELSE 0 END) = 1
           AND SUM(CASE WHEN move_num = 2 AND move_alg = ? THEN 1 ELSE 0 END) = 1
           AND SUM(CASE WHEN move_num = 3 AND move_alg = ? THEN 1 ELSE 0 END) = 1
        """,
        OPENING_SANITY_PREFIX,
    )
    con.execute(
        """
        INSERT OR IGNORE INTO sane_opening_positions(pos_key)
        SELECT DISTINCT gm.pos_key
        FROM opening_games.game_moves gm
        JOIN sane_opening_games sg ON sg.game_id = gm.game_id
        """
    )
    return True


def _json_opening_filter(con: sqlite3.Connection) -> str:
    if not _prepare_opening_sanity_filter(con):
        return ""
    return "AND p.pos_key IN (SELECT pos_key FROM sane_opening_positions)"


def _usage_queue_available(con: sqlite3.Connection) -> bool:
    ensure_schema(con)
    row = con.execute("SELECT COUNT(*) FROM position_usage").fetchone()
    return bool(row and int(row[0]) > 0)


def _usage_active_where(*, include_replay: bool = False, retired_only: bool = False) -> str:
    if retired_only:
        base = "u.retired = 1 AND u.protected_replay = 0"
    elif include_replay:
        base = "(u.retired = 0 OR u.protected_replay = 1)"
    else:
        base = "u.retired = 0 AND u.protected_replay = 0"
    return (
        base
        + " AND (u.retirement_reason IS NULL OR u.retirement_reason != 'opening_sanity_failed')"
        + " AND u.source != 'opening_sanity'"
    )


def _usage_count(con: sqlite3.Connection, where_sql: str) -> int:
    row = con.execute(f"SELECT COUNT(*) FROM position_usage u WHERE {where_sql}").fetchone()
    return int(row[0]) if row else 0


def _sample_usage_by_rowid(
    con: sqlite3.Connection,
    *,
    limit: int,
    seed: int,
    where_sql: str,
    prefer_low_visits: bool = True,
) -> list[str]:
    """Fast bounded sampler over the materialized position_usage queue.

    Avoids ORDER BY RANDOM() over million-row source joins.  The queue is
    append-ish and rowid-addressable, so several small rowid windows give enough
    diversity while keeping startup bounded.
    """
    if limit <= 0:
        return []
    max_rowid = con.execute("SELECT MAX(rowid) FROM position_usage").fetchone()[0]
    if not max_rowid:
        return []
    rng = np.random.default_rng(seed)
    selected: list[str] = []
    seen: set[str] = set()
    del prefer_low_visits
    # Read bounded rowid slices without ORDER BY; sorting even modest windows is
    # the startup killer on the 20+ GB canonical DB.  Random starts plus a final
    # shuffle give enough diversity for an epoch sample.
    window_limit = min(max(limit // 8, 512), 8192)
    window_span = max(window_limit * 8, 8192)
    for _ in range(128):
        if len(selected) >= limit:
            break
        start = int(rng.integers(1, int(max_rowid) + 1))
        end = min(int(max_rowid), start + window_span)
        rows = con.execute(
            f"""
            SELECT u.pos_key, u.source
            FROM position_usage u
            WHERE u.rowid BETWEEN ? AND ?
              AND {where_sql}
            LIMIT ?
            """,
            (start, end, window_limit),
        ).fetchall()
        for key, source in rows:
            key = str(key)
            if not (key.startswith("json:") or key.startswith("teacher:")):
                key = f"json:{key}"
            if key not in seen:
                seen.add(key)
                selected.append(key)
                if len(selected) >= limit:
                    break
    return selected[:limit]


def _normalize_usage_key(key: object) -> str:
    text = str(key)
    if text.startswith("json:") or text.startswith("teacher:"):
        return text
    return f"json:{text}"


def _fetch_usage_keys(con: sqlite3.Connection, sql: str, params: tuple = ()) -> list[str]:
    return [_normalize_usage_key(row[0]) for row in con.execute(sql, params).fetchall()]


def _usage_db_key(key: str) -> str:
    return key[5:] if key.startswith("json:") else key


def _install_epoch_excluded_keys(con: sqlite3.Connection, excluded_keys: list[str]) -> None:
    """Install per-epoch exclusion keys in a temp table on this connection."""
    con.executescript(
        """
        CREATE TEMP TABLE IF NOT EXISTS epoch_excluded_keys(
            pos_key TEXT PRIMARY KEY
        );
        DELETE FROM epoch_excluded_keys;
        """
    )
    if excluded_keys:
        con.executemany(
            "INSERT OR IGNORE INTO epoch_excluded_keys(pos_key) VALUES (?)",
            ((key,) for key in excluded_keys),
        )


class LabelsRepository:
    """Small repository wrapper around canonical labels.db."""

    def __init__(self, labels_db: Path = DEFAULT_LABELS_DB):
        self.labels_db = Path(labels_db)
        self.con = open_labels_db(self.labels_db)

    def close(self) -> None:
        self.con.close()

    def load_labeled_positions(self, selected_ids: list[str]) -> list[LabeledPosition]:
        if not selected_ids:
            return []
        json_ids = [pid[5:] for pid in selected_ids if pid.startswith("json:")]
        teacher_ids = [bytes.fromhex(pid[8:]) for pid in selected_ids if pid.startswith("teacher:")]
        rows_by_key: dict[str, tuple[bytes, float, float, str, str]] = {}
        labels_by_pos: dict[str, list[tuple[str, float, int]]] = {}
        for offset in range(0, len(json_ids), 500):
            ids = json_ids[offset : offset + 500]
            placeholders = ",".join("?" * len(ids))
            label_rows = self.con.execute(
                f"""
                SELECT l.pos_key, l.source, l.value_stm, COALESCE(l.n_samples, 1)
                FROM labels l
                WHERE l.pos_key IN ({placeholders})
                """,
                ids,
            ).fetchall()
            for pos_key, source, value_stm, n_samples in label_rows:
                key = str(pos_key)
                labels_by_pos.setdefault(key, []).append(
                    (str(source), float(value_stm), int(n_samples or 1))
                )

            pos_rows = self.con.execute(
                f"""
                SELECT p.pos_key, p.position_data
                FROM positions p
                WHERE p.pos_key IN ({placeholders})
                """,
                ids,
            ).fetchall()
            for pos_key, data in pos_rows:
                key = str(pos_key)
                label_list = labels_by_pos.get(key, [])
                engine_eval = next(
                    (float(v) for s, v, _n in label_list if s.endswith("_engine")),
                    None,
                )
                raw = bytes(data) if isinstance(data, bytes) else str(data).encode("utf-8")
                try:
                    rec = json.loads(raw.decode("utf-8"))
                    phase = game_phase_from_record(rec)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    phase = "midgame"
                bundle = resolve_position_label_bundle(
                    label_list,
                    engine_eval_stm=engine_eval,
                    game_phase=phase,
                )
                if bundle is None:
                    continue
                target = json_row_target_prob(float(bundle.target))
                rows_by_key["json:" + key] = (
                    raw,
                    target,
                    float(bundle.loss_weight),
                    str(bundle.source_tier),
                    str(bundle.game_phase),
                )
        teacher_by_key: dict[str, tuple[bytes, float, int]] = {}
        for offset in range(0, len(teacher_ids), 500):
            ids = teacher_ids[offset : offset + 500]
            placeholders = ",".join("?" * len(ids))
            rows = self.con.execute(
                f"""
                SELECT p.position_key, p.packed_state, p.side_to_move,
                       SUM(l.value_i16 * COALESCE(l.observation_count, 1)) * 1.0
                         / NULLIF(SUM(COALESCE(l.observation_count, 1)), 0)
                FROM teacher_positions p
                JOIN teacher_labels l ON l.position_key = p.position_key
                WHERE p.position_key IN ({placeholders})
                  AND l.value_i16 IS NOT NULL
                GROUP BY p.position_key, p.packed_state, p.side_to_move
                """,
                ids,
            ).fetchall()
            for position_key, packed_state, dataset_stm, value_i16 in rows:
                teacher_by_key["teacher:" + bytes(position_key).hex()] = (
                    bytes(packed_state),
                    value_i16_to_dataset_stm(int(round(float(value_i16)))),
                    int(dataset_stm),
                )

        out: list[LabeledPosition] = []
        for pid in selected_ids:
            if pid.startswith("json:") and pid in rows_by_key:
                raw, target, sample_weight, source_tier, game_phase = rows_by_key[pid]
                out.append(
                    LabeledPosition(
                        pid,
                        raw,
                        target,
                        sample_weight,
                        "json",
                        source_tier,
                        game_phase,
                    )
                )
            elif pid.startswith("teacher:") and pid in teacher_by_key:
                raw, value_dataset_stm, dataset_stm = teacher_by_key[pid]
                out.append(
                    LabeledPosition(
                        pid,
                        raw,
                        0.0,
                        "packed",
                        dataset_side_to_move=dataset_stm,
                        value_dataset_stm=value_dataset_stm,
                    )
                )
        return out

    def load_policy_targets(self, selected_ids: list[str]) -> object | None:
        return None


def db_counts(labels_db: Path = DEFAULT_LABELS_DB) -> DbCounts:
    con = open_labels_db(labels_db)
    try:
        if _usage_queue_available(con):
            labeled = con.execute("SELECT COUNT(*) FROM position_usage").fetchone()[0]
            eligible = _usage_count(con, _usage_active_where(include_replay=True))
            usage = labeled
            return DbCounts(int(labeled), int(eligible), int(usage))
        opening_filter = _json_opening_filter(con)
        labeled = con.execute(
            f"""
            SELECT COUNT(DISTINCT 'json:' || p.pos_key)
            FROM positions p
            JOIN labels l ON l.pos_key = p.pos_key
            WHERE 1=1
              {opening_filter}
            """
        ).fetchone()[0]
        teacher_labeled = con.execute(
            """
            SELECT COUNT(DISTINCT 'teacher:' || hex(position_key))
            FROM teacher_labels
            WHERE value_i16 IS NOT NULL
            """
        ).fetchone()[0]
        ensure_schema(con)
        usage = con.execute("SELECT COUNT(*) FROM position_usage").fetchone()[0]
        json_eligible = con.execute(
            f"""
            SELECT COUNT(DISTINCT 'json:' || p.pos_key)
            FROM positions p
            JOIN labels l ON l.pos_key = p.pos_key
            LEFT JOIN position_usage u ON u.pos_key = 'json:' || p.pos_key
            WHERE (u.pos_key IS NULL
               OR u.retired = 0
               OR u.protected_replay = 1)
              {opening_filter}
            """
        ).fetchone()[0]
        teacher_eligible = con.execute(
            """
            SELECT COUNT(DISTINCT 'teacher:' || hex(t.position_key))
            FROM teacher_labels t
            LEFT JOIN position_usage u ON u.pos_key = 'teacher:' || hex(t.position_key)
            WHERE t.value_i16 IS NOT NULL
              AND (u.pos_key IS NULL OR u.retired = 0 OR u.protected_replay = 1)
            """
        ).fetchone()[0]
        return DbCounts(int(labeled + teacher_labeled), int(json_eligible + teacher_eligible), int(usage))
    finally:
        con.close()


def sample_epoch_keys(
    con: sqlite3.Connection,
    *,
    epoch_size: int,
    seed: int = 0,
    ordinary_fraction: float = 0.80,
    retired_replay_fraction: float = 0.0,
    old_refresh_fraction: float = 0.05,
    full_active_epoch: bool = False,
) -> list[str]:
    """Build one streaming epoch from the live training queue.

    Production streaming is intentionally new-data dominated: take up to
    ``epoch_size`` active generated positions that have not been trained yet,
    then add a small low-visit refresh from older/teacher rows.  This keeps each
    trigger focused on the freshly generated 2,048-position batch without
    overfitting forever on one frozen slice of the historical corpus.

    The legacy fallback below is retained for tiny tests and DBs without a
    materialized ``position_usage`` queue.
    """
    ensure_schema(con)
    rng = np.random.default_rng(seed)

    def fetch_excluding_seen(sql_base: str, seen: list[str], needed: int) -> list[str]:
        if needed <= 0:
            return []
        if seen:
            _install_epoch_excluded_keys(con, seen)
            exclude_sql = "AND u.pos_key NOT IN (SELECT pos_key FROM epoch_excluded_keys)"
        else:
            exclude_sql = ""
        return _fetch_usage_keys(
            con,
            f"{sql_base}\n                {exclude_sql}\n                ORDER BY COALESCE(u.training_visits, 0) ASC, u.rowid ASC\n                LIMIT ?",
            (needed,),
        )

    if _usage_queue_available(con):
        if full_active_epoch:
            keys = _fetch_usage_keys(
                con,
                f"""
                SELECT u.pos_key
                FROM position_usage u
                WHERE {_usage_active_where(include_replay=True)}
                ORDER BY COALESCE(u.training_visits, 0) ASC, u.rowid ASC
                """,
            )
            rng.shuffle(keys)
            return keys

        n_new = max(0, int(epoch_size))
        n_old = max(0, int(math.ceil(n_new * max(0.0, old_refresh_fraction))))

        active_base = _usage_active_where(include_replay=False)
        new_keys = _fetch_usage_keys(
            con,
            f"""
            SELECT u.pos_key
            FROM position_usage u
            WHERE {active_base}
              AND u.source = 'canonical_json'
              AND COALESCE(u.training_visits, 0) = 0
            ORDER BY u.rowid ASC
            LIMIT ?
            """,
            (n_new,),
        )

        merged = list(dict.fromkeys(new_keys))
        if len(merged) < n_new:
            db_seen = [_usage_db_key(key) for key in merged]
            fill = fetch_excluding_seen(
                f"""
                SELECT u.pos_key
                FROM position_usage u
                WHERE {active_base}
                """,
                db_seen,
                n_new - len(merged),
            )
            for key in fill:
                if key not in merged:
                    merged.append(key)

        if n_old > 0:
            db_seen = [_usage_db_key(key) for key in merged]
            old = fetch_excluding_seen(
                f"""
                SELECT u.pos_key
                FROM position_usage u
                WHERE {_usage_active_where(include_replay=True)}
                """,
                db_seen,
                n_old,
            )
            for key in old:
                if key not in merged:
                    merged.append(key)

        rng.shuffle(merged)
        return merged

    n_ordinary = int(epoch_size * ordinary_fraction)
    n_replay = max(0, epoch_size - n_ordinary)
    n_retired_replay = max(0, int(epoch_size * max(0.0, retired_replay_fraction)))
    opening_filter = _json_opening_filter(con)

    def fetch(sql: str, limit: int) -> list[str]:
        if limit <= 0:
            return []
        return [str(r[0]) for r in con.execute(sql, (limit,)).fetchall()]

    active_join = """
        FROM positions p
        JOIN labels l ON l.pos_key = p.pos_key
        LEFT JOIN position_usage u ON u.pos_key = 'json:' || p.pos_key
        WHERE (u.pos_key IS NULL OR u.retired = 0 OR u.protected_replay = 1)
    """
    retired_join = """
        FROM positions p
        JOIN labels l ON l.pos_key = p.pos_key
        JOIN position_usage u ON u.pos_key = 'json:' || p.pos_key
        WHERE u.retired = 1
          AND COALESCE(u.protected_replay, 0) = 0
    """
    ordinary = fetch(
        f"""
        SELECT DISTINCT 'json:' || p.pos_key {active_join}
          AND COALESCE(u.protected_replay, 0) = 0
          {opening_filter}
        ORDER BY COALESCE(u.training_visits, 0) ASC, RANDOM() LIMIT ?
        """,
        n_ordinary,
    )
    teacher_ordinary = fetch(
        """
        SELECT DISTINCT 'teacher:' || hex(t.position_key)
        FROM teacher_labels t
        LEFT JOIN position_usage u ON u.pos_key = 'teacher:' || hex(t.position_key)
        WHERE t.value_i16 IS NOT NULL
          AND COALESCE(u.protected_replay, 0) = 0
          AND (u.pos_key IS NULL OR u.retired = 0 OR u.protected_replay = 1)
        ORDER BY COALESCE(u.training_visits, 0) ASC, RANDOM() LIMIT ?
        """,
        max(0, n_ordinary - len(ordinary)),
    )
    ordinary.extend(teacher_ordinary)
    replay = fetch(
        f"""
        SELECT DISTINCT 'json:' || p.pos_key {active_join}
          {opening_filter}
          AND (
            COALESCE(u.protected_replay, 0) = 1
            OR COALESCE(u.training_visits, 0) > 0
            OR p.pos_key IN (
                SELECT pos_key FROM labels
                WHERE source LIKE '%zeroink%' OR source LIKE '%oracle%' OR source LIKE '%wallz%'
            )
          )
        ORDER BY RANDOM() LIMIT ?
        """,
        n_replay,
    )
    replay.extend(
        fetch(
            """
            SELECT DISTINCT 'teacher:' || hex(t.position_key)
            FROM teacher_labels t
            LEFT JOIN position_usage u ON u.pos_key = 'teacher:' || hex(t.position_key)
            WHERE t.value_i16 IS NOT NULL
              AND (COALESCE(u.protected_replay, 0) = 1 OR t.source_cohort != '')
              AND (u.pos_key IS NULL OR u.retired = 0 OR u.protected_replay = 1)
            ORDER BY RANDOM() LIMIT ?
            """,
            max(0, n_replay - len(replay)),
        )
    )
    merged = list(dict.fromkeys(ordinary + replay))
    if len(merged) < epoch_size:
        fill = fetch(
            f"""
            SELECT DISTINCT 'json:' || p.pos_key {active_join}
              {opening_filter}
            ORDER BY COALESCE(u.training_visits, 0) ASC, RANDOM() LIMIT ?
            """,
            epoch_size - len(merged),
        )
        if len(fill) < epoch_size - len(merged):
            fill.extend(
                fetch(
                    """
                    SELECT DISTINCT 'teacher:' || hex(t.position_key)
                    FROM teacher_labels t
                    LEFT JOIN position_usage u ON u.pos_key = 'teacher:' || hex(t.position_key)
                    WHERE t.value_i16 IS NOT NULL
                      AND (u.pos_key IS NULL OR u.retired = 0 OR u.protected_replay = 1)
                    ORDER BY COALESCE(u.training_visits, 0) ASC, RANDOM() LIMIT ?
                    """,
                    epoch_size - len(merged) - len(fill),
                )
            )
        for key in fill:
            if key not in merged:
                merged.append(key)
    merged = merged[:epoch_size]

    retired_replay: list[str] = []
    if n_retired_replay > 0:
        retired_replay.extend(
            fetch(
                f"""
                SELECT DISTINCT 'json:' || p.pos_key {retired_join}
                  {opening_filter}
                  AND COALESCE(u.retirement_reason, '') != 'opening_sanity_failed'
                ORDER BY COALESCE(u.training_visits, 0) ASC, RANDOM() LIMIT ?
                """,
                n_retired_replay,
            )
        )
        retired_replay.extend(
            fetch(
                """
                SELECT DISTINCT 'teacher:' || hex(t.position_key)
                FROM teacher_labels t
                JOIN position_usage u ON u.pos_key = 'teacher:' || hex(t.position_key)
                WHERE t.value_i16 IS NOT NULL
                  AND u.retired = 1
                  AND COALESCE(u.protected_replay, 0) = 0
                  AND COALESCE(u.retirement_reason, '') != 'opening_sanity_failed'
                ORDER BY COALESCE(u.training_visits, 0) ASC, RANDOM() LIMIT ?
                """,
                max(0, n_retired_replay - len(retired_replay)),
            )
        )
    for key in retired_replay:
        if key not in merged:
            merged.append(key)
    rng.shuffle(merged)
    return merged


def _featurize_records(
    rows: list[LabeledPosition],
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    ids: list[str] = []
    features: list[np.ndarray] = []
    targets: list[float] = []
    weights: list[float] = []
    tiers: list[str] = []
    phases: list[str] = []
    packed_rows = [row for row in rows if row.storage_kind == "packed"]
    json_rows = [row for row in rows if row.storage_kind != "packed"]
    for row in json_rows:
        try:
            rec = json.loads(row.packed_state.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        target = float(row.value_target)
        fv = record_to_fv(rec, target)
        if fv is None or fv.shape != (FV_LEN,):
            continue
        ids.append(row.position_id)
        features.append(fv)
        targets.append(target)
        weights.append(float(row.sample_weight))
        tiers.append(str(row.source_tier))
        phases.append(str(row.game_phase))
    if packed_rows:
        evals = eval_packed_batch_allow_errors(
            [(i, row.packed_state) for i, row in enumerate(packed_rows)]
        )
        for row, rec in zip(packed_rows, evals, strict=False):
            if not rec.get("ok", False):
                continue
            if row.dataset_side_to_move is None or row.value_dataset_stm is None:
                continue
            engine_turn = int(rec.get("turn", -1))
            if engine_turn not in (0, 1):
                continue
            target = packed_row_target_prob(
                value_dataset_stm=row.value_dataset_stm,
                engine_turn=engine_turn,
                dataset_side_to_move=row.dataset_side_to_move,
            )
            fv = record_to_fv(rec, target)
            if fv is None or fv.shape != (FV_LEN,):
                continue
            ids.append(row.position_id)
            features.append(fv)
            targets.append(target)
            weights.append(float(row.sample_weight))
            tiers.append(str(row.source_tier))
            phases.append(str(row.game_phase))
    if not features:
        return (
            [],
            np.empty((0, FV_LEN), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            [],
            [],
        )
    return (
        ids,
        np.asarray(features, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        np.asarray(weights, dtype=np.float32),
        tiers,
        phases,
    )


def iter_db_training_batches(
    repository: LabelsRepository,
    selected_ids: list[str],
    featurize=None,
    *,
    chunk_size: int = FEATURIZE_CHUNK_DEFAULT,
) -> Iterator[DbTrainingBatch]:
    """Yield bounded database training chunks.

    ``featurize`` is accepted for the public API shape.  The current repository
    stores eval JSON; we use the existing ``record_to_fv`` converter directly.
    """
    del featurize
    for offset in range(0, len(selected_ids), chunk_size):
        ids = selected_ids[offset : offset + chunk_size]
        rows = repository.load_labeled_positions(ids)
        ok_ids, features, value_targets, sample_weights, source_tiers, game_phases = _featurize_records(rows)
        if features.shape != (len(ok_ids), FV_LEN):
            raise RuntimeError(
                f"Unexpected features shape: {features.shape}; expected ({len(ok_ids)}, {FV_LEN})"
            )
        yield DbTrainingBatch(
            position_ids=ok_ids,
            features=features,
            value_targets=value_targets,
            sample_weights=sample_weights,
            source_tiers=source_tiers,
            game_phases=game_phases,
            policy_targets=repository.load_policy_targets(ok_ids),
        )


def features_to_torch_batch(
    features: np.ndarray,
    position_ids: list[str],
    *,
    sample_weights: np.ndarray | None = None,
    source_tiers: list[str] | None = None,
    game_phases: list[str] | None = None,
) -> dict:
    """Convert a feature matrix slice into the existing trainer batch schema."""
    batch = {
        "_pos_keys": position_ids,
        "target": torch.from_numpy(features[:, 0].copy()).float(),
        "d_me": torch.from_numpy(features[:, 1].copy()).float(),
        "d_opp": torch.from_numpy(features[:, 2].copy()).float(),
        "w_me": torch.from_numpy(features[:, 3].copy()).float(),
        "w_opp": torch.from_numpy(features[:, 4].copy()).float(),
        "legal_wall_norm": torch.from_numpy(features[:, 5].copy()).float(),
        "width_opp": torch.from_numpy(features[:, 6].copy()).float(),
        "legal_cross_me_norm": torch.from_numpy(features[:, 7].copy()).float(),
        "legal_cross_opp_norm": torch.from_numpy(features[:, 8].copy()).float(),
        "cat_best_me_norm": torch.from_numpy(features[:, 9].copy()).float(),
        "cat_best_opp_norm": torch.from_numpy(features[:, 10].copy()).float(),
        "wall_mask": torch.from_numpy(features[:, 11:139].copy()).float(),
        ROUTE_ME: torch.from_numpy(features[:, 139:220].copy()).float(),
        ROUTE_OPP: torch.from_numpy(features[:, 220:301].copy()).float(),
        ROUTE_NEAR_ME: torch.from_numpy(features[:, 301:382].copy()).float(),
        ROUTE_NEAR_OPP: torch.from_numpy(features[:, 382:463].copy()).float(),
        ROUTE_CONTESTED: torch.from_numpy(features[:, 463:544].copy()).float(),
        CAT_HEAT: torch.from_numpy(features[:, 544:625].copy()).float(),
        "bucket": torch.from_numpy(features[:, 625].astype(np.int64, copy=True)),
        "pawn_me": torch.from_numpy(features[:, 626].astype(np.int64, copy=True)),
        "pawn_opp": torch.from_numpy(features[:, 627].astype(np.int64, copy=True)),
    }
    if sample_weights is not None and len(sample_weights) == len(position_ids):
        batch["sample_weight"] = torch.from_numpy(sample_weights.astype(np.float32, copy=True)).float()
    if source_tiers is not None and len(source_tiers) == len(position_ids):
        batch["_source_tier"] = list(source_tiers)
    if game_phases is not None and len(game_phases) == len(position_ids):
        batch["_game_phase"] = list(game_phases)
    return batch


class DbTrainingIterableDataset(IterableDataset):
    """IterableDataset that yields trainer minibatches from bounded DB chunks."""

    FV_LEN = FV_LEN

    def __init__(
        self,
        labels_db: Path,
        selected_ids: list[str],
        *,
        trainer_batch_size: int = 512,
        chunk_size: int = FEATURIZE_CHUNK_DEFAULT,
    ):
        self.labels_db = Path(labels_db)
        self.selected_ids = list(selected_ids)
        self.trainer_batch_size = max(1, int(trainer_batch_size))
        self.chunk_size = max(1, int(chunk_size))

    def __len__(self) -> int:
        return len(self.selected_ids)

    def __iter__(self) -> Iterator[dict]:
        repo = LabelsRepository(self.labels_db)
        try:
            for chunk in iter_db_training_batches(repo, self.selected_ids, chunk_size=self.chunk_size):
                n = len(chunk.position_ids)
                for start in range(0, n, self.trainer_batch_size):
                    end = min(n, start + self.trainer_batch_size)
                    yield features_to_torch_batch(
                        chunk.features[start:end],
                        chunk.position_ids[start:end],
                        sample_weights=chunk.sample_weights[start:end],
                        source_tiers=chunk.source_tiers[start:end],
                        game_phases=chunk.game_phases[start:end],
                    )
                # chunk arrays are released before the next iteration
        finally:
            repo.close()
