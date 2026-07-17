#!/usr/bin/env python3
"""Import/audit teacher_dataset_good into labels.db packed-teacher tables.

This preserves the existing JSON labels schema and adds packed-state teacher
tables for the database-backed trainer:
  - teacher_positions: one packed_state per position_key
  - teacher_labels: provenance/observation-aware labels keyed by source cohort

The import is idempotent and transactional.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

_TRAINING = Path(__file__).resolve().parents[1]
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

from db_import import LABELS_DB_PATH
from position_usage_db import ensure_schema, increment_new_eligible, open_labels_db, upsert_positions

TEACHER_DIR = _TRAINING / "data" / "teacher_dataset_good"

TEACHER_SQL = """
CREATE TABLE IF NOT EXISTS teacher_positions (
    position_key       BLOB PRIMARY KEY,
    canonical_hash     BLOB,
    packed_state       BLOB NOT NULL,
    side_to_move       INTEGER NOT NULL,
    provenance         TEXT NOT NULL,
    source_flags       INTEGER,
    total_observations INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS teacher_labels (
    position_key       BLOB NOT NULL,
    label_set_id       TEXT NOT NULL,
    target_kind        TEXT NOT NULL,
    value_i16          INTEGER,
    best_move_u8       INTEGER,
    policy_record_id   INTEGER,
    has_policy         INTEGER NOT NULL DEFAULT 0,
    observation_count  INTEGER NOT NULL DEFAULT 1,
    source_cohort      TEXT NOT NULL,
    updated_at         TEXT NOT NULL,
    PRIMARY KEY(position_key, label_set_id, target_kind, source_cohort)
);

CREATE INDEX IF NOT EXISTS idx_teacher_labels_position ON teacher_labels(position_key);
CREATE INDEX IF NOT EXISTS idx_teacher_labels_source ON teacher_labels(source_cohort);
CREATE INDEX IF NOT EXISTS idx_teacher_labels_policy ON teacher_labels(has_policy);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_teacher_schema(con: sqlite3.Connection) -> None:
    ensure_schema(con)
    con.executescript(TEACHER_SQL)


def manifest_parts(dataset_dir: Path) -> tuple[Path, Path]:
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    root = dataset_dir.parents[2]
    positions = root / manifest["parts"]["positions"][0]
    labels = root / manifest["parts"]["labels"][0]
    return positions, labels


def audit(con: sqlite3.Connection) -> dict:
    ensure_teacher_schema(con)
    out: dict = {}
    out["json_position_rows"] = con.execute("SELECT COUNT(*) FROM positions").fetchone()[0]
    out["json_unique_value_labeled_positions"] = con.execute(
        """
        SELECT COUNT(DISTINCT p.pos_key)
        FROM positions p JOIN labels l ON l.pos_key = p.pos_key
        WHERE l.value_stm IS NOT NULL
        """
    ).fetchone()[0]
    out["json_label_rows"] = con.execute("SELECT COUNT(*) FROM labels").fetchone()[0]
    out["json_label_observations"] = con.execute("SELECT COALESCE(SUM(n_samples), 0) FROM labels").fetchone()[0]
    out["teacher_positions"] = con.execute("SELECT COUNT(*) FROM teacher_positions").fetchone()[0]
    out["teacher_unique_value_labeled_positions"] = con.execute(
        """
        SELECT COUNT(DISTINCT position_key)
        FROM teacher_labels
        WHERE value_i16 IS NOT NULL
        """
    ).fetchone()[0]
    out["teacher_unique_policy_labeled_positions"] = con.execute(
        """
        SELECT COUNT(DISTINCT position_key)
        FROM teacher_labels
        WHERE has_policy = 1
        """
    ).fetchone()[0]
    out["teacher_label_rows"] = con.execute("SELECT COUNT(*) FROM teacher_labels").fetchone()[0]
    out["teacher_label_observations"] = con.execute(
        "SELECT COALESCE(SUM(observation_count), 0) FROM teacher_labels"
    ).fetchone()[0]
    out["usage"] = [
        {
            "source": row[0],
            "total": int(row[1]),
            "retired": int(row[2] or 0),
            "protected": int(row[3] or 0),
        }
        for row in con.execute(
            """
            SELECT source, COUNT(*),
                   SUM(CASE WHEN retired = 1 AND protected_replay = 0 THEN 1 ELSE 0 END),
                   SUM(CASE WHEN protected_replay = 1 THEN 1 ELSE 0 END)
            FROM position_usage
            GROUP BY source
            ORDER BY COUNT(*) DESC
            """
        ).fetchall()
    ]
    out["source_breakdown"] = [
        {
            "source_cohort": row[0],
            "label_rows": int(row[1]),
            "unique_positions": int(row[2]),
            "observations": int(row[3] or 0),
        }
        for row in con.execute(
            """
            SELECT source_cohort, COUNT(*), COUNT(DISTINCT position_key),
                   COALESCE(SUM(observation_count), 0)
            FROM teacher_labels
            GROUP BY source_cohort
            ORDER BY COUNT(*) DESC
            """
        ).fetchall()
    ]
    return out


def import_teacher(dataset_dir: Path, labels_db: Path) -> dict:
    positions_path, labels_path = manifest_parts(dataset_dir)
    con = open_labels_db(labels_db)
    ensure_teacher_schema(con)
    ts = utc_now()
    inserted_usage = 0
    try:
        before = audit(con)
        con.execute("BEGIN IMMEDIATE")

        pos_table = pq.read_table(positions_path)
        pos_rows = []
        usage_keys: list[str] = []
        for batch in pos_table.to_batches(max_chunksize=10_000):
            cols = {name: batch.column(name) for name in batch.schema.names}
            pos_rows.clear()
            usage_keys.clear()
            for i in range(batch.num_rows):
                key = cols["position_key"][i].as_py()
                packed = cols["packed_state"][i].as_py()
                pos_rows.append(
                    (
                        key,
                        cols["canonical_hash"][i].as_py(),
                        packed,
                        int(cols["side_to_move"][i].as_py()),
                        "teacher_dataset_good",
                        int(cols["source_flags"][i].as_py() or 0),
                        int(cols["total_observations"][i].as_py() or 0),
                    )
                )
                usage_keys.append("teacher:" + bytes(key).hex())
            con.executemany(
                """
                INSERT INTO teacher_positions(
                    position_key, canonical_hash, packed_state, side_to_move,
                    provenance, source_flags, total_observations
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(position_key) DO UPDATE SET
                    total_observations = MAX(total_observations, excluded.total_observations)
                """,
                pos_rows,
            )
            inserted_usage += upsert_positions(
                con,
                usage_keys,
                source="teacher_dataset_good",
                protected_replay=True,
            )

        label_table = pq.read_table(labels_path)
        label_rows = []
        for batch in label_table.to_batches(max_chunksize=10_000):
            cols = {name: batch.column(name) for name in batch.schema.names}
            label_rows.clear()
            for i in range(batch.num_rows):
                label_rows.append(
                    (
                        cols["position_key"][i].as_py(),
                        str(cols["label_set_id"][i].as_py()),
                        str(cols["target_kind"][i].as_py()),
                        None if cols["value_i16"][i].as_py() is None else int(cols["value_i16"][i].as_py()),
                        None if cols["best_move_u8"][i].as_py() is None else int(cols["best_move_u8"][i].as_py()),
                        None
                        if cols["policy_record_id"][i].as_py() is None
                        else int(cols["policy_record_id"][i].as_py()),
                        1 if bool(cols["has_policy"][i].as_py()) else 0,
                        int(cols["observation_count"][i].as_py() or 1),
                        str(cols["source_cohort"][i].as_py() or ""),
                        ts,
                    )
                )
            con.executemany(
                """
                INSERT INTO teacher_labels(
                    position_key, label_set_id, target_kind, value_i16,
                    best_move_u8, policy_record_id, has_policy,
                    observation_count, source_cohort, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(position_key, label_set_id, target_kind, source_cohort)
                DO UPDATE SET
                    value_i16 = COALESCE(excluded.value_i16, teacher_labels.value_i16),
                    best_move_u8 = COALESCE(excluded.best_move_u8, teacher_labels.best_move_u8),
                    policy_record_id = COALESCE(excluded.policy_record_id, teacher_labels.policy_record_id),
                    has_policy = MAX(teacher_labels.has_policy, excluded.has_policy),
                    observation_count = MAX(teacher_labels.observation_count, excluded.observation_count),
                    updated_at = excluded.updated_at
                """,
                label_rows,
            )

        # Historical corpus becomes visible to sampling, but imported history should
        # not count as freshly generated trigger work.
        increment_new_eligible(con, 0)
        con.execute("COMMIT")
        after = audit(con)
        return {"before": before, "after": after, "usage_rows_created": inserted_usage}
    except Exception:
        con.execute("ROLLBACK")
        raise
    finally:
        con.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", type=Path, default=TEACHER_DIR)
    ap.add_argument("--labels-db", type=Path, default=LABELS_DB_PATH)
    ap.add_argument("--audit-only", action="store_true")
    args = ap.parse_args()
    con = open_labels_db(args.labels_db)
    try:
        if args.audit_only:
            print(json.dumps(audit(con), indent=2))
            return 0
    finally:
        con.close()
    print(json.dumps(import_teacher(args.dataset_dir, args.labels_db), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
