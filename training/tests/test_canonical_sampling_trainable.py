from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from canonical_sampling import apply_phase_sampling_quota
from position_usage_db import ensure_schema
from streaming_db_loader import LabelsRepository, sample_epoch_keys


@pytest.fixture(autouse=True)
def _allow_streaming_loader_functional_tests(monkeypatch):
    monkeypatch.setenv("TRAINING_PREP_ONLY", "0")


def _mk_db(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    ensure_schema(con)
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS positions (
            pos_key TEXT PRIMARY KEY,
            position_data BLOB NOT NULL
        );
        CREATE TABLE IF NOT EXISTS labels (
            pos_key TEXT NOT NULL,
            source TEXT NOT NULL,
            value_stm REAL NOT NULL,
            n_samples INTEGER NOT NULL DEFAULT 1,
            PRIMARY KEY (pos_key, source)
        );
        """
    )
    return con


def _insert_json_position(
    con: sqlite3.Connection,
    pos_key: str,
    *,
    label_source: str,
    label_value: float = 0.5,
) -> None:
    rec = {
        "schema": "quoridor-position-v1",
        "pawns": [[4, 0], [4, 8]],
        "walls": [],
        "turn": 0,
        "ply": 10,
    }
    con.execute(
        "INSERT INTO positions(pos_key, position_data) VALUES (?, ?)",
        (pos_key, json.dumps(rec).encode("utf-8")),
    )
    con.execute(
        "INSERT INTO labels(pos_key, source, value_stm, n_samples) VALUES (?, ?, ?, 1)",
        (pos_key, label_source, label_value),
    )
    con.execute(
        """
        INSERT INTO position_usage(pos_key, first_seen_at, observation_count, source)
        VALUES (?, '2026-07-14T00:00:00+00:00', 1, 'canonical_json')
        """,
        (pos_key,),
    )
    con.commit()


def test_phase_quota_skips_excluded_outcome_only_positions(tmp_path: Path) -> None:
    db = tmp_path / "labels.db"
    con = _mk_db(db)
    _insert_json_position(con, "trainable_a", label_source="ka_nn", label_value=0.4)
    _insert_json_position(con, "trainable_b", label_source="ka_nn", label_value=-0.2)
    _insert_json_position(
        con,
        "bookkeeping_only",
        label_source="oracle_mixed_outcome",
        label_value=0.0,
    )
    con.close()

    con = sqlite3.connect(db)
    raw = sample_epoch_keys(con, epoch_size=3, seed=0, old_refresh_fraction=0.0)
    con.close()
    assert "json:bookkeeping_only" not in raw

    rebalanced = apply_phase_sampling_quota(raw * 40, db, seed=0)
    repo = LabelsRepository(db)
    rows = repo.load_labeled_positions(rebalanced)
    repo.close()
    assert len(rows) == len(rebalanced)
    assert all(row.position_id.startswith("json:trainable_") for row in rows)
