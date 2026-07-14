from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from game_opening_gate import (
    DEPLOY_COLLAPSE_OPENING,
    TEMPORARY_GARBAGE_FILTER_NOT_DIVERSITY_COMPLIANCE,
    training_opening_ok,
)
from streaming_db_loader import db_counts, sample_epoch_keys

_TRAINING = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _allow_streaming_loader_functional_tests(monkeypatch):
    monkeypatch.setenv("TRAINING_PREP_ONLY", "0")


def _init_labels(path: Path) -> None:
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE positions (
            pos_key TEXT PRIMARY KEY,
            position_data BLOB NOT NULL,
            side_to_move INTEGER NOT NULL
        );
        CREATE TABLE labels (
            pos_key TEXT NOT NULL,
            source TEXT NOT NULL,
            value_stm REAL NOT NULL,
            n_samples INTEGER NOT NULL DEFAULT 1,
            PRIMARY KEY(pos_key, source)
        );
        CREATE TABLE teacher_labels (
            position_key BLOB NOT NULL,
            value_i16 INTEGER,
            source_cohort TEXT
        );
        """
    )
    for key in [f"good{i}" for i in range(4)] + [f"div{i}" for i in range(4)] + [f"bad{i}" for i in range(4)]:
        con.execute("INSERT INTO positions VALUES (?, ?, 0)", (key, b"{}",))
        con.execute("INSERT INTO labels VALUES (?, 'oracle_outcome', 1.0, 1)", (key,))
    con.commit()
    con.close()


def _init_games(path: Path) -> None:
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE games (
            game_id TEXT PRIMARY KEY,
            source TEXT,
            outcome_p0 INT,
            move_count INT,
            imported_at TEXT
        );
        CREATE TABLE game_moves (
            game_id TEXT,
            move_num INT,
            pos_key TEXT,
            move_alg TEXT,
            PRIMARY KEY(game_id, move_num)
        );
        """
    )
    con.execute("INSERT INTO games VALUES ('good', 'oracle', 1, 4, '2026-06-29T00:00:00Z')")
    con.execute("INSERT INTO games VALUES ('bad', 'oracle', 1, 4, '2026-06-29T00:00:00Z')")
    con.execute("INSERT INTO games VALUES ('diverse', 'oracle', 1, 4, '2026-06-29T00:00:00Z')")
    # Deploy collapse trunk — must pass 2-ply gate but is not required as plies 3–4.
    for i, move in enumerate(("e2", "e8", "e3", "e7")):
        con.execute("INSERT INTO game_moves VALUES ('good', ?, ?, ?)", (i, f"good{i}", move))
    # Same 2-ply gate, different plies 3–4 (must not be rejected).
    for i, move in enumerate(("e2", "e8", "d2", "f8")):
        con.execute("INSERT INTO game_moves VALUES ('diverse', ?, ?, ?)", (i, f"div{i}", move))
    for i, move in enumerate(("a7h", "d8h", "d3v", "a2h")):
        con.execute("INSERT INTO game_moves VALUES ('bad', ?, ?, ?)", (i, f"bad{i}", move))
    con.commit()
    con.close()


def test_temporary_two_ply_gate_not_diversity_compliance():
    assert TEMPORARY_GARBAGE_FILTER_NOT_DIVERSITY_COMPLIANCE is True
    assert training_opening_ok(["e2", "e8", "d2", "f8"])
    assert training_opening_ok(["e2", "e8", "e3", "e7"])
    assert DEPLOY_COLLAPSE_OPENING == ("e2", "e8", "e3", "e7")


def test_streaming_loader_has_no_four_ply_sql_trunk():
    src = (_TRAINING / "streaming_db_loader.py").read_text(encoding="utf-8")
    assert "OPENING_SANITY_PREFIX" not in src
    assert "move_num BETWEEN 0 AND 3" not in src
    assert "COUNT(DISTINCT move_num) = 4" not in src
    assert "TEMPORARY_GARBAGE_FILTER_NOT_DIVERSITY_COMPLIANCE" in src
    assert "WHITE_OPENING_PAWNS" in src
    assert "BLACK_OPENING_PAWNS" in src


def test_streaming_sampler_filters_collapsed_openings(tmp_path: Path) -> None:
    labels_db = tmp_path / "labels.db"
    games_db = tmp_path / "games.db"
    _init_labels(labels_db)
    _init_games(games_db)

    counts = db_counts(labels_db)
    assert counts.labeled_positions == 8
    assert counts.eligible_positions == 8

    con = sqlite3.connect(labels_db)
    try:
        selected = sample_epoch_keys(con, epoch_size=8, seed=0)
    finally:
        con.close()

    assert selected
    assert all(key.startswith("json:good") or key.startswith("json:div") for key in selected)
    assert not any(key.startswith("json:bad") for key in selected)
    # Diverse plies 3–4 must survive; only wall-first garbage is rejected.
    assert any(key.startswith("json:div") for key in selected)
