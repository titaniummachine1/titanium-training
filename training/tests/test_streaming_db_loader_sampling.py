from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from position_usage_db import claim_all_pending, ensure_schema, increment_new_eligible
from streaming_db_loader import sample_epoch_keys


@pytest.fixture(autouse=True)
def _allow_streaming_loader_functional_tests(monkeypatch):
    monkeypatch.setenv("TRAINING_PREP_ONLY", "0")


def _mk_usage_db(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    ensure_schema(con)
    return con


def _insert_usage_rows(
    con: sqlite3.Connection,
    *,
    start: int,
    count: int,
    visits: int,
    retired: int = 0,
    source: str = "canonical_json",
) -> None:
    rows = [
        (
            f"k{i:06d}",
            visits,
            retired,
            0,
            source,
            None,
            None,
            "2026-07-09T00:00:00+00:00",
            1,
        )
        for i in range(start, start + count)
    ]
    con.executemany(
        """
        INSERT INTO position_usage(
            pos_key, training_visits, retired, protected_replay, source,
            retirement_reason, last_trained_at, first_seen_at, observation_count
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    con.commit()


def test_large_exclusion_set_does_not_hit_sqlite_variable_limit(tmp_path: Path) -> None:
    con = _mk_usage_db(tmp_path / "labels.db")
    # 39,900 fresh + 200 old gives a large seen-set on fill path.
    _insert_usage_rows(con, start=0, count=39_900, visits=0)
    _insert_usage_rows(con, start=39_900, count=200, visits=1)

    keys = sample_epoch_keys(con, epoch_size=40_000, seed=7, old_refresh_fraction=0.0)
    assert len(keys) == 40_000
    assert len(set(keys)) == len(keys)
    assert all(k.startswith("json:") or k.startswith("teacher:") for k in keys)

    keys_repeat = sample_epoch_keys(con, epoch_size=40_000, seed=7, old_refresh_fraction=0.0)
    assert keys == keys_repeat
    con.close()


def test_empty_exclusions_and_nearly_exhausted_pool(tmp_path: Path) -> None:
    con = _mk_usage_db(tmp_path / "labels.db")
    _insert_usage_rows(con, start=0, count=25, visits=1)

    # new_keys is empty, so exclusion table path should handle empty exclusions.
    keys = sample_epoch_keys(con, epoch_size=20, seed=3, old_refresh_fraction=0.0)
    assert len(keys) == 20
    assert len(set(keys)) == len(keys)

    # Ask for more than available active rows; should return only available.
    almost_all = sample_epoch_keys(con, epoch_size=200, seed=3, old_refresh_fraction=0.0)
    assert len(almost_all) == 25
    assert len(set(almost_all)) == len(almost_all)
    con.close()


def test_claim_all_pending_is_single_consumer(tmp_path: Path) -> None:
    db = tmp_path / "labels.db"
    con1 = _mk_usage_db(db)
    increment_new_eligible(con1, 1234)
    con1.commit()

    con2 = sqlite3.connect(db)
    ensure_schema(con2)

    first = claim_all_pending(con1)
    second = claim_all_pending(con2)

    assert first["claimed"] is True
    assert first["claimed_count"] == 1234
    assert first["remaining"] == 0
    assert second["claimed"] is False
    assert second["claimed_count"] == 0
    assert second["remaining"] == 0

    con1.close()
    con2.close()
