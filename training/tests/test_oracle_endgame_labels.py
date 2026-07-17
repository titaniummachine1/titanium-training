from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import pytest

TRAINING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING))

from oracle_endgame_labels import generate_labels  # noqa: E402
from titanium_training.data.hands_empty_oracle import HandsEmptyOracleResult  # noqa: E402
from titanium_training.store.state import PositionState  # noqa: E402


def _make_db(path: Path) -> tuple[bytes, bytes]:
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE positions(position_id INTEGER PRIMARY KEY, canonical_hash BLOB, packed_state BLOB, side_to_move INTEGER)")
    empty = PositionState(player0_walls=0, player1_walls=0).packed_state()
    walls = PositionState(player0_walls=1, player1_walls=0).packed_state()
    con.execute("INSERT INTO positions VALUES(1, ?, ?, 0)", (hashlib.sha256(empty).digest(), empty))
    con.execute("INSERT INTO positions VALUES(2, ?, ?, 0)", (hashlib.sha256(walls).digest(), walls))
    con.commit()
    con.close()
    return empty, walls


def test_generate_labels_filters_hands_empty_and_keeps_source_read_only(tmp_path: Path):
    db = tmp_path / "canonical.db"
    empty, _walls = _make_db(db)
    before = db.read_bytes()
    out = tmp_path / "labels.jsonl"
    seen = []

    def oracle(items, *, timeout_sec=None):
        seen.append(items)
        return [HandsEmptyOracleResult(row, True, 5, 1, 5) for row, _packed in items]

    summary = generate_labels(db, out, batch_size=1, oracle=oracle)
    assert db.read_bytes() == before
    assert summary["records"] == 1
    assert seen == [[(0, empty)]]
    record = json.loads(out.read_text(encoding="utf-8"))
    assert record["position_id"] == 1
    assert record["packed_state_hex"] == empty.hex()
    assert record["protocol"] == "hands-empty-oracle-packed-v1"
    assert record["value_stm"] == 1
    assert len(record["source_db_sha256"]) == 64


def test_generate_labels_is_create_only(tmp_path: Path):
    db = tmp_path / "canonical.db"
    _make_db(db)
    out = tmp_path / "labels.jsonl"
    out.write_text("prior artifact\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="immutable corpus"):
        generate_labels(db, out, oracle=lambda *_args, **_kwargs: [])
    assert out.read_text(encoding="utf-8") == "prior artifact\n"


def test_oracle_failure_does_not_publish_partial_output(tmp_path: Path):
    db = tmp_path / "canonical.db"
    _make_db(db)
    out = tmp_path / "labels.jsonl"
    with pytest.raises(RuntimeError, match="answer count mismatch"):
        generate_labels(db, out, oracle=lambda *_args, **_kwargs: [])
    assert not out.exists()
    assert not list(tmp_path.glob(".labels.jsonl.*.partial"))
