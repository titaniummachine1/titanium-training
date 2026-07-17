from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from pathlib import Path
from subprocess import CompletedProcess

import pytest

TRAINING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING))

import score_out_labels as collector  # noqa: E402
from label_resolution import stm_from_eval_cp  # noqa: E402
from titanium_training.store.state import PositionState  # noqa: E402


def _db(path: Path, *, canonical: bytes | None = None) -> tuple[bytes, int]:
    packed = PositionState.initial().packed_state()
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE positions(position_id INTEGER PRIMARY KEY, canonical_hash BLOB, packed_state BLOB, side_to_move INTEGER)"
    )
    con.execute("INSERT INTO positions VALUES(?,?,?,?)", (7, canonical or hashlib.sha256(packed).digest(), packed, 0))
    con.commit()
    con.close()
    return packed, 0


def _engine(tmp_path: Path) -> Path:
    path = tmp_path / "titanium"
    path.write_bytes(b"test engine")
    return path


def _score_out(*, stm: int, bound: str = "exact", score: int = 200, node_budget: int = 1234) -> dict:
    return {
        "schema": "score-out-v1",
        "ok": True,
        "input": "packed",
        "side_to_move": stm,
        "score": score,
        "bound": bound,
        "proven": False,
        "nodes": 100,
        "node_budget": node_budget,
        "depth": 9,
        "selected_move": "e2",
    }


def test_collects_deterministic_provenance_and_value(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    db = tmp_path / "positions.db"
    packed, stm = _db(db)
    engine = _engine(tmp_path)
    out = tmp_path / "labels.jsonl"
    calls: list[list[str]] = []

    def run(args, **kwargs):
        calls.append(args)
        return CompletedProcess(args, 0, json.dumps(_score_out(stm=stm)), "")

    monkeypatch.setattr(collector.subprocess, "run", run)
    summary = collector.collect_labels(db, out, node_budget=1234, engine_bin=engine)
    record = json.loads(out.read_text(encoding="utf-8"))
    assert summary["records"] == 1 and summary["skipped"] == 0
    assert calls == [[str(engine), "score-out", "--nodes", "1234", "--packed", packed.hex()]]
    assert record["format"] == "titanium-bounded-ab-value-label-v1"
    assert record["value_i16"] == collector.float_stm_to_value_i16(stm_from_eval_cp(200))
    assert record["value_i16"] != 100
    assert record["score_out"]["proven"] is False
    assert record["protocol"] == {"schema": "score-out-v1", "node_budget": 1234}
    assert record["source_db_sha256"] == hashlib.sha256(db.read_bytes()).hexdigest()
    assert record["engine_executable_sha256"] == hashlib.sha256(engine.read_bytes()).hexdigest()
    assert record["score_out"]["depth"] == 9


@pytest.mark.parametrize("kind", ["packed", "canonical"])
def test_rejects_corrupt_database_identity(tmp_path: Path, kind: str) -> None:
    db = tmp_path / "positions.db"
    packed, _stm = _db(db, canonical=b"x" * 32 if kind == "canonical" else None)
    if kind == "packed":
        con = sqlite3.connect(db)
        bad = bytearray(packed)
        bad[0] ^= 1
        con.execute("UPDATE positions SET packed_state=? WHERE position_id=7", (bytes(bad),))
        con.commit()
        con.close()
    with pytest.raises(ValueError, match="canonical_hash|packed_state"):
        collector.collect_labels(db, tmp_path / "labels.jsonl", engine_bin=_engine(tmp_path))


def test_skips_unknown_bound_and_refuses_overwrite(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    db = tmp_path / "positions.db"
    packed, stm = _db(db)
    engine = _engine(tmp_path)
    out = tmp_path / "labels.jsonl"
    monkeypatch.setattr(
        collector.subprocess, "run",
        lambda args, **kwargs: CompletedProcess(
            args, 0, json.dumps(_score_out(stm=stm, bound="lower", node_budget=200_000)), ""),
    )
    summary = collector.collect_labels(db, out, engine_bin=engine)
    assert summary["records"] == 0 and summary["skipped"] == 1
    out.write_text("keep\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="overwrite"):
        collector.collect_labels(db, out, engine_bin=engine)
    assert out.read_text(encoding="utf-8") == "keep\n"


def test_skips_trailing_score_out_frames(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    db = tmp_path / "positions.db"
    _packed, stm = _db(db)
    engine = _engine(tmp_path)
    out = tmp_path / "labels.jsonl"
    payload = json.dumps(_score_out(stm=stm, node_budget=200_000))
    monkeypatch.setattr(
        collector.subprocess, "run",
        lambda args, **kwargs: CompletedProcess(args, 0, payload + "\n" + payload, ""),
    )
    summary = collector.collect_labels(db, out, engine_bin=engine)
    assert summary["records"] == 0 and summary["skipped"] == 1


def test_engine_failure_leaves_no_output_or_partial(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    db = tmp_path / "positions.db"
    _db(db)
    engine = _engine(tmp_path)
    out = tmp_path / "labels.jsonl"
    monkeypatch.setattr(
        collector.subprocess, "run",
        lambda args, **kwargs: CompletedProcess(args, 2, "", "boom"),
    )
    with pytest.raises(RuntimeError, match="score-out failed"):
        collector.collect_labels(db, out, engine_bin=engine)
    assert not out.exists()
    assert not list(tmp_path.glob(".labels.jsonl.*.partial"))
