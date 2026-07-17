from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from external_sources.claustrophobia import crossplay_titanium_ladder as ladder
from external_sources.claustrophobia.verify_mining_resume_integrity import (
    EXPECTED_CHAMPION,
    verify,
)


def _row(gid, termination="goal"):
    return {
        "source_game_id": gid,
        "run_id": "run",
        "style": "epoch2",
        "moves": ["e2", "e8"],
        "termination": termination,
        "opening_id": "mine-open-0000",
        "titanium_first": True,
        "titanium_weights_sha256": "weights",
        "claustrophobia_checkpoint_sha256": EXPECTED_CHAMPION,
        "generation_config": {"sims": 2, "time_sec": 1.0},
    }


def _pilot(tmp_path: Path, rows):
    results = tmp_path / "results.jsonl"
    lines = [json.dumps(row, sort_keys=True) + "\n" for row in rows]
    results.write_text("".join(lines), encoding="utf-8")
    fp = {
        "source_game_ids": [rows[0]["source_game_id"]],
        "line_sha256": [
            hashlib.sha256(lines[0].encode()).hexdigest()
        ],
        "last_id": rows[0]["source_game_id"],
    }
    (tmp_path / "PRE_RESET_FINGERPRINT.json").write_text(
        json.dumps(fp), encoding="utf-8"
    )
    return tmp_path


def test_duplicate_id_detection(tmp_path):
    gid = "run:epoch2:0000"
    report = verify(_pilot(tmp_path, [_row(gid), _row(gid)]))
    assert not report["accept"]
    assert not report["checks"]["unique_source_game_ids"]["pass"]


def test_fingerprint_mismatch_detection(tmp_path):
    pilot = _pilot(tmp_path, [_row("run:epoch2:0000")])
    path = pilot / "results.jsonl"
    path.write_text(json.dumps(_row("run:epoch2:0000", "complete")) + "\n")
    report = verify(pilot)
    assert not report["accept"]
    assert not report["checks"]["pre_reset_fingerprint"]["pass"]


def test_infrastructure_is_not_protocol_classification():
    assert ladder.InfrastructureError.classification == "infrastructure_error"
    assert ladder.InfrastructureError("reset").classification != "protocol_error"


def test_retry_discards_partial_game(monkeypatch):
    calls = []
    successful = {"termination": "goal", "moves": ["e2", "e8"], "actions": ["e2", "e8"]}

    def fake_play_one(**kwargs):
        calls.append(True)
        if len(calls) == 1:
            raise ladder.InfrastructureError("ConnectionResetError")
        return successful

    monkeypatch.setattr(ladder, "play_one", fake_play_one)
    result = ladder.play_one_with_retry(max_game_retries=1, opening=(), ti=None,
                                        titanium_first=True, sims=2, device="cpu")
    assert result is successful
    assert len(calls) == 2
    assert result["moves"] == ["e2", "e8"]
