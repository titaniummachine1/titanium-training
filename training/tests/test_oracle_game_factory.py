from __future__ import annotations

import gzip
import json
import shutil
import sqlite3
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

from oracle_game_factory.generation import GenerationStore
from oracle_game_factory.protocol import atomic_write_json, game_payload_checksum
from oracle_game_factory.schedule import (
    CURRENT_CURRENT,
    CURRENT_PRIOR_P0,
    PRIOR_CURRENT_P0,
    make_schedule,
    schedule_counts,
)
from oracle_game_factory.spool import DurableSpool, SpoolConfig
from oracle_laptop_client import import_remote_game


def sample_payload(game_id: str = "oracle-test-1") -> dict:
    payload = {
        "game_id": game_id,
        "protocol_version": "titanium-oracle-game-factory/1",
        "schema_version": "titanium-oracle-game/1",
        "engine_build_hash": "engine",
        "current_weight_hash": "cur",
        "prior_weight_hash": "prior",
        "side_weight_hashes": {"p0": "cur", "p1": "prior"},
        "generation_id": "gen-1",
        "matchup_type": CURRENT_PRIOR_P0,
        "worker_id": 0,
        "seed": 123,
        "moves": ["e2e3", "e8e7"],
        "result": "DRAW",
        "termination_reason": "max_plies",
        "plies": 2,
        "time_control": {"move_time_sec": 0.1},
        "search": {"engine": "titanium-v15"},
        "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T00:00:01+00:00",
        "stats": {},
    }
    payload["payload_checksum"] = game_payload_checksum(payload)
    return payload


def test_exact_schedule_mixture_and_unique_seeds():
    sched = make_schedule(generation_seed=7)
    assert len(sched) == 1024
    assert schedule_counts(sched) == {
        CURRENT_CURRENT: 717,
        CURRENT_PRIOR_P0: 154,
        PRIOR_CURRENT_P0: 153,
    }
    assert len({g.seed for g in sched}) == 1024


def test_no_distinct_prior_falls_back_to_current_current():
    sched = make_schedule(generation_seed=7, has_distinct_prior=False)
    assert schedule_counts(sched) == {CURRENT_CURRENT: 1024}


def test_spool_write_download_duplicate_ack(tmp_path: Path):
    spool = DurableSpool(SpoolConfig(tmp_path / "spool", warn_bytes=1_000_000, stop_bytes=2_000_000))
    payload = sample_payload()
    path = spool.write_game(payload)
    assert path.is_file()
    assert spool.list_ready()[0]["game_id"] == payload["game_id"]
    downloaded = json.loads(gzip.decompress(spool.read_game_bytes(payload["game_id"])).decode("utf-8"))
    assert downloaded["game_id"] == payload["game_id"]
    assert spool.ack(payload["game_id"])["status"] == "acknowledged"
    assert spool.ack(payload["game_id"])["status"] == "already_acknowledged"


def test_spool_backpressure(tmp_path: Path):
    spool = DurableSpool(SpoolConfig(tmp_path / "spool", warn_bytes=1, stop_bytes=1))
    spool.write_game(sample_payload())
    bp = spool.backpressure()
    assert bp["warn"] is True
    assert bp["stop"] is True


def test_generation_stage_activate_hashes(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "current.bin").write_bytes(b"current")
    (src / "prior.bin").write_bytes(b"prior")
    import hashlib

    atomic_write_json(
        src / "generation.json",
        {
            "generation_id": "gen-1",
            "current_deployed_hash": hashlib.sha256(b"current").hexdigest(),
            "prior_deployed_hash": hashlib.sha256(b"prior").hexdigest(),
        },
    )
    store = GenerationStore(tmp_path / "gens")
    assert store.stage(src)["staged"] == "gen-1"
    assert store.activate("gen-1")["activated"] == "gen-1"
    assert store.active_manifest()["generation_id"] == "gen-1"


def test_oracle_result_not_acked_until_import_success():
    with patch("oracle_laptop_client._game_exists", return_value=False), patch(
        "oracle_laptop_client._line_hash_exists", return_value=False
    ), patch(
        "oracle_laptop_client.open_db", return_value=MagicMock()
    ), patch("oracle_laptop_client.write_batch", return_value=(1, 5, 5)), patch(
        "oracle_laptop_client.load_synced_ids", return_value=set()
    ), patch(
        "oracle_laptop_client.sync_single_game", side_effect=RuntimeError("teacher failed")
    ):
        try:
            import_remote_game(sample_payload("oracle-fail"))
            assert False, "expected teacher failure"
        except RuntimeError:
            pass


def test_duplicate_line_hash_is_acked_without_reimport_or_teacher_sync():
    """Same move sequence already stored under a different game_id (e.g. a
    deterministic current-vs-previous game replayed) must be treated as a
    safe skip, not an error -- and must NOT touch teacher sync, since
    sync_single_game() looks the game up by this game_id, which is
    deliberately never inserted for a duplicate."""
    with patch("oracle_laptop_client._game_exists", return_value=False), patch(
        "oracle_laptop_client._line_hash_exists", return_value=True
    ), patch("oracle_laptop_client.write_batch") as write_batch, patch(
        "oracle_laptop_client.sync_single_game"
    ) as sync_single_game:
        result = import_remote_game(sample_payload("oracle-duplicate"))
    write_batch.assert_not_called()
    sync_single_game.assert_not_called()
    assert result["synced"]["duplicate"] is True


def test_db_success_then_teacher_retry_completes_missing_stage():
    with patch("oracle_laptop_client._game_exists", return_value=True), patch(
        "oracle_laptop_client.load_synced_ids", return_value=set()
    ), patch(
        "oracle_laptop_client.sync_single_game", return_value={"new_positions": 3, "counted": True}
    ) as sync:
        result = import_remote_game(sample_payload("oracle-retry"))
    assert result["synced"]["new_positions"] == 3
    sync.assert_called_once()


def test_partial_download_payload_can_be_quarantined(tmp_path: Path):
    bad = tmp_path / "bad.json.gz"
    bad.write_bytes(b"not gzip")
    try:
        gzip.decompress(bad.read_bytes())
        assert False
    except Exception:
        quarantine = tmp_path / "quarantine"
        quarantine.mkdir()
        shutil.move(str(bad), quarantine / bad.name)
    assert (tmp_path / "quarantine" / "bad.json.gz").is_file()

