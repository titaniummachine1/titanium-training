"""Migration and legacy-reference tests for canonical datastore."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from titanium_training.store.config import CANONICAL_DB, DATA_DIR, LEGACY_GAME_DB
from titanium_training.store.guards import LegacyTrainingSourceError, assert_canonical_training_db, is_smoke_database
from titanium_training.store.lib import init_db, insert_game, semantic_checksum
from titanium_training.store.migration import audit_legacy_references, full_artifact_inventory, resolve_import_paths


def test_legacy_training_source_blocked() -> None:
    if not LEGACY_GAME_DB.exists():
        pytest.skip("legacy game db not present locally")
    with pytest.raises(LegacyTrainingSourceError):
        assert_canonical_training_db(LEGACY_GAME_DB)


def test_smoke_database_detected() -> None:
    assert is_smoke_database(DATA_DIR / "position_graph_smoke.db")


def test_full_inventory_has_dispositions() -> None:
    records = full_artifact_inventory()
    assert records
    for rec in records[:20]:
        assert rec.recommended_disposition
        assert rec.recommended_disposition != "unknown"


def test_resolve_import_paths_includes_core_sources() -> None:
    paths = resolve_import_paths(include_friend=False)
    rels = {str(p.relative_to(Path(__file__).resolve().parent.parent)).replace("\\", "/") for p in paths}
    assert "data/all_games.db" in rels


def test_semantic_checksum_stable(tmp_path: Path) -> None:
    db = tmp_path / "test.db"
    init_db(db)
    conn = sqlite3.connect(str(db))
    conn.executescript(
        "INSERT INTO positions(canonical_hash,fast_hash,packed_state,side_to_move,"
        "first_seen_at,last_seen_at,schema_version) "
        "VALUES(x'00',0,x'00',0,'t','t',1);"
    )
    conn.close()
    a = semantic_checksum(db)
    b = semantic_checksum(db)
    assert a == b


def test_discover_friend_shards() -> None:
    from titanium_training.store.friend import discover_friend_shards

    shards = discover_friend_shards()
    assert len(shards) == 20
    assert all(p.name == "shard_000.jsonl" for p in shards)
    assert shards[0].parent.name == "iter_000001"
    assert shards[-1].parent.name == "iter_000020"
