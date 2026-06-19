"""Canonical position-store paths and schema identity.

All active training-data consumers should resolve paths through this module.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = ROOT / "training"
DATA_DIR = TRAINING_DIR / "data"
CANONICAL_DIR = DATA_DIR / "canonical"
ARCHIVE_DIR = DATA_DIR / "archive" / "legacy_sources"
SMOKE_DIR = DATA_DIR / "smoke"
REPORT_DIR = DATA_DIR / "position_store_reports"
EXPORT_DIR = DATA_DIR / "exports"
SHARD_INBOX = DATA_DIR / "selfplay_shards" / "inbox"
SHARD_PROCESSED = DATA_DIR / "selfplay_shards" / "processed"
SIDECAR_DIR = CANONICAL_DIR / "sidecars"

# Active production database (schema v1 graph store — not compact v2).
CANONICAL_DB = Path(
    os.environ.get(
        "TI_POSITION_STORE_DB",
        str(CANONICAL_DIR / "position_store_v2.db"),
    )
)

# Environment overrides
SIDECARS = Path(os.environ.get("TI_POSITION_STORE_SIDECARS", str(SIDECAR_DIR)))
SHARD_INBOX_PATH = Path(os.environ.get("TI_POSITION_STORE_SHARD_INBOX", str(SHARD_INBOX)))
EXPORTS = Path(os.environ.get("TI_POSITION_STORE_EXPORTS", str(EXPORT_DIR)))
ARCHIVE = Path(os.environ.get("TI_POSITION_STORE_ARCHIVE", str(ARCHIVE_DIR)))

# Schema versions (frozen for this migration)
DATABASE_SCHEMA_VERSION = 1
POSITION_SCHEMA_VERSION = 1
MOVE_SCHEMA_VERSION = 1
LABEL_SCHEMA_VERSION = 1
SHARD_SCHEMA_VERSION = 1

# Legacy paths — fail closed for active training unless explicitly allowed
LEGACY_GAME_DB = DATA_DIR / "all_games.db"
LEGACY_GAME_JSONL = DATA_DIR / "all_games.jsonl"
LEGACY_SEARCH_PRESSURE = DATA_DIR / "search_pressure.jsonl"
LEGACY_SMOKE_DBS = frozenset(
    {
        DATA_DIR / "position_graph.db",
        DATA_DIR / "position_graph_smoke.db",
        DATA_DIR / "position_graph_compact_smoke.db",
        DATA_DIR / "position_graph_compact_smoke.bin",
    }
)

LEGACY_TRAINING_SOURCES = frozenset(
    {
        LEGACY_GAME_DB,
        LEGACY_GAME_JSONL,
        LEGACY_SEARCH_PRESSURE,
        DATA_DIR / "search_pressure_smoke.jsonl",
        DATA_DIR / "smoke_test.jsonl",
        DATA_DIR / "smoke2.jsonl",
        DATA_DIR / "ka_teacher_cache.jsonl",
    }
)

# Paths allowed to reference legacy data (importers, migration, docs)
LEGACY_REFERENCE_ALLOW_PREFIXES = (
    "training/position_store",
    "training/test_position_store",
    "training/CANONICAL_DATASTORE",
    "training/POSITION_STORE_RUNBOOK",
    "training/data/archive/",
    "training/data/position_store_reports/",
    "training/collect_search_importance",
    "training/collect_reduction",
    "training/zero_teacher/collect",
    "training/zero_teacher/HANDOFF",
    "training/run_search_pressure_experiment",
    "training/coordinator",
    "training/datagen",
    "training/ingest_self_match",
    "training/import_clipboard",
    "training/verify_db_games",
    "training/ka_api_teacher",
    "training/AUDIT_REPORT",
    "training/watch_progress",
    "training/train_search_importance",
)

CANONICAL_EXPORT_COMMAND = (
    "python training/position_store.py export-training "
    "training/data/exports/training_export.jsonl --label-type teacher_value"
)
