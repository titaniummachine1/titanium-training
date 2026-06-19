"""Production migration orchestration for the canonical position store."""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from position_store_config import (
    ARCHIVE,
    CANONICAL_DB,
    DATA_DIR,
    EXPORT_DIR,
    LEGACY_REFERENCE_ALLOW_PREFIXES,
    REPORT_DIR,
    ROOT,
    SMOKE_DIR,
)
from position_store_guards import LegacyTrainingSourceError
from position_store_lib import (
    ImportStats,
    db_summary,
    detect_import_format,
    graph_reachability_stats,
    import_path,
    init_db,
    init_production_metadata,
    inventory_scan,
    semantic_checksum,
    sha256_file,
    json_dumps,
)
from position_store_state import MOVE_SCHEMA_VERSION, POSITION_SCHEMA_VERSION

SCAN_SUFFIXES = {
    ".db", ".sqlite", ".sqlite3", ".jsonl", ".json", ".csv", ".bin", ".dat",
    ".npz", ".npy", ".pt", ".pth", ".pkl", ".gz", ".zst", ".zip", ".games",
}

DISPOSITIONS = frozenset({
    "MIGRATED",
    "REJECTED_WITH_REPORT",
    "QUARANTINED_UNKNOWN_SEMANTICS",
    "ARCHIVED_SOURCE",
    "OBSOLETE_REPRODUCIBLE",
    "ACTIVE_CANONICAL",
    "NON_TRAINING_ARTIFACT",
})

# Deterministic production import order (repository-relative paths or globs)
PRODUCTION_IMPORT_SOURCES: list[str] = [
    "training/data/all_games.db",
    "training/data/search_pressure.jsonl",
    "training/data/zero_teacher/labels/search_budget.jsonl",
    "training/data/lmr_phase3_smoke/natural.jsonl",
    # hard_negatives.jsonl omitted when empty — reconciled as zero-record excluded
]

# Optional friend shards (large); pass --include-friend-shards to migrate-production
FRIEND_SHARD_GLOB = "KaAiData/**/shard_000.jsonl"

INTENTIONALLY_EXCLUDED = {
    "training/data/smoke_test.jsonl": "reproducible smoke fixture",
    "training/data/smoke2.jsonl": "reproducible smoke fixture",
    "training/data/search_pressure_smoke.jsonl": "reproducible smoke fixture",
    "training/data/ka_teacher_cache.jsonl": "deprecated Ka API cache; semantics not engine centitempo",
    "training/data/benchmarks_log.jsonl": "benchmark telemetry, not training positions",
    "training/data/supervisor_alerts.jsonl": "operational alerts",
    "training/data/all_games.jsonl": "legacy export duplicate of all_games.db",
    "training/data/lmr_phase3_smoke/hard_negatives.jsonl": "empty placeholder file",
}


@dataclass
class ArtifactRecord:
    artifact_id: str
    path: str
    content_hash: str | None
    file_size: int
    format: str
    record_count: int | None
    semantic_type: str
    schema_version: str | None
    source_provenance: str
    active_consumers: list[str]
    migration_status: str
    canonical_destination: str
    parse_confidence: str
    label_semantics: str | None
    duplicate_relationship: str | None
    recommended_disposition: str
    notes: str = ""


@dataclass
class ReconciliationRow:
    source: str
    seen: int
    migrated: int
    duplicates: int
    rejected: int
    quarantined: int
    excluded: int
    unaccounted: int = 0


@dataclass
class MigrationRunResult:
    migration_run_id: str
    production_db: str
    archive_dir: str
    inventory: list[ArtifactRecord] = field(default_factory=list)
    reconciliation: list[ReconciliationRow] = field(default_factory=list)
    dry_run_stats: list[dict[str, Any]] = field(default_factory=list)
    import_stats: list[dict[str, Any]] = field(default_factory=list)
    idempotence: dict[str, Any] = field(default_factory=dict)
    rebuild: dict[str, Any] = field(default_factory=dict)
    audit: dict[str, Any] = field(default_factory=dict)
    export_smoke: dict[str, Any] = field(default_factory=dict)
    shard_smoke: dict[str, Any] = field(default_factory=dict)
    legacy_scan: dict[str, Any] = field(default_factory=dict)


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(ROOT),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


def engine_submodule_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(ROOT / "engine"),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


def _artifact_id(path: Path) -> str:
    return hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:16]


def _count_jsonl(path: Path) -> int:
    n = 0
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line in handle:
            if line.strip():
                n += 1
    return n


def _semantic_type_for(path: Path, fmt: str) -> str:
    if fmt == "sqlite-games-v1":
        return "historical_games"
    if fmt == "games-text-v1":
        return "historical_games"
    if fmt == "search-pressure-jsonl":
        return "search_pressure_label"
    if fmt == "zero-search-budget-jsonl":
        return "teacher_value_label"
    if fmt == "reduction-counterfactual-jsonl":
        return "lmr_counterfactual"
    if fmt == "alpha-selfplay-jsonl":
        return "friend_selfplay_position"
    if fmt == "benchmark-report":
        return "benchmark"
    if "checkpoint" in path.name or path.suffix in {".pt", ".pth"}:
        return "model_checkpoint"
    if path.suffix == ".json" and "manifest" in path.name:
        return "elo_manifest"
    return "unknown"


def _label_semantics(fmt: str) -> str | None:
    return {
        "search-pressure-jsonl": "normalized_search_pressure_scalar",
        "zero-search-budget-jsonl": "teacher_root_value_float",
        "reduction-counterfactual-jsonl": "lmr_activate_plus_one_binary",
        "alpha-selfplay-jsonl": "policy_probability+root_value",
    }.get(fmt)


def _consumers_for(path: Path) -> list[str]:
    rel = str(path.relative_to(ROOT)).replace("\\", "/")
    hits: list[str] = []
    patterns = {
        "all_games.db": ["train.py", "datagen.py", "coordinator.py", "run_swiss_overnight.py"],
        "search_pressure.jsonl": ["train_search_importance.py", "collect_search_importance.py"],
        "position_graph": ["position_store.py"],
    }
    for key, scripts in patterns.items():
        if key in rel:
            hits.extend(scripts)
    return hits


def _disposition_for(path: Path, fmt: str, rel: str) -> tuple[str, str, str]:
    if rel.replace("\\", "/") == str(CANONICAL_DB.relative_to(ROOT)).replace("\\", "/"):
        return "ACTIVE_CANONICAL", "production", "high"
    if rel in INTENTIONALLY_EXCLUDED or rel.replace("\\", "/") in INTENTIONALLY_EXCLUDED:
        reason = INTENTIONALLY_EXCLUDED.get(rel.replace("\\", "/"), "excluded")
        return "OBSOLETE_REPRODUCIBLE", "none", reason
    if "smoke" in path.name.lower() or "smoke" in rel.lower():
        return "OBSOLETE_REPRODUCIBLE", "smoke/", "reproducible test artifact"
    if fmt in {
        "sqlite-games-v1",
        "games-text-v1",
        "search-pressure-jsonl",
        "zero-search-budget-jsonl",
        "reduction-counterfactual-jsonl",
        "alpha-selfplay-jsonl",
    }:
        return "MIGRATED", str(CANONICAL_DB), "high"
    if fmt == "ka-cache-jsonl":
        return "QUARANTINED_UNKNOWN_SEMANTICS", "quarantine/", "low"
    if fmt == "jsonl-unknown":
        return "QUARANTINED_UNKNOWN_SEMANTICS", "quarantine/", "medium"
    if fmt == "benchmark-report":
        return "NON_TRAINING_ARTIFACT", "none", "high"
    if path.suffix in {".pt", ".pth", ".npz", ".npy"}:
        return "NON_TRAINING_ARTIFACT", "checkpoints/", "high"
    if fmt == "sqlite-unknown" and "position" in path.name:
        return "OBSOLETE_REPRODUCIBLE", "smoke/", "medium"
    return "ARCHIVED_SOURCE", "archive/", "medium"


def full_artifact_inventory() -> list[ArtifactRecord]:
    records: list[ArtifactRecord] = []
    scan_roots = [
        ROOT / "training",
        ROOT / "KaAiData",
        ROOT / "site" / "benchmark",
    ]
    seen: set[str] = set()
    for base in scan_roots:
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in SCAN_SUFFIXES and path.suffix.lower() not in {".games"}:
                continue
            rel = str(path.relative_to(ROOT)).replace("\\", "/")
            if rel in seen:
                continue
            seen.add(rel)
            size = path.stat().st_size
            content_hash = sha256_file(path) if size < 500_000_000 else None
            fmt = "unknown"
            record_count = None
            schema_version = None
            try:
                if path.suffix.lower() == ".jsonl":
                    from position_store_lib import classify_jsonl_object, jsonl_first_object

                    obj = jsonl_first_object(path)
                    if obj:
                        fmt, _, _ = classify_jsonl_object(obj)
                        schema_version = str(obj.get("schema", ""))
                    record_count = _count_jsonl(path)
                elif path.suffix.lower() == ".db":
                    fmt = detect_import_format(path) if path.stat().st_size > 0 else "empty-db"
                    if fmt == "sqlite-games-v1":
                        import sqlite3

                        conn = sqlite3.connect(str(path))
                        record_count = int(conn.execute("SELECT COUNT(*) FROM games").fetchone()[0])
                        conn.close()
                elif path.suffix.lower() == ".games":
                    from position_store_lib import parse_games_text

                    fmt = "games-text-v1"
                    record_count = len(parse_games_text(path.read_text(encoding="utf-8", errors="replace")))
                else:
                    fmt = path.suffix.lower().lstrip(".")
            except Exception as exc:
                fmt = f"unparseable:{exc.__class__.__name__}"
            disposition, dest, confidence = _disposition_for(path, fmt, rel)
            sem = _semantic_type_for(path, fmt)
            records.append(
                ArtifactRecord(
                    artifact_id=_artifact_id(path),
                    path=rel,
                    content_hash=content_hash,
                    file_size=size,
                    format=fmt,
                    record_count=record_count,
                    semantic_type=sem,
                    schema_version=schema_version,
                    source_provenance=rel.split("/")[0] if "/" in rel else "training",
                    active_consumers=_consumers_for(path),
                    migration_status="pending" if disposition == "MIGRATED" else disposition.lower(),
                    canonical_destination=dest,
                    parse_confidence=confidence,
                    label_semantics=_label_semantics(fmt),
                    duplicate_relationship="superset_of_all_games.db" if ".games" in rel and "tournament" in rel else None,
                    recommended_disposition=disposition,
                )
            )
    records.sort(key=lambda r: r.path)
    return records


def resolve_import_paths(*, include_friend: bool = False) -> list[Path]:
    paths: list[Path] = []
    for rel in PRODUCTION_IMPORT_SOURCES:
        p = ROOT / rel
        if p.exists():
            paths.append(p)
    if include_friend:
        for p in sorted(ROOT.glob(FRIEND_SHARD_GLOB)):
            if p.is_file() and p.stat().st_size > 0 and p not in paths:
                paths.append(p)
    return paths


def create_archive_manifest(
    migration_run_id: str,
    inventory: list[ArtifactRecord],
) -> Path:
    archive_dir = ARCHIVE / migration_run_id
    archive_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "migration_run_id": migration_run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": git_commit(),
        "engine_commit": engine_submodule_commit(),
        "artifacts": [asdict(a) for a in inventory],
        "import_sources": [
            str(p.relative_to(ROOT)).replace("\\", "/") for p in resolve_import_paths(include_friend=False)
        ],
    }
    manifest_path = archive_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    checksums: list[str] = []
    for art in inventory:
        if art.content_hash:
            checksums.append(f"{art.content_hash}  {art.path}")
    (archive_dir / "checksums.sha256").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    reports = archive_dir / "reports"
    reports.mkdir(exist_ok=True)
    return archive_dir


def _reconcile_row(source: str, stats: ImportStats) -> ReconciliationRow:
    seen = stats.record_count
    migrated = stats.accepted_count
    rejected = stats.rejected_count
    duplicates = stats.duplicate_count
    unaccounted = seen - migrated - rejected - duplicates
    return ReconciliationRow(
        source=source,
        seen=seen,
        migrated=migrated,
        duplicates=duplicates,
        rejected=rejected,
        quarantined=0,
        excluded=0,
        unaccounted=unaccounted,
    )


def run_dry_migration(db_path: Path, *, include_friend: bool = False) -> list[ImportStats]:
    stats_list: list[ImportStats] = []
    for source in resolve_import_paths(include_friend=include_friend):
        rel = str(source.relative_to(ROOT)).replace("\\", "/")
        if rel in INTENTIONALLY_EXCLUDED or source.stat().st_size == 0:
            continue
        try:
            fmt = detect_import_format(source)
        except ValueError:
            continue
        if fmt in {"jsonl-unparseable", "empty-db"}:
            continue
        stats_list.append(import_path(db_path, source, dry_run=True, report_dir=REPORT_DIR))
    return stats_list


def run_production_migration(
    *,
    migration_run_id: str | None = None,
    skip_friend_shards: bool = False,
) -> MigrationRunResult:
    run_id = migration_run_id or utc_stamp()
    inventory = full_artifact_inventory()
    archive_dir = create_archive_manifest(run_id, inventory)
    manifest_hash = sha256_file(archive_dir / "manifest.json")

    db_path = CANONICAL_DB
    if db_path.exists():
        db_path.unlink()
    init_db(db_path)

    dry_stats = run_dry_migration(db_path, include_friend=not skip_friend_shards)
    result = MigrationRunResult(
        migration_run_id=run_id,
        production_db=str(db_path),
        archive_dir=str(archive_dir),
        inventory=inventory,
        dry_run_stats=[s.__dict__ for s in dry_stats],
    )

    import_stats: list[ImportStats] = []
    reconciliation: list[ReconciliationRow] = []
    for source in resolve_import_paths(include_friend=not skip_friend_shards):
        rel = str(source.relative_to(ROOT)).replace("\\", "/")
        if rel in INTENTIONALLY_EXCLUDED:
            reconciliation.append(
                ReconciliationRow(
                    source=rel,
                    seen=0,
                    migrated=0,
                    duplicates=0,
                    rejected=0,
                    quarantined=0,
                    excluded=0,
                )
            )
            continue
        if source.stat().st_size == 0:
            reconciliation.append(
                ReconciliationRow(
                    source=rel,
                    seen=0,
                    migrated=0,
                    duplicates=0,
                    rejected=0,
                    quarantined=0,
                    excluded=0,
                )
            )
            continue
        try:
            fmt = detect_import_format(source)
        except ValueError:
            reconciliation.append(
                ReconciliationRow(
                    source=rel,
                    seen=0,
                    migrated=0,
                    duplicates=0,
                    rejected=0,
                    quarantined=1,
                    excluded=0,
                )
            )
            continue
        if fmt in {"jsonl-unparseable", "empty-db"}:
            reconciliation.append(
                ReconciliationRow(
                    source=rel,
                    seen=0,
                    migrated=0,
                    duplicates=0,
                    rejected=0,
                    quarantined=1,
                    excluded=0,
                )
            )
            continue
        if skip_friend_shards and "KaAiData" in rel:
            n = _count_jsonl(source) if source.suffix == ".jsonl" else 0
            reconciliation.append(
                ReconciliationRow(
                    source=rel,
                    seen=n,
                    migrated=0,
                    duplicates=0,
                    rejected=0,
                    quarantined=0,
                    excluded=n,
                )
            )
            continue
        try:
            stats = import_path(db_path, source, dry_run=False, report_dir=REPORT_DIR)
            import_stats.append(stats)
            reconciliation.append(_reconcile_row(rel, stats))
        except FileExistsError:
            reconciliation.append(
                ReconciliationRow(source=rel, seen=0, migrated=0, duplicates=1, rejected=0, quarantined=0, excluded=0)
            )

    init_production_metadata(
        db_path,
        migration_run_id=run_id,
        source_manifest_hash=manifest_hash,
        code_commit=git_commit(),
        engine_commit=engine_submodule_commit(),
    )

    result.import_stats = [s.__dict__ for s in import_stats]
    result.reconciliation = reconciliation
    result.audit = run_full_audit(db_path)
    (archive_dir / "reports" / "migration_result.json").write_text(
        json.dumps(
            {
                "reconciliation": [asdict(r) for r in reconciliation],
                "summary": db_summary(db_path),
                "graph": graph_reachability_stats(db_path),
                "semantic_checksum": semantic_checksum(db_path),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return result


def run_full_audit(db_path: Path = CANONICAL_DB) -> dict[str, Any]:
    from position_store_lib import audit_database

    return audit_database(db_path)


def prove_idempotence(db_path: Path = CANONICAL_DB) -> dict[str, Any]:
    before = db_summary(db_path)
    before_checksum = semantic_checksum(db_path)
    noop_sources: list[str] = []
    errors: list[str] = []
    for source in resolve_import_paths(include_friend=False):
        rel = str(source.relative_to(ROOT)).replace("\\", "/")
        if rel in INTENTIONALLY_EXCLUDED:
            continue
        try:
            import_path(db_path, source, dry_run=False, report_dir=REPORT_DIR)
            errors.append(f"unexpected re-import success: {rel}")
        except FileExistsError:
            noop_sources.append(rel)
        except Exception as exc:
            errors.append(f"{rel}: {exc}")
    after = db_summary(db_path)
    after_checksum = semantic_checksum(db_path)
    return {
        "before": before,
        "after": after,
        "before_checksum": before_checksum,
        "after_checksum": after_checksum,
        "checksums_unchanged": before_checksum == after_checksum,
        "counts_unchanged": before == after,
        "noop_sources": noop_sources,
        "errors": errors,
        "passed": before_checksum == after_checksum and before == after and not errors,
    }


def prove_rebuild(migration_run_id: str) -> dict[str, Any]:
    original = CANONICAL_DB
    if not original.exists():
        return {"passed": False, "error": "no production db"}
    orig_summary = db_summary(original)
    orig_checksum = semantic_checksum(original)
    rebuild_path = SMOKE_DIR / f"rebuild_proof_{migration_run_id}.db"
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    if rebuild_path.exists():
        rebuild_path.unlink()
    # Rebuild into temp path
    import shutil

    saved = CANONICAL_DB
    try:
        # Temporarily redirect by copying migration logic
        from position_store_config import CANONICAL_DIR

        temp_canonical = rebuild_path
        if temp_canonical.exists():
            temp_canonical.unlink()
        init_db(temp_canonical)
        for source in resolve_import_paths(include_friend=False):
            rel = str(source.relative_to(ROOT)).replace("\\", "/")
            if rel in INTENTIONALLY_EXCLUDED:
                continue
            try:
                import_path(temp_canonical, source, dry_run=False, report_dir=REPORT_DIR)
            except FileExistsError:
                pass
        new_summary = db_summary(temp_canonical)
        new_checksum = semantic_checksum(temp_canonical)
    finally:
        pass
    return {
        "rebuild_path": str(rebuild_path),
        "original_summary": orig_summary,
        "rebuild_summary": new_summary,
        "original_checksum": orig_checksum,
        "rebuild_checksum": new_checksum,
        "semantic_match": orig_checksum == new_checksum,
        "passed": orig_checksum == new_checksum and orig_summary == new_summary,
    }


def export_training_smoke(db_path: Path = CANONICAL_DB) -> dict[str, Any]:
    from position_store_lib import export_training_rows

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = EXPORT_DIR / "training_export_smoke.jsonl"
    count = export_training_rows(db_path, out_path=out, label_type="teacher_value")
    unique_positions: set[str] = set()
    decode_failures = 0
    with out.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                obj = json.loads(line)
                unique_positions.add(obj["canonical_hash"])
            except Exception:
                decode_failures += 1
    return {
        "export_path": str(out),
        "row_count": count,
        "unique_positions": len(unique_positions),
        "decode_failures": decode_failures,
        "passed": decode_failures == 0 and count >= 0,
    }


def shard_ingestion_smoke(db_path: Path = CANONICAL_DB) -> dict[str, Any]:
    from position_store_lib import BinaryShardWriter, import_binary_shard

    inbox = ROOT / "training" / "data" / "selfplay_shards" / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    stem = f"migration_smoke_{utc_stamp()}"
    writer = BinaryShardWriter(
        inbox,
        engine_hash="smoke",
        trunk_hash="smoke",
        search_config_hash="smoke",
        worker_id="migration-smoke",
        random_seed_range="0-0",
    )
    writer.add_game(["e2", "e8"], result=1)
    shard = writer.write_ready(stem)
    before = db_summary(db_path)
    stats1 = import_binary_shard(db_path, shard, dry_run=False)
    after1 = db_summary(db_path)
    imported = shard.with_suffix(".imported")
    dup_ok = False
    if imported.exists():
        replay = inbox / f"{stem}_replay.ready"
        replay.write_bytes(imported.read_bytes())
        try:
            import_binary_shard(db_path, replay, dry_run=False)
        except FileExistsError:
            dup_ok = True
    # Corrupt shard
    bad_ready = inbox / f"bad_{utc_stamp()}.ready"
    bad = inbox / f"bad_{utc_stamp()}.partial"
    bad.write_bytes(b"NOTASHARD")
    if bad_ready.exists():
        bad_ready.unlink()
    bad.rename(bad_ready)
    corrupt_rejected = False
    try:
        import_binary_shard(db_path, bad_ready, dry_run=False)
    except (ValueError, Exception):
        corrupt_rejected = True
    return {
        "first_import": stats1.__dict__,
        "games_before": before["games"],
        "games_after": after1["games"],
        "duplicate_noop": dup_ok,
        "corrupt_rejected": corrupt_rejected,
        "passed": after1["games"] == before["games"] + 1 and dup_ok and corrupt_rejected,
    }


LEGACY_PATH_PATTERNS = [
    re.compile(r"training/data/all_games\.db"),
    re.compile(r"training/data/search_pressure\.jsonl"),
    re.compile(r"training/data/position_graph(_smoke)?\.db"),
    re.compile(r"training/data/smoke.*\.jsonl"),
]


def audit_legacy_references(scan_root: Path = ROOT / "training") -> dict[str, Any]:
    violations: list[dict[str, str]] = []
    allowed_files = {Path(p) for p in [
        "position_store_migration.py",
        "position_store_guards.py",
        "position_store_config.py",
        "position_store_lib.py",
        "position_store.py",
        "test_position_store.py",
        "CANONICAL_DATASTORE.md",
        "POSITION_STORE_RUNBOOK.md",
    ]}
    for path in scan_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in {".py", ".md", ".ps1"}:
            continue
        rel = str(path.relative_to(ROOT)).replace("\\", "/")
        allowed = False
        for prefix in LEGACY_REFERENCE_ALLOW_PREFIXES:
            if rel.startswith(prefix.replace("\\", "/")):
                allowed = True
                break
        if path.name in {f.name for f in allowed_files}:
            allowed = True
        text = path.read_text(encoding="utf-8", errors="replace")
        for pat in LEGACY_PATH_PATTERNS:
            if pat.search(text) and not allowed:
                if "LEGACY" in text[:500] or "legacy" in path.name:
                    continue
                if path.name in {"train.py"} and "assert_canonical" in text:
                    continue
                violations.append({"file": rel, "pattern": pat.pattern})
    return {
        "violations": violations,
        "passed": len(violations) == 0,
        "scanned_root": str(scan_root),
    }


def relocate_smoke_artifacts() -> dict[str, str]:
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    moved: dict[str, str] = {}
    for name in [
        "position_graph.db",
        "position_graph_smoke.db",
        "position_graph_compact_smoke.db",
        "position_graph_compact_smoke.bin",
        "position_store_smoke_export.jsonl",
    ]:
        src = DATA_DIR / name
        if src.exists():
            dst = SMOKE_DIR / name
            if dst.exists():
                dst.unlink()
            src.rename(dst)
            moved[str(src)] = str(dst)
    sidecar_src = DATA_DIR / "position_graph_compact_smoke.sidecars"
    if sidecar_src.exists():
        dst = SMOKE_DIR / "position_graph_compact_smoke.sidecars"
        if dst.exists():
            import shutil
            shutil.rmtree(dst)
        sidecar_src.rename(dst)
        moved[str(sidecar_src)] = str(dst)
    return moved
