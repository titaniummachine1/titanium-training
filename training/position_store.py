#!/usr/bin/env python3
"""Canonical Quoridor position store CLI.

Production database default: training/data/canonical/position_store_v2.db
See training/CANONICAL_DATASTORE.md before using legacy import commands.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from position_store_config import CANONICAL_DB, SMOKE_DIR
from position_store_guards import assert_canonical_training_db, is_smoke_database
from position_store_lib import (
    DEFAULT_DB_PATH,
    DEFAULT_REPORT_DIR,
    audit_database,
    db_summary,
    export_training_rows,
    import_all_known,
    import_binary_shard,
    import_path,
    init_db,
    inventory_scan,
    storage_report,
    write_inventory_report,
)
from position_store_compact import (
    export_training_binary,
    init_compact_db,
    rebuild_compact_db,
    score_semantics_report,
    storage_audit,
)
from position_store_migration import (
    audit_legacy_references,
    export_training_smoke,
    full_artifact_inventory,
    prove_idempotence,
    prove_rebuild,
    relocate_smoke_artifacts,
    run_production_migration,
    shard_ingestion_smoke,
)


def print_json(data: object) -> None:
    print(json.dumps(data, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Canonical Quoridor position store (production DB: canonical/position_store_v2.db)",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"canonical production database (default {DEFAULT_DB_PATH})",
    )
    parser.add_argument("--reports", type=Path, default=DEFAULT_REPORT_DIR, help="report output directory")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="create the canonical production database schema")
    sub.add_parser("init-compact", help="[experimental] create compact v2 schema — not production canonical")
    inv = sub.add_parser("inventory", help="scan repository and emit inventory reports")
    inv.add_argument("--full", action="store_true", help="full artifact inventory with disposition")

    migrate = sub.add_parser(
        "migrate-production",
        help="run full production migration from archived sources into canonical DB",
    )
    migrate.add_argument("--dry-run-only", action="store_true", help="dry-run imports only, do not commit")
    migrate.add_argument("--include-friend-shards", action="store_true", help="import KaAiData friend JSONL shards")

    import_all = sub.add_parser("import-all-legacy", help="[LEGACY] import datasets discovered by inventory scan")
    import_all.add_argument("--dry-run", action="store_true")

    import_games = sub.add_parser("import-legacy-games", help="[LEGACY] import one game DB or .games file")
    import_games.add_argument("path", type=Path)
    import_games.add_argument("--dry-run", action="store_true")

    import_positions = sub.add_parser("import-legacy-positions", help="[LEGACY] import one position/label JSONL")
    import_positions.add_argument("path", type=Path)
    import_positions.add_argument("--dry-run", action="store_true")

    smoke = sub.add_parser("create-smoke-db", help="create an isolated smoke database under training/data/smoke/")
    smoke.add_argument("name", nargs="?", default="position_graph_smoke.db")

    ingest = sub.add_parser("ingest-shards", help="ingest .ready binary shards into the canonical production DB")
    ingest.add_argument("path", type=Path)
    ingest.add_argument("--dry-run", action="store_true")

    sub.add_parser("audit-canonical", help="full integrity + graph reachability audit of production DB")
    sub.add_parser("audit-legacy-references", help="scan active code for prohibited legacy data references")
    sub.add_parser("prove-idempotence", help="re-import all sources and verify no semantic change")
    prove_reb = sub.add_parser("prove-rebuild", help="rebuild disposable copy and compare semantic checksums")
    prove_reb.add_argument("--migration-run-id", required=True)
    sub.add_parser("storage-audit", help="measure table/index/payload storage using SQLite page stats")
    sub.add_parser("stats", help="print a compact database summary")
    storage = sub.add_parser("storage-report", help="print measured storage usage and source-size comparisons")
    storage.add_argument("source", nargs="*", type=Path)
    sub.add_parser("relabel-queue", help="show pending relabel work")
    sub.add_parser("score-semantics", help="[experimental] compact label score/unit contract")
    sub.add_parser("relocate-smoke-artifacts", help="move ambiguous smoke DB files into training/data/smoke/")
    sub.add_parser("export-training-smoke", help="export training rows and verify decode")

    rebuild = sub.add_parser("rebuild-compact", help="[experimental] migrate v1 store into compact v2 schema")
    rebuild.add_argument("src", type=Path)
    rebuild.add_argument("dst", type=Path)
    rebuild.add_argument("--sidecars", type=Path, default=None)

    export_train = sub.add_parser("export-training", help="export packed states with labels from canonical DB")
    export_train.add_argument("out", type=Path)
    export_train.add_argument("--label-type", default="teacher_value")
    export_train.add_argument("--limit", type=int, default=None)

    export_train_bin = sub.add_parser(
        "export-training-binary",
        help="[experimental] export compact v2 labels as fixed-width binary records",
    )
    export_train_bin.add_argument("out", type=Path)
    export_train_bin.add_argument("--limit", type=int, default=None)
    export_train_bin.add_argument("--label-type-code", type=int, default=2)

    shard_smoke = sub.add_parser("shard-ingestion-smoke", help="write/ingest/corrupt-test a binary shard")
    shard_smoke.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)

    # Back-compat aliases
    sub.add_parser("audit", help="alias for audit-canonical")
    ig = sub.add_parser("import-games", help="alias for import-legacy-games")
    ig.add_argument("path", type=Path)
    ig.add_argument("--dry-run", action="store_true")
    ip = sub.add_parser("import-positions", help="alias for import-legacy-positions")
    ip.add_argument("path", type=Path)
    ip.add_argument("--dry-run", action="store_true")
    ia = sub.add_parser("import-all", help="alias for import-all-legacy")
    ia.add_argument("--dry-run", action="store_true")
    return parser


def _require_production_db(db: Path) -> None:
    if is_smoke_database(db):
        raise SystemExit(f"Refusing smoke database for production command: {db}\nUse: {CANONICAL_DB}")


def command_inventory(args: argparse.Namespace) -> int:
    if args.full:
        from dataclasses import asdict
        from datetime import datetime, timezone

        records = full_artifact_inventory()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = args.reports / f"full-inventory-{stamp}.json"
        args.reports.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps([asdict(r) for r in records], indent=2) + "\n", encoding="utf-8")
        print(f"full inventory: {len(records)} artifacts -> {out}")
        return 0
    records = inventory_scan()
    json_path, md_path = write_inventory_report(records, args.reports)
    print(f"inventory records: {len(records)}")
    print(f"json: {json_path}")
    print(f"markdown: {md_path}")
    return 0


def command_import_all(args: argparse.Namespace) -> int:
    init_db(args.db)
    stats = import_all_known(args.db, dry_run=args.dry_run)
    print_json([s.__dict__ for s in stats])
    return 0


def command_import_one(args: argparse.Namespace) -> int:
    init_db(args.db)
    stats = import_path(args.db, args.path, dry_run=args.dry_run, report_dir=args.reports)
    print_json(stats.__dict__)
    return 0


def command_ingest_shards(args: argparse.Namespace) -> int:
    _require_production_db(args.db)
    init_db(args.db)
    stats = []
    if args.path.is_file():
        stats.append(import_binary_shard(args.db, args.path, dry_run=args.dry_run).__dict__)
    else:
        for shard in sorted(args.path.glob("*.ready")):
            stats.append(import_binary_shard(args.db, shard, dry_run=args.dry_run).__dict__)
    print_json(stats)
    return 0


def command_audit(args: argparse.Namespace) -> int:
    _require_production_db(args.db)
    print_json(audit_database(args.db))
    return 0


def command_migrate_production(args: argparse.Namespace) -> int:
    if args.dry_run_only:
        from position_store_migration import run_dry_migration

        init_db(CANONICAL_DB)
        stats = run_dry_migration(CANONICAL_DB)
        print_json([s.__dict__ for s in stats])
        return 0
    result = run_production_migration(skip_friend_shards=not args.include_friend_shards)
    print_json(
        {
            "migration_run_id": result.migration_run_id,
            "production_db": result.production_db,
            "archive_dir": result.archive_dir,
            "reconciliation": [__import__("dataclasses").asdict(r) for r in result.reconciliation],
            "summary": db_summary(Path(result.production_db)),
        }
    )
    unaccounted = sum(r.unaccounted for r in result.reconciliation)
    if unaccounted != 0:
        print(f"ERROR: unaccounted records = {unaccounted}", file=sys.stderr)
        return 1
    return 0


def command_create_smoke_db(args: argparse.Namespace) -> int:
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    path = SMOKE_DIR / args.name
    init_db(path)
    print_json({"smoke_db": str(path)})
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cmd = args.command
    if cmd == "init":
        _require_production_db(args.db)
        init_db(args.db)
        print(json.dumps({"initialized": str(args.db)}, indent=2))
        return 0
    if cmd == "init-compact":
        init_compact_db(args.db)
        print(json.dumps({"initialized_compact_experimental": str(args.db)}, indent=2))
        return 0
    if cmd == "inventory":
        return command_inventory(args)
    if cmd == "migrate-production":
        return command_migrate_production(args)
    if cmd in {"import-all-legacy", "import-all"}:
        return command_import_all(args)
    if cmd in {"import-legacy-games", "import-games"}:
        return command_import_one(args)
    if cmd in {"import-legacy-positions", "import-positions"}:
        return command_import_one(args)
    if cmd == "create-smoke-db":
        return command_create_smoke_db(args)
    if cmd == "ingest-shards":
        return command_ingest_shards(args)
    if cmd in {"audit-canonical", "audit"}:
        return command_audit(args)
    if cmd == "audit-legacy-references":
        result = audit_legacy_references()
        print_json(result)
        return 0 if result["passed"] else 1
    if cmd == "prove-idempotence":
        result = prove_idempotence(args.db)
        print_json(result)
        return 0 if result["passed"] else 1
    if cmd == "prove-rebuild":
        result = prove_rebuild(args.migration_run_id)
        print_json(result)
        return 0 if result["passed"] else 1
    if cmd == "relocate-smoke-artifacts":
        print_json(relocate_smoke_artifacts())
        return 0
    if cmd == "export-training-smoke":
        result = export_training_smoke(args.db)
        print_json(result)
        return 0 if result["passed"] else 1
    if cmd == "shard-ingestion-smoke":
        result = shard_ingestion_smoke(args.db)
        print_json(result)
        return 0 if result["passed"] else 1
    if cmd == "storage-audit":
        print_json(storage_audit(args.db))
        return 0
    if cmd == "stats":
        print_json(db_summary(args.db))
        return 0
    if cmd == "storage-report":
        print_json(storage_report(args.db, source_paths=args.source))
        return 0
    if cmd == "score-semantics":
        print_json(score_semantics_report())
        return 0
    if cmd == "rebuild-compact":
        print_json(rebuild_compact_db(args.src, args.dst, sidecar_dir=args.sidecars))
        return 0
    if cmd == "relabel-queue":
        import sqlite3

        conn = sqlite3.connect(str(args.db))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT queue_id, position_id, requested_label_type, requested_node_budget, priority, reason, status, attempt_count "
            "FROM relabel_queue ORDER BY status, priority DESC, created_at LIMIT 100"
        ).fetchall()
        conn.close()
        print_json([dict(row) for row in rows])
        return 0
    if cmd == "export-training":
        _require_production_db(args.db)
        count = export_training_rows(args.db, out_path=args.out, label_type=args.label_type, limit=args.limit)
        print(json.dumps({"wrote": str(args.out), "rows": count}, indent=2))
        return 0
    if cmd == "export-training-binary":
        result = export_training_binary(args.db, out_path=args.out, limit=args.limit, label_type_code_filter=args.label_type_code)
        print_json(result)
        return 0
    raise AssertionError(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
