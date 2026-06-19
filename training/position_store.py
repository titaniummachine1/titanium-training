#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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


def print_json(data: object) -> None:
    print(json.dumps(data, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Canonical Quoridor position store")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help=f"database path (default {DEFAULT_DB_PATH})")
    parser.add_argument("--reports", type=Path, default=DEFAULT_REPORT_DIR, help="report output directory")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="create the canonical position database schema")
    sub.add_parser("init-compact", help="create the compact v2 position database schema")
    sub.add_parser("inventory", help="scan repository data and emit inventory reports")

    import_all = sub.add_parser("import-all", help="import every known supported dataset discovered by inventory")
    import_all.add_argument("--dry-run", action="store_true", help="parse and validate without committing")

    import_games = sub.add_parser("import-games", help="import one supported game dataset")
    import_games.add_argument("path", type=Path)
    import_games.add_argument("--dry-run", action="store_true")

    import_positions = sub.add_parser("import-positions", help="import one supported position/label dataset")
    import_positions.add_argument("path", type=Path)
    import_positions.add_argument("--dry-run", action="store_true")

    ingest = sub.add_parser("ingest-shards", help="ingest .ready binary shards from a directory or one file")
    ingest.add_argument("path", type=Path)
    ingest.add_argument("--dry-run", action="store_true")

    sub.add_parser("audit", help="run SQLite + graph integrity checks")
    sub.add_parser("storage-audit", help="measure table/index/payload storage using SQLite page stats")
    sub.add_parser("stats", help="print a compact database summary")
    storage = sub.add_parser("storage-report", help="print measured storage usage and source-size comparisons")
    storage.add_argument("source", nargs="*", type=Path, help="optional source files to compare against")
    sub.add_parser("relabel-queue", help="show pending relabel work")
    sub.add_parser("score-semantics", help="print the compact label score/unit contract")

    rebuild = sub.add_parser("rebuild-compact", help="migrate an existing v1 position store into the compact v2 schema")
    rebuild.add_argument("src", type=Path)
    rebuild.add_argument("dst", type=Path)
    rebuild.add_argument("--sidecars", type=Path, default=None, help="directory for compact payload sidecars")

    export_train = sub.add_parser("export-training", help="export packed states with compatible labels")
    export_train.add_argument("out", type=Path)
    export_train.add_argument("--label-type", default="teacher_value")
    export_train.add_argument("--limit", type=int, default=None)

    export_train_bin = sub.add_parser("export-training-binary", help="export compact v2 labels as fixed-width binary records")
    export_train_bin.add_argument("out", type=Path)
    export_train_bin.add_argument("--limit", type=int, default=None)
    export_train_bin.add_argument("--label-type-code", type=int, default=2)
    return parser


def command_inventory(args: argparse.Namespace) -> int:
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
    print_json(audit_database(args.db))
    return 0


def command_stats(args: argparse.Namespace) -> int:
    print_json(db_summary(args.db))
    return 0


def command_storage_audit(args: argparse.Namespace) -> int:
    print_json(storage_audit(args.db))
    return 0


def command_storage_report(args: argparse.Namespace) -> int:
    print_json(storage_report(args.db, source_paths=args.source))
    return 0


def command_score_semantics(args: argparse.Namespace) -> int:
    print_json(score_semantics_report())
    return 0


def command_rebuild_compact(args: argparse.Namespace) -> int:
    print_json(rebuild_compact_db(args.src, args.dst, sidecar_dir=args.sidecars))
    return 0


def command_relabel_queue(args: argparse.Namespace) -> int:
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


def command_export_training(args: argparse.Namespace) -> int:
    count = export_training_rows(args.db, out_path=args.out, label_type=args.label_type, limit=args.limit)
    print(json.dumps({"wrote": str(args.out), "rows": count}, indent=2))
    return 0


def command_export_training_binary(args: argparse.Namespace) -> int:
    result = export_training_binary(args.db, out_path=args.out, limit=args.limit, label_type_code_filter=args.label_type_code)
    print_json(result)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "init":
        init_db(args.db)
        print(json.dumps({"initialized": str(args.db)}, indent=2))
        return 0
    if args.command == "init-compact":
        init_compact_db(args.db)
        print(json.dumps({"initialized_compact": str(args.db)}, indent=2))
        return 0
    if args.command == "inventory":
        return command_inventory(args)
    if args.command == "import-all":
        return command_import_all(args)
    if args.command in {"import-games", "import-positions"}:
        return command_import_one(args)
    if args.command == "ingest-shards":
        return command_ingest_shards(args)
    if args.command == "audit":
        return command_audit(args)
    if args.command == "storage-audit":
        return command_storage_audit(args)
    if args.command == "stats":
        return command_stats(args)
    if args.command == "storage-report":
        return command_storage_report(args)
    if args.command == "score-semantics":
        return command_score_semantics(args)
    if args.command == "rebuild-compact":
        return command_rebuild_compact(args)
    if args.command == "relabel-queue":
        return command_relabel_queue(args)
    if args.command == "export-training":
        return command_export_training(args)
    if args.command == "export-training-binary":
        return command_export_training_binary(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
