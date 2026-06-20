#!/usr/bin/env python3
"""Canonical Quoridor position store CLI — separate game and teacher databases.

Game store default: training/data/canonical/game_store.db
Teacher store:      training/data/canonical/position_teacher_store.db
See training/CANONICAL_DATASTORE.md before using legacy import commands.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from position_store_config import (
    CANONICAL_DB,
    FRIEND_CORPUS_DIR,
    GAME_STORE_DB,
    REPORT_DIR,
    ROOT,
    RUST_IMPORTER_BIN,
    SMOKE_DIR,
    TEACHER_SIDECARS,
    TEACHER_STORE_DB,
)
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
from position_store_friend import (
    export_friend_training_smoke,
    import_friend_shards,
    inspect_all_friend_shards,
    prove_friend_idempotence,
    update_archive_manifest_friend_dispositions,
)
from position_store_teacher import (
    audit_game_store,
    audit_teacher_store,
    export_mixed_training,
    export_teacher_training,
    import_teacher_sources,
    init_game_store,
    init_teacher_store,
    prove_teacher_idempotence,
    teacher_semantic_checksum,
    verify_codec_parity,
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
from position_store_split import run_split_migration
from teacher_dataset.cli import (
    cmd_audit_position_parity,
    cmd_audit_teacher_dataset,
    cmd_benchmark_teacher_readers,
    cmd_build_teacher_dataset,
    cmd_finalize_teacher_candidate,
    cmd_freeze_teacher_reference,
    cmd_reconcile_teacher_source,
    cmd_repair_candidate_manifest,
    cmd_run_teacher_gate_audits,
    cmd_stats_teacher_dataset,
    cmd_verify_candidate,
    cmd_verify_teacher_policies,
)
from teacher_dataset.config import TEACHER_CATALOG_DB, TEACHER_DATASET_CANDIDATE_DIR


def print_json(data: object) -> None:
    print(json.dumps(data, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Quoridor position store — game_store.db (replayable games) + position_teacher_store.db (pathless labels)",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"game store database (default {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--teacher-db",
        type=Path,
        default=TEACHER_STORE_DB,
        help=f"teacher store database (default {TEACHER_STORE_DB})",
    )
    parser.add_argument("--reports", type=Path, default=DEFAULT_REPORT_DIR, help="report output directory")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="alias for init-game-store")
    sub.add_parser("init-game-store", help="create the replayable game store schema")
    sub.add_parser("init-teacher-store", help="create the pathless teacher store schema")
    sub.add_parser("init-compact", help="[experimental] create compact v2 schema — not production canonical")
    inv = sub.add_parser("inventory", help="scan repository and emit inventory reports")
    inv.add_argument("--full", action="store_true", help="full artifact inventory with disposition")

    migrate = sub.add_parser(
        "migrate-production",
        help="run full production migration from archived sources into canonical DB",
    )
    migrate.add_argument("--dry-run-only", action="store_true", help="dry-run imports only, do not commit")
    migrate.add_argument("--include-friend-shards", action="store_true", help="import KaAiData friend JSONL shards")

    friend = sub.add_parser("import-friend-shards", help="import KaAiData friend JSONL shards into teacher store only")
    friend.add_argument("--dry-run", action="store_true", help="validate all shards without committing")
    friend.add_argument("--inspect-only", action="store_true", help="inspect shard schemas only")
    friend.add_argument("--prove-idempotence", action="store_true", help="re-import and verify no-op after import")

    friend_rust = sub.add_parser(
        "import-friend-rust",
        help="import friend shards via Rust micropool importer (default; no Python fallback)",
    )
    friend_rust.add_argument("--threads", type=int, default=None, help="worker threads (default: logical CPUs)")
    friend_rust.add_argument("--batch-records", type=int, default=50_000, help="JSONL records per batch")
    friend_rust.add_argument("--no-resume", action="store_true", help="do not skip completed shards")
    friend_rust.add_argument(
        "--input",
        type=Path,
        default=None,
        help="friend corpus root (default: KaAiData selfplay_iters_000001_000020)",
    )

    inspect_friend = sub.add_parser("inspect-friend-shards", help="report schema/size/hash for all friend shards")

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

    ingest = sub.add_parser("ingest-shards", help="ingest .ready binary shards into the game store")
    ingest.add_argument("path", type=Path)
    ingest.add_argument("--dry-run", action="store_true")

    sub.add_parser("audit-canonical", help="alias for audit-game-store (game store only; see also audit-teacher-store)")
    sub.add_parser("audit-game-store", help="integrity audit of replayable game store (game_store.db)")
    sub.add_parser("audit-teacher-store", help="integrity audit of pathless teacher store")
    sub.add_parser(
        "teacher-semantic-checksum",
        help="order-independent teacher store checksum for cross-pipeline parity",
    )

    verify_pol = sub.add_parser("verify-teacher-policies", help="diagnose and verify all teacher policy sidecar references")
    verify_pol.add_argument("--path-only", action="store_true", help="path resolution only, skip payload read")
    verify_pol.add_argument("--limit", type=int, default=None, help="audit first N labels only")

    sub.add_parser("freeze-teacher-reference", help="mark SQLite teacher store as correctness reference (training_active=false)")

    build_ds = sub.add_parser("build-teacher-dataset", help="build candidate Parquet teacher dataset (not promoted until parity gates pass)")
    build_ds.add_argument("--output", type=Path, default=TEACHER_DATASET_CANDIDATE_DIR)
    build_ds.add_argument("--catalog", type=Path, default=TEACHER_CATALOG_DB)
    build_ds.add_argument("--compression", default="zstd", choices=["zstd", "snappy", "none"])

    parity = sub.add_parser("audit-position-parity", help="Rust/Python packed-state parity audit over friend corpus")
    parity.add_argument("--limit", type=int, default=None, help="audit first N JSONL records only")

    sub.add_parser("audit-teacher-dataset", help="audit built candidate Parquet teacher dataset manifest")
    stats_ds = sub.add_parser("stats-teacher-dataset", help="stats for candidate teacher dataset")
    stats_ds.add_argument("--output", type=Path, default=TEACHER_DATASET_CANDIDATE_DIR)
    stats_ds.add_argument("--catalog", type=Path, default=TEACHER_CATALOG_DB)
    bench_ds = sub.add_parser("benchmark-teacher-readers", help="benchmark DuckDB/Parquet read throughput")
    bench_ds.add_argument("--catalog", type=Path, default=TEACHER_CATALOG_DB)
    sub.add_parser("reconcile-teacher-source", help="explain SQLite label counts vs friend source records")
    verify_cand = sub.add_parser(
        "verify-candidate",
        help="read-only post-build check: manifest gates, no partial files, policy resolution (does NOT promote)",
    )
    verify_cand.add_argument("--output", type=Path, default=TEACHER_DATASET_CANDIDATE_DIR)

    fin = sub.add_parser(
        "finalize-teacher-candidate",
        help="copy source candidate to new target via .partial with fresh manifest (does not promote)",
    )
    fin.add_argument("--source", type=Path, required=True)
    fin.add_argument("--output", type=Path, required=True)
    fin.add_argument("--parent", type=str, default=None)
    fin.add_argument("--recovery-method", type=str, default=None)
    fin.add_argument(
        "--gate-bundle",
        type=Path,
        default=None,
        help="attach structured gate evidence bundle when finalizing (e.g. v9 -> v10)",
    )

    repair_m = sub.add_parser(
        "repair-candidate-manifest",
        help="rewrite manifest parts/bytes/hashes for on-disk candidate files (legacy repair only)",
    )
    repair_m.add_argument("--output", type=Path, required=True)

    gate_aud = sub.add_parser(
        "run-teacher-gate-audits",
        help="run promotion gate audits; writes reports and gate_evidence_bundle (does not mutate candidate manifest)",
    )
    gate_aud.add_argument(
        "--output",
        type=Path,
        default=ROOT / "training" / "data" / "teacher_dataset_candidate_v9",
    )
    gate_aud.add_argument("--reports", type=Path, default=DEFAULT_REPORT_DIR)
    gate_aud.add_argument(
        "--skip-slow",
        action="store_true",
        help="skip full position parity and JSONL miss classification",
    )
    gate_aud.add_argument(
        "--test-evidence",
        type=Path,
        default=None,
        help="path to teacher_dataset_test_evidence.json for required_tests gate",
    )
    sub.add_parser("stats-game-store", help="summary counts for game store")
    sub.add_parser("stats-teacher-store", help="summary counts for teacher store")
    sub.add_parser("verify-codec-parity", help="cross-store position codec/hash parity check")
    split = sub.add_parser("split-migration", help="preserve combined DB, restore game store, populate teacher store")
    split.add_argument("--skip-friend-import", action="store_true", help="skip friend shard import (teacher JSONL only)")
    sub.add_parser("audit-legacy-references", help="scan active code for prohibited legacy data references")
    sub.add_parser("prove-idempotence", help="re-import all sources and verify no semantic change")
    prove_reb = sub.add_parser("prove-rebuild", help="rebuild disposable copy and compare semantic checksums")
    prove_reb.add_argument("--migration-run-id", required=True)
    sub.add_parser("storage-audit", help="measure table/index/payload storage using SQLite page stats")
    sub.add_parser("stats", help="alias for stats-game-store (game store only; use stats-teacher-store for teacher DB)")
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

    export_train = sub.add_parser("export-training", help="alias for export-game-training")
    export_train.add_argument("out", type=Path)
    export_train.add_argument("--label-type", default="teacher_value")
    export_train.add_argument("--limit", type=int, default=None)

    export_game = sub.add_parser("export-game-training", help="export labeled rows from game store only")
    export_game.add_argument("out", type=Path)
    export_game.add_argument("--label-type", default="teacher_value")
    export_game.add_argument("--limit", type=int, default=None)

    export_teacher = sub.add_parser("export-teacher-training", help="export labeled rows from teacher store (--include-teacher-labels)")
    export_teacher.add_argument("out", type=Path)
    export_teacher.add_argument("--include-teacher-labels", action="store_true", help="required flag — teacher export is explicit")
    export_teacher.add_argument("--label-type", default="teacher_value")
    export_teacher.add_argument("--limit", type=int, default=None)

    export_mixed = sub.add_parser("export-mixed-training", help="explicit join/dedupe export from both stores")
    export_mixed.add_argument("out", type=Path)
    export_mixed.add_argument("--label-type", default="teacher_value")
    export_mixed.add_argument("--limit", type=int, default=None)

    import_teacher = sub.add_parser("import-teacher-positions", help="import pathless teacher JSONL sources into teacher store")
    import_teacher.add_argument("path", type=Path, nargs="?", default=None, help="optional single source; default imports all teacher sources")
    import_teacher.add_argument("--dry-run", action="store_true")

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
    sub.add_parser("audit", help="alias for audit-canonical → audit-game-store (game store only)")
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
        raise SystemExit(f"Refusing smoke database for production command: {db}\nUse: {GAME_STORE_DB}")


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
        if args.include_friend_shards:
            result = import_friend_shards(dry_run=True)
            print_json({"totals": result.totals, "per_shard": result.per_shard})
            return 0 if result.totals.get("unaccounted", 0) == 0 else 1
        from position_store_migration import run_dry_migration

        stats = run_dry_migration(GAME_STORE_DB, include_friend=False)
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


def command_import_friend_rust(args: argparse.Namespace) -> int:
    import subprocess

    bin_path = RUST_IMPORTER_BIN
    if not bin_path.exists():
        print(
            f"ERROR: Rust importer not built: {bin_path}\n"
            "Build with:\n"
            "  cd tools/position_store_importer && cargo build --release",
            file=sys.stderr,
        )
        return 1
    input_dir = args.input or FRIEND_CORPUS_DIR
    cmd = [
        str(bin_path),
        "--input",
        str(input_dir),
        "--teacher-db",
        str(args.teacher_db),
        "--sidecar-dir",
        str(TEACHER_SIDECARS),
        "--rel-root",
        str(ROOT),
        "--batch-records",
        str(args.batch_records),
    ]
    if args.threads is not None:
        cmd.extend(["--threads", str(args.threads)])
    if args.no_resume:
        cmd.append("--no-resume")
    print("Running:", " ".join(cmd), file=sys.stderr)
    proc = subprocess.run(cmd, cwd=str(ROOT))
    return int(proc.returncode or 0)


def command_import_friend_shards(args: argparse.Namespace) -> int:
    if args.inspect_only:
        print_json([__import__("dataclasses").asdict(i) for i in inspect_all_friend_shards()])
        return 0
    if args.prove_idempotence:
        result = prove_friend_idempotence(db_path=args.teacher_db)
        print_json(result)
        return 0 if result["passed"] else 1
    result = import_friend_shards(dry_run=args.dry_run, db_path=args.teacher_db)
    print_json(
        {
            "dry_run": args.dry_run,
            "migration_run_id": result.migration_run_id,
            "backup": __import__("dataclasses").asdict(result.backup) if result.backup else None,
            "totals": result.totals,
            "before": result.before,
            "after": result.after,
        }
    )
    if not args.dry_run:
        update_archive_manifest_friend_dispositions(result.migration_run_id)
    return 0 if result.totals.get("unaccounted", 0) == 0 else 1


def command_inspect_friend_shards(args: argparse.Namespace) -> int:
    inspections = inspect_all_friend_shards()
    print_json([__import__("dataclasses").asdict(i) for i in inspections])
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
    if cmd in {"init", "init-game-store"}:
        _require_production_db(args.db)
        init_game_store(args.db)
        print(json.dumps({"initialized_game_store": str(args.db)}, indent=2))
        return 0
    if cmd == "init-teacher-store":
        init_teacher_store(args.teacher_db)
        print(json.dumps({"initialized_teacher_store": str(args.teacher_db)}, indent=2))
        return 0
    if cmd == "init-compact":
        init_compact_db(args.db)
        print(json.dumps({"initialized_compact_experimental": str(args.db)}, indent=2))
        return 0
    if cmd == "inventory":
        return command_inventory(args)
    if cmd == "split-migration":
        result = run_split_migration(skip_friend_import=args.skip_friend_import)
        print_json(__import__("dataclasses").asdict(result))
        return 0 if result.game_audit.get("passed") and result.teacher_audit.get("passed") else 1
    if cmd == "migrate-production":
        return command_migrate_production(args)
    if cmd == "import-friend-rust":
        return command_import_friend_rust(args)
    if cmd == "import-friend-shards":
        return command_import_friend_shards(args)
    if cmd == "inspect-friend-shards":
        return command_inspect_friend_shards(args)
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
    if cmd in {"audit-canonical", "audit", "audit-game-store"}:
        _require_production_db(args.db)
        print_json(audit_game_store(args.db))
        return 0
    if cmd == "audit-teacher-store":
        print_json(audit_teacher_store(args.teacher_db))
        return 0
    if cmd == "teacher-semantic-checksum":
        print_json(teacher_semantic_checksum(args.teacher_db))
        return 0
    if cmd == "verify-teacher-policies":
        return cmd_verify_teacher_policies(args)
    if cmd == "freeze-teacher-reference":
        return cmd_freeze_teacher_reference(args)
    if cmd == "audit-position-parity":
        return cmd_audit_position_parity(args)
    if cmd == "build-teacher-dataset":
        return cmd_build_teacher_dataset(args)
    if cmd == "audit-teacher-dataset":
        if not hasattr(args, "output"):
            args.output = TEACHER_DATASET_CANDIDATE_DIR
        return cmd_audit_teacher_dataset(args)
    if cmd == "stats-teacher-dataset":
        if not hasattr(args, "output"):
            args.output = TEACHER_DATASET_CANDIDATE_DIR
        return cmd_stats_teacher_dataset(args)
    if cmd == "benchmark-teacher-readers":
        return cmd_benchmark_teacher_readers(args)
    if cmd == "reconcile-teacher-source":
        return cmd_reconcile_teacher_source(args)
    if cmd == "verify-candidate":
        return cmd_verify_candidate(args)
    if cmd == "finalize-teacher-candidate":
        return cmd_finalize_teacher_candidate(args)
    if cmd == "repair-candidate-manifest":
        return cmd_repair_candidate_manifest(args)
    if cmd == "run-teacher-gate-audits":
        return cmd_run_teacher_gate_audits(args)
    if cmd == "verify-codec-parity":
        print_json(verify_codec_parity(args.db, args.teacher_db))
        return 0
    if cmd == "import-teacher-positions":
        if args.path is not None:
            init_teacher_store(args.teacher_db)
            stats = import_path(
                args.teacher_db,
                args.path,
                dry_run=args.dry_run,
                report_dir=args.reports,
                teacher_import=True,
            )
            print_json(stats.__dict__)
            return 0
        result = import_teacher_sources(args.teacher_db, dry_run=args.dry_run)
        print_json(__import__("dataclasses").asdict(result))
        return 0 if result.totals.get("unaccounted", 0) == 0 else 1
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
    if cmd in {"stats", "stats-game-store"}:
        print_json(db_summary(args.db))
        return 0
    if cmd == "stats-teacher-store":
        print_json(db_summary(args.teacher_db))
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
    if cmd in {"export-training", "export-game-training"}:
        _require_production_db(args.db)
        count = export_training_rows(args.db, out_path=args.out, label_type=args.label_type, limit=args.limit)
        print(json.dumps({"wrote": str(args.out), "rows": count, "store": "game"}, indent=2))
        return 0
    if cmd == "export-teacher-training":
        if not args.include_teacher_labels:
            print("ERROR: teacher export requires --include-teacher-labels", file=sys.stderr)
            return 1
        count = export_teacher_training(
            args.out, db_path=args.teacher_db, label_type=args.label_type, limit=args.limit
        )
        print(json.dumps({"wrote": str(args.out), "rows": count, "store": "teacher"}, indent=2))
        return 0
    if cmd == "export-mixed-training":
        result = export_mixed_training(
            args.out,
            game_db=args.db,
            teacher_db=args.teacher_db,
            label_type=args.label_type,
            limit=args.limit,
        )
        print_json(result)
        return 0
    if cmd == "export-training-binary":
        result = export_training_binary(args.db, out_path=args.out, limit=args.limit, label_type_code_filter=args.label_type_code)
        print_json(result)
        return 0
    raise AssertionError(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
