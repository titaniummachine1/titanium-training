#!/usr/bin/env python3
"""Synthetic DIVERSITY_SPEC_V1 corpus assembly rehearsal — no engines, no real DB writes."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_TRAINING = Path(__file__).resolve().parent
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

from prep_guard import assert_dry_run_allowed, validate_dry_run_output_path
from corpus_semantic_manifest import build_prep_manifest
from diversity.centroids import build_fixture_seed_bank
from diversity.certificate import (
    CertificateMeasurementContext,
    CertificateStatus,
    CorpusSampleRow,
    validate_full_certificate,
)
from diversity.canonical import CanonicalStateRow
from diversity.lanes import (
    BehavioralCrossplayLane,
    ClosedLoopPopulationLane,
    DiversityLane,
    ExactAnchorsLane,
    ForkSubLane,
    PairedForksLane,
    SolverSeamLane,
)
from diversity.planner import allocation_content_hash, default_prep_inputs
from diversity.prefix_metrics import PREFIX_METRIC_VERSION, fixture_prefix_context, prefix2_key
from diversity.promotion_record import build_synthetic_validation_report, fixture_promotion_record
from diversity.provenance import (
    IngestDecision,
    SourceCategory,
    fixture_provenance,
    validate_import_provenance,
)
from diversity.quota import allocate_quota_rows, validate_quota_shares
from engine_semantic_contract import prep_placeholder_contract
from launch_gate import validate_launch_gate
from prepare_diversity_plan import _synthetic_rows
from titanium_training.paths import ENGINE_BIN

LOG_DIR = _TRAINING / "data" / "overnight_logs"
BANNER = "SYNTHETIC_PREPARATION_REHEARSAL_ONLY"


def _fixture_scenarios() -> dict[str, dict]:
    plan = allocate_quota_rows(1000, err_map_validated=False)
    pass_rows = _synthetic_rows(plan)[:64]
    block_rows = _synthetic_rows(plan)
    for i, row in enumerate(block_rows):
        if i < 900:
            block_rows[i] = CorpusSampleRow(
                metadata_lane=row.metadata_lane,
                source_game_id="same-game",
                fork_lineage_id=row.fork_lineage_id,
                side_to_move=row.side_to_move,
                canonical=row.canonical,
            )
    invalid_rows = _synthetic_rows(plan)[:4]
    return {
        "PASS": {"rows": pass_rows, "prefixes_trusted": True},
        "BLOCK": {"rows": block_rows, "prefixes_trusted": True},
        "INVALID": {"rows": invalid_rows, "prefixes_trusted": False},
    }


def main(argv: list[str] | None = None) -> int:
    assert_dry_run_allowed()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, default=100_000)
    ap.add_argument("--planner-seed", type=int, default=1337)
    ap.add_argument("--corpus-id", default="prep-rehearsal-001")
    ap.add_argument("--dry-run", action="store_true", required=True)
    ap.add_argument("--out-dir", type=Path, default=LOG_DIR)
    args = ap.parse_args(argv)
    for name in (
        "diversity_rehearsal_plan.json",
        "diversity_rehearsal_certificate.json",
        "diversity_rehearsal_manifest.json",
        "diversity_rehearsal_provenance_report.json",
        "diversity_rehearsal_blockers.json",
    ):
        validate_dry_run_output_path(args.out_dir / name)

    planner_inputs = default_prep_inputs(
        corpus_generation_id=args.corpus_id,
        row_count=args.rows,
        planner_seed=args.planner_seed,
    )
    plan = allocate_quota_rows(args.rows, err_map_validated=False)
    quota_errors = validate_quota_shares(plan.per_lane, plan.total_rows)
    seeds = build_fixture_seed_bank()
    sample_rows = _synthetic_rows(plan)
    prefixes = [seed.moves for seed in seeds[:64]]

    cert = validate_full_certificate(
        sample_rows,
        prefixes=prefixes,
        per_lane_counts=plan.per_lane,
        total_rows=plan.total_rows,
        measurement_context=CertificateMeasurementContext(
            diversity_spec_version=planner_inputs.diversity_spec_version,
            engine_semantics_hash=planner_inputs.engine_semantics_hash,
            corpus_generation_id=planner_inputs.corpus_generation_id,
            prefixes_trusted=True,
            prior_corpora_trusted=False,
        ),
    )
    manifest = build_prep_manifest(
        engine_bin=ENGINE_BIN,
        corpus_generation_id=args.corpus_id,
        generation_config={"dry_run": True, "prefix_metric_version": PREFIX_METRIC_VERSION},
        label_config={"label_semantics": "prep"},
    )
    prov = fixture_provenance(source_kind=SourceCategory.SYNTHETIC_FIXTURE)
    prov_result = validate_import_provenance(prov, prep_only=True, generation_approved=False)

    scenario_certs = {}
    for label, cfg in _fixture_scenarios().items():
        scenario_certs[label] = validate_full_certificate(
            cfg["rows"],
            prefixes=prefixes if cfg["prefixes_trusted"] else [],
            per_lane_counts=plan.per_lane,
            total_rows=len(cfg["rows"]),
            measurement_context=CertificateMeasurementContext(
                diversity_spec_version=planner_inputs.diversity_spec_version,
                engine_semantics_hash=planner_inputs.engine_semantics_hash,
                corpus_generation_id=planner_inputs.corpus_generation_id,
                prefixes_trusted=cfg["prefixes_trusted"],
                prior_corpora_trusted=False,
            ),
        ).to_dict()

    prefix_keys = []
    for seed in seeds[:8]:
        ctx = fixture_prefix_context(root_seed_id=seed.seed_id, start_state=seed.start_state if hasattr(seed, "start_state") else None)
        if len(seed.moves) >= 2:
            key = prefix2_key(ctx, (seed.moves[0], seed.moves[1]))
            if key:
                prefix_keys.append(key)

    promotion_pass = fixture_promotion_record(passed=True).to_dict()
    promotion_fail = fixture_promotion_record(passed=False).to_dict()
    synthetic_validation = build_synthetic_validation_report(passed=True)

    gate = validate_launch_gate(
        corpus_generation_id=args.corpus_id,
        manifest=manifest,
        engine_contract=prep_placeholder_contract(),
    )

    plan_report = {
        BANNER: True,
        "planned_rows": plan.total_rows,
        "allocation_content_hash": allocation_content_hash(planner_inputs),
        "per_lane": plan.per_lane,
        "prefix_metric_version": PREFIX_METRIC_VERSION,
        "distinct_prefix2_keys_sample": len(set(prefix_keys)),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    cert_report = {
        BANNER: True,
        "main_certificate": cert.to_dict(),
        "scenario_certificates": scenario_certs,
        "synthetic_promotion_pass": promotion_pass,
        "synthetic_promotion_fail": promotion_fail,
        "synthetic_validation": synthetic_validation,
    }
    manifest_report = {
        BANNER: True,
        "manifest": manifest.to_dict(),
        "manifest_errors": manifest.validate(),
        "engine_contract_hash": prep_placeholder_contract().semantics_hash(),
    }
    provenance_report = {
        BANNER: True,
        "fixture_provenance": prov.to_dict(),
        "fixture_decision": prov_result.decision.value,
        "ka_teacher_audit": {
            "direct_labels_db_writes": [
                "training/tools/ka_teacher/ka_ab_collect_labels.py",
                "training/tools/ka_teacher/ka_nn_collect_labels.py",
            ],
            "policy": "must pass validate_import_provenance before real ingest",
            "legacy_rows": "INVALID_LEGACY_PROVENANCE until metadata backfilled",
        },
        "opening_book_audit": {
            "paths": ["training/tools/opening_book/"],
            "policy": "opening_book_import requires full provenance at ingest boundary",
        },
    }
    blockers = {
        BANNER: True,
        "quota_errors": quota_errors,
        "launch_gate_blockers": list(gate.blockers),
        "unresolved": [
            "real seeded centroid bank",
            "ERR-MAP validation",
            "engine semantic freeze",
            "TRAINING_PREP_ONLY=1",
            "no APPROVE_GENERATION.json",
        ],
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "diversity_rehearsal_plan.json").write_text(
        json.dumps(plan_report, indent=2) + "\n", encoding="utf-8"
    )
    (args.out_dir / "diversity_rehearsal_certificate.json").write_text(
        json.dumps(cert_report, indent=2) + "\n", encoding="utf-8"
    )
    (args.out_dir / "diversity_rehearsal_manifest.json").write_text(
        json.dumps(manifest_report, indent=2) + "\n", encoding="utf-8"
    )
    (args.out_dir / "diversity_rehearsal_provenance_report.json").write_text(
        json.dumps(provenance_report, indent=2) + "\n", encoding="utf-8"
    )
    (args.out_dir / "diversity_rehearsal_blockers.json").write_text(
        json.dumps(blockers, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"banner": BANNER, "corpus_id": args.corpus_id, "certificate_status": cert.status.value}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
