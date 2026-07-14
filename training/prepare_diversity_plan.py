#!/usr/bin/env python3
"""Synthetic DIVERSITY_SPEC_V1 corpus plan — dry-run only, no engine launch."""
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
from diversity.planner import (
    allocation_content_hash,
    build_deterministic_allocation,
    default_prep_inputs,
)
from diversity.quota import allocate_quota_rows, validate_quota_shares
from titanium_training.paths import ENGINE_BIN


def _synthetic_rows(plan) -> list[CorpusSampleRow]:
    rows: list[CorpusSampleRow] = []
    producers = {
        DiversityLane.CLOSED_LOOP_POPULATION: ClosedLoopPopulationLane(),
        DiversityLane.BEHAVIORAL_CROSSPLAY: BehavioralCrossplayLane(),
        DiversityLane.PAIRED_FORKS: PairedForksLane(fork_sub_lane=ForkSubLane.REGRET_MINED),
        DiversityLane.SOLVER_SEAM: SolverSeamLane(),
        DiversityLane.EXACT_ANCHORS: ExactAnchorsLane(),
    }
    for lane, count in plan.per_lane.items():
        producer = producers.get(lane)
        if not producer:
            continue
        for i, meta in enumerate(producer.synthetic_rows(count)):
            rows.append(
                CorpusSampleRow(
                    metadata_lane=lane,
                    source_game_id=meta.source_game_id,
                    fork_lineage_id=meta.fork_lineage_id,
                    side_to_move=meta.side_to_move,
                    canonical=CanonicalStateRow(
                        pawn_positions=f"pawns-{meta.source_game_id}",
                        horizontal_walls=f"h-{i % 7}",
                        vertical_walls=f"v-{i % 5}",
                        wall_stocks="10,10",
                        side_to_move=meta.side_to_move,
                    ),
                    is_exact=lane == DiversityLane.EXACT_ANCHORS,
                    phase="opening" if i % 2 == 0 else "midgame",
                    tension="default" if i % 3 else "contested",
                )
            )
    return rows


def main(argv: list[str] | None = None) -> int:
    assert_dry_run_allowed()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, default=100_000)
    ap.add_argument("--dry-run", action="store_true", help="required flag; never launches engines")
    ap.add_argument("--corpus-id", default="dryrun-plan-0001")
    ap.add_argument("--planner-seed", type=int, default=0)
    ap.add_argument("--engine-semantics-hash", default="prep")
    ap.add_argument(
        "--out",
        type=Path,
        default=_TRAINING / "data" / "overnight_logs" / "diversity_dryrun_plan.json",
    )
    args = ap.parse_args(argv)
    if not args.dry_run:
        print("ERROR: --dry-run is required (prep phase)", file=sys.stderr)
        return 2
    validate_dry_run_output_path(args.out)

    planner_inputs = default_prep_inputs(
        corpus_generation_id=args.corpus_id,
        row_count=args.rows,
        planner_seed=args.planner_seed,
        engine_semantics_hash=args.engine_semantics_hash,
    )
    deterministic_allocation = build_deterministic_allocation(planner_inputs)
    allocation_hash = allocation_content_hash(planner_inputs)

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
        generation_config={"dry_run": True, "rows": args.rows, "planner_seed": args.planner_seed},
        label_config={"label_semantics": "prep"},
    )
    manifest_errors = manifest.validate()

    report = {
        "dry_run": True,
        "planned_rows": plan.total_rows,
        "deterministic_allocation": deterministic_allocation,
        "allocation_content_hash": allocation_hash,
        "per_lane": plan.to_dict()["per_lane"],
        "per_cell": plan.per_cell,
        "stm_distribution": plan.to_dict()["stm_distribution"],
        "seed_count": len(seeds),
        "distinct_opening_classes": len({s.two_ply_class for s in seeds}),
        "source_game_count": len({r.source_game_id for r in sample_rows}),
        "fork_lineage_count": len(
            {r.fork_lineage_id for r in sample_rows if r.fork_lineage_id}
        ),
        "certificate_feasibility": cert.to_dict(),
        "quota_validation_errors": quota_errors,
        "semantic_manifest": manifest.to_dict(),
        "manifest_errors": manifest_errors,
        "blockers": list(plan.blockers)
        + quota_errors
        + manifest_errors
        + (list(cert.reasons) if cert.status != CertificateStatus.PASS else []),
        "unresolved_blockers": [
            "real seeded centroid bank not generated",
            "ERR-MAP adaptive residual disabled",
            "solver seam / exact anchor tables not generated",
            "TRAINING_PREP_ONLY=1",
        ],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "generated_at"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
