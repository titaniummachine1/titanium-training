"""Deterministic DIVERSITY_SPEC_V1 dry-run planner (no side effects)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from diversity.lanes import DIVERSITY_SPEC_VERSION
from diversity.quota import QuotaPlan, allocate_quota_rows, validate_quota_shares


@dataclass(frozen=True)
class PlannerInputs:
    corpus_generation_id: str
    row_count: int
    planner_seed: int
    diversity_spec_version: str
    engine_semantics_hash: str


def build_deterministic_allocation(inputs: PlannerInputs) -> dict[str, Any]:
    """Byte-stable allocation payload (excludes timestamps)."""
    plan: QuotaPlan = allocate_quota_rows(inputs.row_count, err_map_validated=False)
    per_lane = {lane.value: count for lane, count in sorted(plan.per_lane.items(), key=lambda kv: kv[0].value)}
    per_cell = dict(sorted(plan.per_cell.items()))
    payload = {
        "corpus_generation_id": inputs.corpus_generation_id,
        "row_count": inputs.row_count,
        "planner_seed": inputs.planner_seed,
        "diversity_spec_version": inputs.diversity_spec_version,
        "engine_semantics_hash": inputs.engine_semantics_hash,
        "per_lane": per_lane,
        "per_cell": per_cell,
        "stm_white": plan.stm_white,
        "stm_black": plan.stm_black,
        "quota_validation_errors": validate_quota_shares(plan.per_lane, plan.total_rows),
    }
    return payload


def allocation_content_hash(inputs: PlannerInputs) -> str:
    payload = build_deterministic_allocation(inputs)
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def default_prep_inputs(
    *,
    corpus_generation_id: str,
    row_count: int,
    planner_seed: int = 0,
    engine_semantics_hash: str = "prep",
) -> PlannerInputs:
    return PlannerInputs(
        corpus_generation_id=corpus_generation_id,
        row_count=row_count,
        planner_seed=planner_seed,
        diversity_spec_version=DIVERSITY_SPEC_VERSION,
        engine_semantics_hash=engine_semantics_hash,
    )
