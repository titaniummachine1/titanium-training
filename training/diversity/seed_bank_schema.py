"""Versioned Titanium seed-bank schema (design + fixture helpers only).

Do NOT create a production seed bank from this module until explicitly authorized.
Unknown origin categories are rejected.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

SEED_BANK_SCHEMA_VERSION = "titanium-seed-bank-v1"
PREFIX_METRIC_VERSION_V2 = "prefix-metric-v2-state-transitions"

ALLOWED_ORIGIN_CATEGORIES = frozenset(
    {
        "standard_start",
        "champion_generated_centroid",
        "historical_champion_centroid",
        "Claustrophobia_disagreement",
        "Claustrophobia_loss_root",
        "paired_fork",
        "solver_seam",
        "exact_anchor",
        "synthetic_fixture",
    }
)


class SeedActiveState(str, Enum):
    ACTIVE = "active"
    RETIRED = "retired"


class EvaluationLeakageStatus(str, Enum):
    CLEAR = "clear"
    EVAL_ONLY = "eval_only"
    UNKNOWN = "unknown"


class LegalityValidationStatus(str, Enum):
    VALID = "valid"
    INVALID = "invalid"
    UNCHECKED = "unchecked"


@dataclass(frozen=True)
class SeedRecord:
    """Every Titanium seed must carry these fields before selection is legal."""

    seed_id: str
    seed_family_id: str
    # Complete legal GameState (fixture: CanonicalStateRow fields serialized).
    game_state: dict[str, Any]
    reflection_canonical_state_key: str
    side_to_move: int
    pawn_locations: str
    placed_horizontal_walls: str
    placed_vertical_walls: str
    wall_stocks: str
    origin_source: str
    origin_game_id: str
    origin_lineage_id: str
    generation_method: str
    source_engine_opponent_ids: tuple[str, ...]
    phase: str
    tension_class: str
    engine_semantic_hash: str
    canonical_state_version: str
    prefix_metric_version: str
    evaluation_leakage_status: str
    legality_validation_status: str
    creation_timestamp: str
    active_retired_state: str
    schema_version: str = SEED_BANK_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.origin_source not in ALLOWED_ORIGIN_CATEGORIES:
            raise ValueError(f"unknown origin category rejected: {self.origin_source!r}")
        if self.evaluation_leakage_status == EvaluationLeakageStatus.EVAL_ONLY.value:
            # Records may exist for audit, but selection must reject them.
            pass
        if self.active_retired_state not in {s.value for s in SeedActiveState}:
            raise ValueError(f"invalid active/retired state: {self.active_retired_state!r}")
        if self.legality_validation_status not in {s.value for s in LegalityValidationStatus}:
            raise ValueError(f"invalid legality status: {self.legality_validation_status!r}")
        if self.side_to_move not in (0, 1):
            raise ValueError("side_to_move must be 0 or 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_seed_for_selection(seed: SeedRecord) -> list[str]:
    """Return rejection reasons; empty list means selectable."""
    reasons: list[str] = []
    if seed.origin_source not in ALLOWED_ORIGIN_CATEGORIES:
        reasons.append("unknown_origin")
    if seed.active_retired_state != SeedActiveState.ACTIVE.value:
        reasons.append("retired")
    if seed.evaluation_leakage_status != EvaluationLeakageStatus.CLEAR.value:
        reasons.append("evaluation_leakage")
    if seed.legality_validation_status != LegalityValidationStatus.VALID.value:
        reasons.append("illegal_or_unchecked")
    if not seed.reflection_canonical_state_key:
        reasons.append("missing_canonical_key")
    if not seed.engine_semantic_hash:
        reasons.append("missing_engine_semantic_hash")
    return reasons
