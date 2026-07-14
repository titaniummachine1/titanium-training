"""Frozen engine-semantics contract for corpus compatibility decisions."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields
from enum import Enum
from pathlib import Path
from typing import Any

SCHEMA_PATH = Path(__file__).resolve().parent / "contracts" / "engine_semantics.schema.json"

REQUIRED_SEMANTIC_FIELDS = (
    "engine_semantic_version",
    "game_rules_version",
    "canonical_state_version",
    "move_encoding_version",
    "nnue_feature_schema_version",
    "evaluation_semantics_version",
    "score_band_version",
    "oracle_semantics_version",
    "search_label_semantics_version",
    "zobrist_version",
    "binary_sha256",
    "source_commit",
    "generated_at",
)


class CompatibilityClass(str, Enum):
    COMPATIBLE = "compatible"
    RELABEL_REQUIRED = "relabel_required"
    REGENERATION_REQUIRED = "regeneration_required"
    INVALID = "invalid"


@dataclass(frozen=True)
class EngineSemanticsContract:
    engine_semantic_version: str
    game_rules_version: str
    canonical_state_version: str
    move_encoding_version: str
    nnue_feature_schema_version: str
    evaluation_semantics_version: str
    score_band_version: str
    oracle_semantics_version: str
    search_label_semantics_version: str
    zobrist_version: str
    binary_sha256: str
    source_commit: str
    generated_at: str
    dirty_tree_hash: str | None = None
    prefix_metric_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out = {f.name: getattr(self, f.name) for f in fields(self)}
        return {k: v for k, v in out.items() if v is not None}

    def semantics_hash(self) -> str:
        payload = {k: getattr(self, k) for k in REQUIRED_SEMANTIC_FIELDS}
        if self.dirty_tree_hash:
            payload["dirty_tree_hash"] = self.dirty_tree_hash
        if self.prefix_metric_version:
            payload["prefix_metric_version"] = self.prefix_metric_version
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

    def validate(self) -> list[str]:
        errors: list[str] = []
        for name in REQUIRED_SEMANTIC_FIELDS:
            val = getattr(self, name, None)
            if val is None or str(val).strip() == "":
                errors.append(f"missing required field: {name}")
        if self.binary_sha256 and len(self.binary_sha256) != 64:
            errors.append("binary_sha256 must be 64 hex chars")
        unknown_marker = ("unknown", "prep", "missing", "todo")
        for name in REQUIRED_SEMANTIC_FIELDS:
            val = str(getattr(self, name, "")).lower()
            if any(m in val for m in unknown_marker) and name != "engine_semantic_version":
                errors.append(f"unknown or placeholder semantics: {name}={getattr(self, name)!r}")
        return errors


def classify_compatibility(
    left: EngineSemanticsContract,
    right: EngineSemanticsContract,
) -> CompatibilityClass:
    lv = left.validate()
    rv = right.validate()
    if lv or rv:
        return CompatibilityClass.INVALID
    if left.semantics_hash() == right.semantics_hash():
        return CompatibilityClass.COMPATIBLE
    if (
        left.game_rules_version == right.game_rules_version
        and left.canonical_state_version == right.canonical_state_version
        and left.move_encoding_version == right.move_encoding_version
        and left.nnue_feature_schema_version == right.nnue_feature_schema_version
        and left.evaluation_semantics_version != right.evaluation_semantics_version
    ):
        return CompatibilityClass.RELABEL_REQUIRED
    if left.game_rules_version != right.game_rules_version:
        return CompatibilityClass.REGENERATION_REQUIRED
    if left.canonical_state_version != right.canonical_state_version:
        return CompatibilityClass.REGENERATION_REQUIRED
    if left.move_encoding_version != right.move_encoding_version:
        return CompatibilityClass.REGENERATION_REQUIRED
    if left.zobrist_version != right.zobrist_version:
        return CompatibilityClass.REGENERATION_REQUIRED
    if left.nnue_feature_schema_version != right.nnue_feature_schema_version:
        return CompatibilityClass.REGENERATION_REQUIRED
    if left.score_band_version != right.score_band_version:
        return CompatibilityClass.RELABEL_REQUIRED
    if left.search_label_semantics_version != right.search_label_semantics_version:
        return CompatibilityClass.RELABEL_REQUIRED
    if left.oracle_semantics_version != right.oracle_semantics_version:
        return CompatibilityClass.REGENERATION_REQUIRED
    left_prefix = left.prefix_metric_version or ""
    right_prefix = right.prefix_metric_version or ""
    if left_prefix and right_prefix and left_prefix != right_prefix:
        return CompatibilityClass.RELABEL_REQUIRED
    return CompatibilityClass.INVALID


def reject_incompatible(left: EngineSemanticsContract, right: EngineSemanticsContract) -> None:
    decision = classify_compatibility(left, right)
    if decision == CompatibilityClass.COMPATIBLE:
        return
    raise ValueError(
        f"incompatible engine semantics: {decision.value} "
        f"({left.engine_semantic_version} vs {right.engine_semantic_version})"
    )


def prep_placeholder_contract(*, binary_sha256: str = "0" * 64) -> EngineSemanticsContract:
    from datetime import datetime, timezone

    return EngineSemanticsContract(
        engine_semantic_version="engine-semantics-prep",
        game_rules_version="quoridor-rules-prep",
        canonical_state_version="canonical-state-prep",
        move_encoding_version="move-alg-v1",
        nnue_feature_schema_version="nnue-fv-v1",
        evaluation_semantics_version="eval-semantics-prep",
        score_band_version="score-band-prep",
        oracle_semantics_version="oracle-semantics-prep",
        search_label_semantics_version="search-label-prep",
        zobrist_version="zobrist-prep",
        binary_sha256=binary_sha256,
        source_commit="prep",
        generated_at=datetime.now(timezone.utc).isoformat(),
        dirty_tree_hash=None,
    )
