"""Central provenance contract for imported training candidates."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields
from enum import Enum
from typing import Any

from diversity.canonical import CanonicalStateRow


class SourceCategory(str, Enum):
    GENERATED_SELFPLAY = "generated_selfplay"
    BEHAVIORAL_CROSSPLAY = "behavioral_crossplay"
    PAIRED_FORK = "paired_fork"
    SOLVER_SEAM = "solver_seam"
    EXACT_ANCHOR = "exact_anchor"
    KA_TEACHER_IMPORT = "ka_teacher_import"
    OPENING_BOOK_IMPORT = "opening_book_import"
    EXTERNAL_GAME_IMPORT = "external_game_import"
    HISTORICAL_CORPUS_IMPORT = "historical_corpus_import"
    SYNTHETIC_FIXTURE = "synthetic_fixture"


class IngestDecision(str, Enum):
    ACCEPT_PREP_FIXTURE = "ACCEPT_PREP_FIXTURE"
    REJECT_MISSING_PROVENANCE = "REJECT_MISSING_PROVENANCE"
    REJECT_SEMANTIC_MISMATCH = "REJECT_SEMANTIC_MISMATCH"
    REJECT_EVALUATION_LEAKAGE = "REJECT_EVALUATION_LEAKAGE"
    REJECT_CANONICAL_DUPLICATE = "REJECT_CANONICAL_DUPLICATE"
    REJECT_FORBIDDEN_LINEAGE = "REJECT_FORBIDDEN_LINEAGE"
    REJECT_UNKNOWN_SOURCE = "REJECT_UNKNOWN_SOURCE"
    REJECT_NOT_GENERATION_APPROVED = "REJECT_NOT_GENERATION_APPROVED"
    INVALID_LEGACY_PROVENANCE = "INVALID_LEGACY_PROVENANCE"


MANDATORY_PROVENANCE_FIELDS = (
    "source_kind",
    "source_tool",
    "source_repository_commit",
    "source_file_hash",
    "source_game_id",
    "engine_semantic_hash",
    "generation_config_hash",
    "label_config_hash",
    "game_rules_version",
    "canonical_state_version",
    "move_encoding_version",
    "evaluation_only",
    "passed_garbage_filter",
    "corpus_eligible",
)


@dataclass(frozen=True)
class ImportedRowProvenance:
    source_kind: SourceCategory
    source_tool: str
    source_repository_commit: str
    source_file_hash: str
    source_game_id: str
    engine_semantic_hash: str
    generation_config_hash: str
    label_config_hash: str
    game_rules_version: str
    canonical_state_version: str
    move_encoding_version: str
    evaluation_only: bool
    passed_garbage_filter: bool
    corpus_eligible: bool
    source_lineage_id: str | None = None
    parent_lineage_id: str | None = None
    root_seed_id: str | None = None
    engine_identity: str | None = None
    opponent_identity: str | None = None
    style_id: str | None = None
    exact_label_kind: str | None = None
    exact_label_provenance: str | None = None
    side_to_move: int | None = None
    passed_eval_leakage_check: bool = False
    passed_canonical_dedup: bool = False
    legacy_untrusted: bool = False

    def to_dict(self) -> dict[str, Any]:
        out = {f.name: getattr(self, f.name) for f in fields(self)}
        out["source_kind"] = self.source_kind.value
        return out

    def content_hash(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()


@dataclass(frozen=True)
class IngestValidationResult:
    decision: IngestDecision
    reasons: tuple[str, ...] = ()
    provenance_hash: str | None = None

    @property
    def accepted_fixture(self) -> bool:
        return self.decision == IngestDecision.ACCEPT_PREP_FIXTURE


def validate_import_provenance(
    provenance: ImportedRowProvenance | None,
    *,
    canonical: CanonicalStateRow | None = None,
    prep_only: bool = True,
    generation_approved: bool = False,
) -> IngestValidationResult:
    if provenance is None:
        return IngestValidationResult(
            decision=IngestDecision.REJECT_MISSING_PROVENANCE,
            reasons=("missing provenance record",),
        )
    if provenance.legacy_untrusted:
        return IngestValidationResult(
            decision=IngestDecision.INVALID_LEGACY_PROVENANCE,
            reasons=("legacy row lacks trustworthy provenance",),
        )
    missing: list[str] = []
    for name in MANDATORY_PROVENANCE_FIELDS:
        val = getattr(provenance, name, None)
        if val is None or (isinstance(val, str) and not str(val).strip()):
            missing.append(name)
    if missing:
        return IngestValidationResult(
            decision=IngestDecision.REJECT_MISSING_PROVENANCE,
            reasons=tuple(f"missing: {m}" for m in missing),
        )
    try:
        SourceCategory(provenance.source_kind.value if isinstance(provenance.source_kind, SourceCategory) else provenance.source_kind)
    except ValueError:
        return IngestValidationResult(
            decision=IngestDecision.REJECT_UNKNOWN_SOURCE,
            reasons=(f"unknown source_kind: {provenance.source_kind!r}",),
        )
    if provenance.evaluation_only and provenance.corpus_eligible:
        return IngestValidationResult(
            decision=IngestDecision.REJECT_EVALUATION_LEAKAGE,
            reasons=("evaluation-only source marked corpus_eligible",),
        )
    if not provenance.passed_eval_leakage_check and provenance.source_kind not in (
        SourceCategory.SYNTHETIC_FIXTURE,
    ):
        return IngestValidationResult(
            decision=IngestDecision.REJECT_EVALUATION_LEAKAGE,
            reasons=("evaluation leakage check not passed",),
        )
    if prep_only and not generation_approved:
        return IngestValidationResult(
            decision=IngestDecision.REJECT_NOT_GENERATION_APPROVED,
            reasons=("TRAINING_PREP_ONLY=1 blocks real corpus ingest",),
            provenance_hash=provenance.content_hash(),
        )
    if canonical is not None and not provenance.passed_canonical_dedup:
        return IngestValidationResult(
            decision=IngestDecision.REJECT_CANONICAL_DUPLICATE,
            reasons=("canonical dedup not recorded",),
        )
    return IngestValidationResult(
        decision=IngestDecision.ACCEPT_PREP_FIXTURE,
        provenance_hash=provenance.content_hash(),
    )


def fixture_provenance(
    *,
    source_kind: SourceCategory = SourceCategory.SYNTHETIC_FIXTURE,
    source_game_id: str = "fixture-001",
) -> ImportedRowProvenance:
    return ImportedRowProvenance(
        source_kind=source_kind,
        source_tool="fixture",
        source_repository_commit="fixture",
        source_file_hash="0" * 64,
        source_game_id=source_game_id,
        engine_semantic_hash="a" * 64,
        generation_config_hash="b" * 64,
        label_config_hash="c" * 64,
        game_rules_version="quoridor-rules-prep",
        canonical_state_version="canonical-state-v1",
        move_encoding_version="move-alg-v1",
        evaluation_only=False,
        passed_garbage_filter=True,
        corpus_eligible=False,
        passed_eval_leakage_check=True,
        passed_canonical_dedup=True,
        root_seed_id="seed-fixture-001",
    )
