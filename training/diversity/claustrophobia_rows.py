"""Claustrophobia-derived row contract — positions, not automatic Titanium teacher."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

CLAUSTROPHOBIA_ROW_SCHEMA_VERSION = "claustrophobia-derived-row-v1"
PILOT_CORPUS_CAP_FRACTION = 0.05
MAX_ROWS_PER_SOURCE_GAME = 32
MAX_ROWS_PER_FORK_LINEAGE = 128


class ClaustrophobiaDatasetKind(str, Enum):
    FROZEN_EVALUATION_GAMES = "frozen_evaluation_games"
    BOOK_CANDIDATES = "book_candidates"
    TRAINING_ELIGIBLE_CROSSPLAY = "training_eligible_crossplay"
    DISAGREEMENT_ROOTS = "disagreement_roots"
    RELABELED_FORKS = "relabeled_forks"


@dataclass(frozen=True)
class ClaustrophobiaDerivedRow:
    dataset_kind: str
    claustrophobia_release_tag: str
    claustrophobia_checkpoint_sha256: str
    repository_commit: str
    source_game_id: str
    opening_seed_id: str
    claustrophobia_chosen_move: str
    titanium_move: str
    final_game_outcome: str
    relabeling_status: str
    evaluation_eligible: bool
    training_eligible: bool
    schema_version: str = CLAUSTROPHOBIA_ROW_SCHEMA_VERSION
    book_eligible: bool = False

    def __post_init__(self) -> None:
        if self.dataset_kind not in {k.value for k in ClaustrophobiaDatasetKind}:
            raise ValueError(f"unknown dataset kind: {self.dataset_kind}")
        if self.dataset_kind == ClaustrophobiaDatasetKind.FROZEN_EVALUATION_GAMES.value:
            if self.training_eligible:
                raise ValueError("frozen evaluation games must never be training-eligible")
            if self.book_eligible:
                raise ValueError("frozen evaluation games must never be book-eligible")
        if self.evaluation_eligible and self.training_eligible:
            raise ValueError("same game must not be both evaluation and training eligible")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def enforce_pilot_caps(
    *,
    total_pilot_rows: int,
    claustrophobia_rows: int,
    rows_for_source_game: int,
    rows_for_fork_lineage: int,
) -> list[str]:
    errors: list[str] = []
    if total_pilot_rows > 0 and (claustrophobia_rows / total_pilot_rows) > PILOT_CORPUS_CAP_FRACTION + 1e-12:
        errors.append("pilot_cap_fraction_exceeded")
    if rows_for_source_game > MAX_ROWS_PER_SOURCE_GAME:
        errors.append("source_game_cap_exceeded")
    if rows_for_fork_lineage > MAX_ROWS_PER_FORK_LINEAGE:
        errors.append("fork_lineage_cap_exceeded")
    return errors
