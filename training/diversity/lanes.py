"""Typed lane interfaces for DIVERSITY_SPEC_V1 (fixture producers only for now)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator

DIVERSITY_SPEC_VERSION = "DIVERSITY_SPEC_V1"


class DiversityLane(str, Enum):
    CLOSED_LOOP_POPULATION = "closed_loop_population"
    BEHAVIORAL_CROSSPLAY = "behavioral_crossplay"
    PAIRED_FORKS = "paired_forks"
    SOLVER_SEAM = "solver_seam"
    EXACT_ANCHORS = "exact_anchors"
    ADAPTIVE_RESIDUAL = "adaptive_residual"


class ForkSubLane(str, Enum):
    REGRET_MINED = "regret_mined"
    PLAUSIBLE_DEVIATION = "plausible_deviation"


@dataclass(frozen=True)
class CorpusRowMetadata:
    lane: DiversityLane
    source_engine_id: str
    opponent_engine_id: str | None
    style_treatment: str | None
    root_seed_id: str | None
    source_game_id: str
    fork_lineage_id: str | None
    parent_lineage_id: str | None
    solver_topology_id: str | None
    exact_label_kind: str | None
    side_to_move: int
    generation_semantic_version: str
    fork_sub_lane: ForkSubLane | None = None
    seed_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lane": self.lane.value,
            "source_engine_id": self.source_engine_id,
            "opponent_engine_id": self.opponent_engine_id,
            "style_treatment": self.style_treatment,
            "root_seed_id": self.root_seed_id,
            "seed_id": self.seed_id,
            "source_game_id": self.source_game_id,
            "fork_lineage_id": self.fork_lineage_id,
            "parent_lineage_id": self.parent_lineage_id,
            "solver_topology_id": self.solver_topology_id,
            "exact_label_kind": self.exact_label_kind,
            "side_to_move": self.side_to_move,
            "generation_semantic_version": self.generation_semantic_version,
            "fork_sub_lane": self.fork_sub_lane.value if self.fork_sub_lane else None,
            **self.extra,
        }


class LaneProducer(ABC):
    lane: DiversityLane

    @abstractmethod
    def synthetic_rows(self, count: int) -> Iterator[CorpusRowMetadata]:
        raise NotImplementedError


@dataclass(frozen=True)
class ClosedLoopPopulationLane(LaneProducer):
    lane: DiversityLane = DiversityLane.CLOSED_LOOP_POPULATION

    def synthetic_rows(self, count: int) -> Iterator[CorpusRowMetadata]:
        for i in range(count):
            yield CorpusRowMetadata(
                lane=self.lane,
                source_engine_id="current",
                opponent_engine_id="previous_accepted",
                style_treatment=None,
                root_seed_id=f"seed-clp-{i:05d}",
                source_game_id=f"synthetic-clp-{i:05d}",
                fork_lineage_id=None,
                parent_lineage_id=None,
                solver_topology_id=None,
                exact_label_kind=None,
                side_to_move=i % 2,
                generation_semantic_version=DIVERSITY_SPEC_VERSION,
                seed_id=f"seed-clp-{i:05d}",
            )


@dataclass(frozen=True)
class BehavioralCrossplayLane(LaneProducer):
    lane: DiversityLane = DiversityLane.BEHAVIORAL_CROSSPLAY
    style_treatment: str = "style_fixture_a"

    def synthetic_rows(self, count: int) -> Iterator[CorpusRowMetadata]:
        for i in range(count):
            yield CorpusRowMetadata(
                lane=self.lane,
                source_engine_id="current",
                opponent_engine_id="style_variant",
                style_treatment=self.style_treatment,
                root_seed_id=f"seed-bcx-{i:05d}",
                source_game_id=f"synthetic-bcx-{i:05d}",
                fork_lineage_id=None,
                parent_lineage_id=None,
                solver_topology_id=None,
                exact_label_kind=None,
                side_to_move=i % 2,
                generation_semantic_version=DIVERSITY_SPEC_VERSION,
                seed_id=f"seed-bcx-{i:05d}",
            )


@dataclass(frozen=True)
class PairedForksLane(LaneProducer):
    lane: DiversityLane = DiversityLane.PAIRED_FORKS
    fork_sub_lane: ForkSubLane = ForkSubLane.REGRET_MINED

    def synthetic_rows(self, count: int) -> Iterator[CorpusRowMetadata]:
        pairs = max(1, count // 2)
        for i in range(pairs):
            lineage = f"fork-lineage-{self.fork_sub_lane.value}-{i:04d}"
            parent = f"parent-pos-{i:04d}"
            for branch, suffix in (("best_pv", "a"), ("alt_pv", "b")):
                yield CorpusRowMetadata(
                    lane=self.lane,
                    source_engine_id="current",
                    opponent_engine_id="current",
                    style_treatment=None,
                    root_seed_id=None,
                    source_game_id=f"synthetic-fork-{i:04d}-{suffix}",
                    fork_lineage_id=lineage,
                    parent_lineage_id=parent,
                    solver_topology_id=None,
                    exact_label_kind=None,
                    side_to_move=0,
                    generation_semantic_version=DIVERSITY_SPEC_VERSION,
                    fork_sub_lane=self.fork_sub_lane,
                    extra={"branch": branch},
                )


@dataclass(frozen=True)
class SolverSeamLane(LaneProducer):
    lane: DiversityLane = DiversityLane.SOLVER_SEAM

    def synthetic_rows(self, count: int) -> Iterator[CorpusRowMetadata]:
        for i in range(count):
            yield CorpusRowMetadata(
                lane=self.lane,
                source_engine_id="current",
                opponent_engine_id="current",
                style_treatment=None,
                root_seed_id=f"seam-seed-{i:05d}",
                source_game_id=f"synthetic-seam-{i:05d}",
                fork_lineage_id=None,
                parent_lineage_id=None,
                solver_topology_id=f"seam-topology-{i % 8:02d}",
                exact_label_kind=None,
                side_to_move=i % 2,
                generation_semantic_version=DIVERSITY_SPEC_VERSION,
                seed_id=f"seam-seed-{i:05d}",
                extra={"exact_status": "non_exact_adjacent"},
            )


@dataclass(frozen=True)
class ExactAnchorsLane(LaneProducer):
    lane: DiversityLane = DiversityLane.EXACT_ANCHORS
    exact_kind: str = "race_exact"

    def synthetic_rows(self, count: int) -> Iterator[CorpusRowMetadata]:
        for i in range(count):
            yield CorpusRowMetadata(
                lane=self.lane,
                source_engine_id="solver",
                opponent_engine_id=None,
                style_treatment=None,
                root_seed_id=f"anchor-{i:05d}",
                source_game_id=f"synthetic-anchor-{i:05d}",
                fork_lineage_id=None,
                parent_lineage_id=None,
                solver_topology_id=f"exact-topology-{i % 4:02d}",
                exact_label_kind=self.exact_kind,
                side_to_move=i % 2,
                generation_semantic_version=DIVERSITY_SPEC_VERSION,
                seed_id=f"anchor-{i:05d}",
                extra={"exact_label": "Some"},
            )


@dataclass(frozen=True)
class AdaptiveResidualLane(LaneProducer):
    """Reserved — disabled until ERR-MAP validates on held-out regret."""

    lane: DiversityLane = DiversityLane.ADAPTIVE_RESIDUAL
    enabled: bool = False

    def synthetic_rows(self, count: int) -> Iterator[CorpusRowMetadata]:
        if self.enabled:
            raise NotImplementedError("ERR-MAP adaptive residual not validated yet")
        if count:
            raise RuntimeError("adaptive residual disabled; use static round-robin fallback")
        return iter(())
