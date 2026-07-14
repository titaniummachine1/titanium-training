"""Paired fork schemas and synthetic fixture validation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from diversity.lanes import ForkSubLane

MAX_ROWS_PER_FORK_LINEAGE = 128
FORK_REGRET_SHARE = 0.05
FORK_PLAUSIBLE_SHARE = 0.05


@dataclass(frozen=True)
class ForkBranch:
    branch: str
    move: str
    ranking_stable: bool
    action_rank: int


@dataclass(frozen=True)
class PairedForkFixture:
    parent_position_id: str
    lineage_id: str
    sub_lane: ForkSubLane
    branches: tuple[ForkBranch, ForkBranch]
    exact_root: bool = False

    def validate(self) -> list[str]:
        errors: list[str] = []
        if len(self.branches) != 2:
            errors.append("paired fork requires exactly two branches")
        if not self.lineage_id:
            errors.append("missing lineage_id")
        if not self.parent_position_id:
            errors.append("missing parent position")
        for br in self.branches:
            if not br.ranking_stable:
                errors.append(f"unstable ranking on branch {br.branch}")
        if not self.exact_root:
            for br in self.branches:
                if br.action_rank <= 0:
                    errors.append("non-exact fork requires positive action ranks")
        return errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "parent_position_id": self.parent_position_id,
            "lineage_id": self.lineage_id,
            "sub_lane": self.sub_lane.value,
            "exact_root": self.exact_root,
            "branches": [
                {
                    "branch": b.branch,
                    "move": b.move,
                    "ranking_stable": b.ranking_stable,
                    "action_rank": b.action_rank,
                }
                for b in self.branches
            ],
        }


def fixture_paired_fork(sub_lane: ForkSubLane = ForkSubLane.REGRET_MINED) -> PairedForkFixture:
    return PairedForkFixture(
        parent_position_id="parent-fixture-001",
        lineage_id=f"lineage-{sub_lane.value}-001",
        sub_lane=sub_lane,
        branches=(
            ForkBranch("best_pv", "e4", True, 1),
            ForkBranch("alt_pv", "d3", True, 2),
        ),
        exact_root=True,
    )
