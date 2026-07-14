"""Solver seam and exact anchor validators (schema-only in prep phase)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

EXACT_ANCHOR_SHARE = 0.05


@dataclass(frozen=True)
class SolverSeamRecord:
    parent_position_id: str
    solver_topology_id: str
    adjacent_to_certified: bool
    honest_non_exact: bool

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.honest_non_exact:
            errors.append("seam parent must be honestly non-exact")
        if not self.adjacent_to_certified:
            errors.append("seam parent must be adjacent to certified region")
        if not self.solver_topology_id:
            errors.append("missing solver_topology_id")
        return errors


@dataclass(frozen=True)
class ExactAnchorRecord:
    position_id: str
    exact_label_kind: str
    exact_label: str
    sign: int
    side_to_move: int

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.exact_label != "Some":
            errors.append("exact anchors require exact_label(Some)")
        if self.sign not in (-1, 0, 1):
            errors.append("sign must be -1, 0, or 1")
        if self.side_to_move not in (0, 1):
            errors.append("side_to_move must be 0 or 1")
        return errors


def validate_exact_anchor_balance(records: list[ExactAnchorRecord]) -> list[str]:
    errors: list[str] = []
    race_exact = sum(1 for r in records if r.exact_label_kind == "race_exact")
    race1w = sum(1 for r in records if r.exact_label_kind == "race1w")
    if abs(race_exact - race1w) > 1:
        errors.append(f"race_exact/race1w imbalance: {race_exact} vs {race1w}")
    stm0 = sum(1 for r in records if r.side_to_move == 0)
    stm1 = len(records) - stm0
    if records:
        frac0 = stm0 / len(records)
        if not (0.45 <= frac0 <= 0.55):
            errors.append(f"STM imbalance: white fraction {frac0:.3f}")
    return errors


def fixture_seam() -> SolverSeamRecord:
    return SolverSeamRecord(
        parent_position_id="seam-parent-001",
        solver_topology_id="topology-seam-001",
        adjacent_to_certified=True,
        honest_non_exact=True,
    )


def fixture_exact_anchors(n: int = 10) -> list[ExactAnchorRecord]:
    rows: list[ExactAnchorRecord] = []
    kinds = ("race_exact", "race1w")
    for i in range(n):
        rows.append(
            ExactAnchorRecord(
                position_id=f"anchor-{i:03d}",
                exact_label_kind=kinds[i % 2],
                exact_label="Some",
                sign=1 if i % 2 == 0 else -1,
                side_to_move=i % 2,
            )
        )
    return rows
