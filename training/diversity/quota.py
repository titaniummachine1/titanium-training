"""Constitutional quota floors and ERR-MAP fallback allocation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from diversity.lanes import DiversityLane

# Exact constitutional floors (fractions of total rows).
QUOTA_CLOSED_LOOP_POP = 0.30
QUOTA_BEHAVIORAL_CROSSPLAY = 0.15
QUOTA_FORKS = 0.10
QUOTA_SOLVER_SEAM = 0.10
QUOTA_EXACT_ANCHORS = 0.05
QUOTA_ADAPTIVE_RESIDUAL = 0.30

QUOTA_TOLERANCE_PP = 0.005  # ±0.5 percentage points

STATIC_RESIDUAL_TARGETS = (
    DiversityLane.CLOSED_LOOP_POPULATION,
    DiversityLane.BEHAVIORAL_CROSSPLAY,
    DiversityLane.PAIRED_FORKS,
)

BASE_STATIC_SHARES: dict[DiversityLane, float] = {
    DiversityLane.CLOSED_LOOP_POPULATION: QUOTA_CLOSED_LOOP_POP,
    DiversityLane.BEHAVIORAL_CROSSPLAY: QUOTA_BEHAVIORAL_CROSSPLAY,
    DiversityLane.PAIRED_FORKS: QUOTA_FORKS,
    DiversityLane.SOLVER_SEAM: QUOTA_SOLVER_SEAM,
    DiversityLane.EXACT_ANCHORS: QUOTA_EXACT_ANCHORS,
}


@dataclass(frozen=True)
class QuotaPlan:
    total_rows: int
    per_lane: dict[DiversityLane, int]
    per_cell: dict[str, int]
    stm_white: int
    stm_black: int
    err_map_validated: bool
    blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_rows": self.total_rows,
            "per_lane": {k.value: v for k, v in self.per_lane.items()},
            "per_cell": self.per_cell,
            "stm_distribution": {"white": self.stm_white, "black": self.stm_black},
            "err_map_validated": self.err_map_validated,
            "blockers": list(self.blockers),
        }


def _round_shares(total: int, shares: dict[DiversityLane, float]) -> dict[DiversityLane, int]:
    raw = {lane: total * frac for lane, frac in shares.items()}
    floors = {lane: int(v) for lane, v in raw.items()}
    assigned = sum(floors.values())
    remainder = total - assigned
    order = sorted(shares.keys(), key=lambda lane: raw[lane] - floors[lane], reverse=True)
    i = 0
    while remainder > 0 and order:
        floors[order[i % len(order)]] += 1
        remainder -= 1
        i += 1
    return floors


def static_residual_shares(err_map_validated: bool = False) -> dict[DiversityLane, float]:
    if err_map_validated:
        raise NotImplementedError("ERR-MAP adaptive residual not enabled in prep phase")
    residual = QUOTA_ADAPTIVE_RESIDUAL
    base_sum = sum(BASE_STATIC_SHARES[lane] for lane in STATIC_RESIDUAL_TARGETS)
    boosted = {lane: BASE_STATIC_SHARES[lane] for lane in BASE_STATIC_SHARES}
    for lane in STATIC_RESIDUAL_TARGETS:
        boosted[lane] += residual * (BASE_STATIC_SHARES[lane] / base_sum)
    return boosted


def allocate_quota_rows(
    total_rows: int,
    *,
    err_map_validated: bool = False,
) -> QuotaPlan:
    shares = static_residual_shares(err_map_validated=err_map_validated)
    per_lane = _round_shares(total_rows, shares)
    per_cell: dict[str, int] = {}
    for lane, count in per_lane.items():
        if lane == DiversityLane.PAIRED_FORKS:
            per_cell["forks.regret_mined"] = count // 2
            per_cell["forks.plausible_deviation"] = count - count // 2
        else:
            per_cell[lane.value] = count
    stm_white = total_rows // 2
    stm_black = total_rows - stm_white
    blockers: list[str] = []
    if not err_map_validated:
        blockers.append("ERR-MAP not validated: adaptive residual uses static round-robin")
    blockers.append("seeded opening centroid bank not generated (fixture-only)")
    blockers.append("real solver seam / exact anchor tables not generated")
    return QuotaPlan(
        total_rows=total_rows,
        per_lane=per_lane,
        per_cell=per_cell,
        stm_white=stm_white,
        stm_black=stm_black,
        err_map_validated=err_map_validated,
        blockers=tuple(blockers),
    )


def validate_quota_shares(per_lane: dict[DiversityLane, int], total: int) -> list[str]:
    if total <= 0:
        return ["total rows must be positive"]
    errors: list[str] = []
    expected = static_residual_shares(err_map_validated=False)
    for lane, target_frac in expected.items():
        actual = per_lane.get(lane, 0) / total
        if abs(actual - target_frac) > QUOTA_TOLERANCE_PP + 1e-9:
            errors.append(
                f"{lane.value}: share {actual:.4f} outside ±{QUOTA_TOLERANCE_PP:.3f} of {target_frac:.4f}"
            )
    return errors
