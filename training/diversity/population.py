"""Population and style-variant descriptors (fixture eligibility checks)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

STYLE_DISAGREE_THRESHOLD = 0.05


@dataclass(frozen=True)
class PopulationDescriptor:
    population_id: str
    role: str  # current | frontier | history_1 | history_2


@dataclass(frozen=True)
class StyleVariant:
    style_id: str
    disagree_rate: float
    panel_id: str = "frozen_10k_style_eligibility_panel"

    def eligible(self) -> bool:
        return self.disagree_rate >= STYLE_DISAGREE_THRESHOLD

    def validate(self) -> list[str]:
        if self.disagree_rate < STYLE_DISAGREE_THRESHOLD:
            return [
                f"{self.style_id}: disagree rate {self.disagree_rate:.3f} < {STYLE_DISAGREE_THRESHOLD}"
            ]
        return []


def default_population_descriptors() -> tuple[PopulationDescriptor, ...]:
    return (
        PopulationDescriptor("current", "current"),
        PopulationDescriptor("frontier", "frontier"),
        PopulationDescriptor("history_n1", "history_1"),
        PopulationDescriptor("history_n2", "history_2"),
    )


def fixture_style_variants() -> tuple[StyleVariant, ...]:
    return (
        StyleVariant("style_a", 0.07),
        StyleVariant("style_b", 0.06),
        StyleVariant("style_c", 0.08),
        StyleVariant("style_reject", 0.02),
    )


def eligible_styles(variants: Iterable[StyleVariant]) -> list[StyleVariant]:
    return [v for v in variants if v.eligible()]
