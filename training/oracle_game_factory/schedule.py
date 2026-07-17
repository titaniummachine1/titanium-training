"""Deterministic 1024-game current/prior self-play schedule."""
from __future__ import annotations

import random
from dataclasses import dataclass


CURRENT_CURRENT = "current_current"
CURRENT_PRIOR_P0 = "current_p0_prior_p1"
PRIOR_CURRENT_P0 = "prior_p0_current_p1"


@dataclass(frozen=True)
class ScheduledGame:
    index: int
    matchup_type: str
    seed: int
    current_is_p0: bool


def make_schedule(
    *,
    generation_seed: int,
    group_size: int = 1024,
    has_distinct_prior: bool = True,
) -> list[ScheduledGame]:
    """Return exact 717/154/153 mixture for a 1024-game group.

    If there is no distinct prior generation, all games become current/current
    and current/prior comparison must be logged as unavailable.
    """
    if group_size != 1024:
        raise ValueError("Oracle training schedule is defined for exactly 1024 games")

    entries: list[tuple[str, bool]] = [(CURRENT_CURRENT, True)] * 717
    if has_distinct_prior:
        entries.extend([(CURRENT_PRIOR_P0, True)] * 154)
        entries.extend([(PRIOR_CURRENT_P0, False)] * 153)
    else:
        entries.extend([(CURRENT_CURRENT, True)] * (154 + 153))

    rng = random.Random(generation_seed)
    rng.shuffle(entries)
    seeds = rng.sample(range(1, 2**63 - 1), group_size)
    return [
        ScheduledGame(index=i, matchup_type=kind, seed=seeds[i], current_is_p0=current_is_p0)
        for i, (kind, current_is_p0) in enumerate(entries)
    ]


def schedule_counts(schedule: list[ScheduledGame]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for game in schedule:
        counts[game.matchup_type] = counts.get(game.matchup_type, 0) + 1
    return counts

