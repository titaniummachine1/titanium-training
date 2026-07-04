"""Deterministic Oracle self-play matchup selection."""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any

GENERATION_SELFPLAY = "generation_selfplay"
GENERATION_MIXED = "generation_mixed"


@dataclass(frozen=True)
class MatchupChoice:
    p0_hash: str
    p1_hash: str
    kind: str
    opening_exploration: bool
    current_hash: str
    prior_hash: str | None


def deterministic_rng(seed_material: str) -> random.Random:
    digest = hashlib.sha256(seed_material.encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def choose_matchup(
    game_id: str,
    current: str,
    previous: str | None,
    *,
    same_net_fraction: float = 0.70,
) -> MatchupChoice:
    """70% current vs current; 30% current vs previous with balanced sides."""
    rng = deterministic_rng(game_id)
    cur = current.lower()
    prev = previous.lower() if previous else None

    if not prev or prev == cur or rng.random() < same_net_fraction:
        return MatchupChoice(
            p0_hash=cur,
            p1_hash=cur,
            kind=GENERATION_SELFPLAY,
            opening_exploration=True,
            current_hash=cur,
            prior_hash=prev,
        )

    if rng.random() < 0.50:
        p0, p1 = cur, prev
    else:
        p0, p1 = prev, cur

    return MatchupChoice(
        p0_hash=p0,
        p1_hash=p1,
        kind=GENERATION_MIXED,
        # Both sides use fixed, deterministic weights here -- with temperature
        # off this reduces to ONE identical game replayed forever (same input
        # -> same output every time), which is both a guaranteed duplicate
        # (chokes the Oracle result queue's FIFO -- see oracle_laptop_client.py
        # dedup handling) and useless as a strength signal: N copies of one
        # game is one data point, not N. Exploration on gives every game a
        # distinct opening derived from game_id's own RNG seed, so
        # current-vs-previous actually samples real variance.
        opening_exploration=True,
        current_hash=cur,
        prior_hash=prev,
    )


def matchup_to_payload_fields(choice: MatchupChoice) -> dict[str, Any]:
    return {
        "matchup_type": choice.kind,
        "side_weight_hashes": {"p0": choice.p0_hash, "p1": choice.p1_hash},
        "opening_exploration": choice.opening_exploration,
        "current_weight_hash": choice.current_hash,
        "prior_weight_hash": choice.prior_hash,
    }
