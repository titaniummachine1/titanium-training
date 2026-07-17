"""Versioned seeded-prefix diversity metrics for DIVERSITY_SPEC_V1."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from diversity.canonical import (
    CanonicalStateRow,
    DEFAULT_CANONICAL_STATE_VERSION,
    DEFAULT_GAME_RULES_VERSION,
    reflection_canonical_board,
)

PREFIX_METRIC_VERSION = "prefix-metric-v1"
# v2: hash canonical start + canonical states after each ply (not raw move strings alone).
PREFIX_METRIC_VERSION_V2 = "prefix-metric-v2-state-transitions"


@dataclass(frozen=True)
class PrefixMetricContext:
    """Seed-aware prefix measurement context."""

    prefix_metric_version: str = PREFIX_METRIC_VERSION
    start_state: CanonicalStateRow | None = None
    root_seed_id: str | None = None
    game_rules_version: str = DEFAULT_GAME_RULES_VERSION
    canonical_state_version: str = DEFAULT_CANONICAL_STATE_VERSION

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.prefix_metric_version:
            errors.append("missing prefix_metric_version")
        if self.start_state is None:
            errors.append("missing seeded start_state for prefix metric")
        if not self.root_seed_id:
            errors.append("missing root_seed_id")
        return errors


def _canonical_start_fingerprint(state: CanonicalStateRow) -> str:
    pawns, h_walls, v_walls = reflection_canonical_board(
        state.pawn_positions,
        state.horizontal_walls,
        state.vertical_walls,
    )
    payload = {
        "pawn_positions": pawns,
        "horizontal_walls": h_walls,
        "vertical_walls": v_walls,
        "wall_stocks": state.wall_stocks,
        "side_to_move": state.side_to_move,
        "game_rules_version": state.game_rules_version,
        "canonical_state_version": state.canonical_state_version,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def _transition_fingerprint(moves: tuple[str, ...]) -> str:
    blob = json.dumps(list(moves), separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def prefix2_key(
    ctx: PrefixMetricContext,
    moves: tuple[str, str],
) -> str | None:
    """Canonical two-ply prefix key including seeded start state."""
    if ctx.validate():
        return None
    assert ctx.start_state is not None
    payload = {
        "prefix_metric_version": ctx.prefix_metric_version,
        "start_state": _canonical_start_fingerprint(ctx.start_state),
        "root_seed_id": ctx.root_seed_id,
        "transition": _transition_fingerprint(moves[:2]),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def prefix4_key(
    ctx: PrefixMetricContext,
    moves: tuple[str, str, str, str],
) -> str | None:
    if ctx.validate():
        return None
    assert ctx.start_state is not None
    payload = {
        "prefix_metric_version": ctx.prefix_metric_version,
        "start_state": _canonical_start_fingerprint(ctx.start_state),
        "root_seed_id": ctx.root_seed_id,
        "transition": _transition_fingerprint(moves[:4]),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def prefix_key_from_state_transitions(
    *,
    prefix_metric_version: str,
    start_state: CanonicalStateRow,
    states_after_plies: tuple[CanonicalStateRow, ...],
) -> str | None:
    """Preferred v2 prefix key: canonical start + canonical states after ply 1..N.

    Reflection-equivalent trajectories share a key via CanonicalStateRow.canonical_key().
    Different seeds with identical algebraic move strings do NOT share a key unless
    their start + transition states match.
    Missing start or any transition state => INVALID (returns None).
    """
    if not prefix_metric_version or start_state is None:
        return None
    if not states_after_plies or any(s is None for s in states_after_plies):
        return None
    payload = {
        "prefix_metric_version": prefix_metric_version,
        "canonical_start_state_key": start_state.canonical_key(),
        "states_after": [s.canonical_key() for s in states_after_plies],
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def prefix2_key_v2(
    start_state: CanonicalStateRow,
    after_ply1: CanonicalStateRow,
    after_ply2: CanonicalStateRow,
) -> str | None:
    return prefix_key_from_state_transitions(
        prefix_metric_version=PREFIX_METRIC_VERSION_V2,
        start_state=start_state,
        states_after_plies=(after_ply1, after_ply2),
    )


def prefix4_key_v2(
    start_state: CanonicalStateRow,
    after_ply1: CanonicalStateRow,
    after_ply2: CanonicalStateRow,
    after_ply3: CanonicalStateRow,
    after_ply4: CanonicalStateRow,
) -> str | None:
    return prefix_key_from_state_transitions(
        prefix_metric_version=PREFIX_METRIC_VERSION_V2,
        start_state=start_state,
        states_after_plies=(after_ply1, after_ply2, after_ply3, after_ply4),
    )


def standard_start_state() -> CanonicalStateRow:
    """Fixture standard Quoridor opening position."""
    return CanonicalStateRow(
        pawn_positions="e2,e8",
        horizontal_walls="",
        vertical_walls="",
        wall_stocks="10,10",
        side_to_move=0,
    )


def fixture_prefix_context(*, root_seed_id: str, start_state: CanonicalStateRow | None = None) -> PrefixMetricContext:
    return PrefixMetricContext(
        start_state=start_state or standard_start_state(),
        root_seed_id=root_seed_id,
    )
