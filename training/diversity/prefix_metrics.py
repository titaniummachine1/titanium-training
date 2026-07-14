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
