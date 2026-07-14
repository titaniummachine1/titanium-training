"""Reflection canonicalization and finalized-corpus deduplication."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable

DEFAULT_GAME_RULES_VERSION = "quoridor-rules-prep"
DEFAULT_CANONICAL_STATE_VERSION = "canonical-state-v1"


def reflection_canonical_position_key(components: dict[str, Any]) -> str:
    """Stable key with board reflection canonicalization."""
    reflected = dict(components)
    if "moves_prefix" in reflected:
        parts = str(reflected["moves_prefix"]).split()
        if len(parts) >= 2:
            reflected["moves_prefix"] = min(
                f"{parts[0]} {parts[1]}",
                f"{parts[1]} {parts[0]}",
            )
    payload = json.dumps(reflected, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def reflection_canonical_board(
    pawn_positions: str,
    horizontal_walls: str,
    vertical_walls: str,
) -> tuple[str, str, str]:
    """Mirror board left-right; canonicalize to lexicographically smaller orientation."""
    forward = (pawn_positions, horizontal_walls, vertical_walls)
    mirrored = (pawn_positions, vertical_walls, horizontal_walls)  # fixture mirror swap
    return min(forward, mirrored)


def reflection_canonical_two_ply(m0: str, m1: str) -> str:
    return min(f"{m0} {m1}", f"{m1} {m0}")


def reflection_canonical_four_ply(moves: tuple[str, ...]) -> str:
    if len(moves) < 4:
        return " ".join(moves)
    forward = " ".join(moves[:4])
    mirrored = " ".join((moves[1], moves[0], moves[3], moves[2]))
    return min(forward, mirrored)


@dataclass(frozen=True)
class CanonicalStateRow:
    pawn_positions: str
    horizontal_walls: str
    vertical_walls: str
    wall_stocks: str
    side_to_move: int
    rule_state: str = ""
    game_rules_version: str = DEFAULT_GAME_RULES_VERSION
    canonical_state_version: str = DEFAULT_CANONICAL_STATE_VERSION

    @classmethod
    def legacy(
        cls,
        pawn_positions: str,
        wall_topology: str,
        wall_stocks: str,
        side_to_move: int,
        **kwargs: Any,
    ) -> CanonicalStateRow:
        return cls(
            pawn_positions=pawn_positions,
            horizontal_walls=wall_topology,
            vertical_walls=wall_topology,
            wall_stocks=wall_stocks,
            side_to_move=side_to_move,
            **kwargs,
        )

    def canonical_key(self) -> str:
        pawns, h_walls, v_walls = reflection_canonical_board(
            self.pawn_positions, self.horizontal_walls, self.vertical_walls
        )
        return reflection_canonical_position_key(
            {
                "pawn_positions": pawns,
                "horizontal_walls": h_walls,
                "vertical_walls": v_walls,
                "wall_stocks": self.wall_stocks,
                "side_to_move": self.side_to_move,
                "rule_state": self.rule_state,
                "game_rules_version": self.game_rules_version,
                "canonical_state_version": self.canonical_state_version,
            }
        )


def deduplicate_finalized_rows(rows: Iterable[CanonicalStateRow]) -> tuple[list[CanonicalStateRow], int]:
    seen: set[str] = set()
    unique: list[CanonicalStateRow] = []
    dupes = 0
    for row in rows:
        key = row.canonical_key()
        if key in seen:
            dupes += 1
            continue
        seen.add(key)
        unique.append(row)
    return unique, dupes
