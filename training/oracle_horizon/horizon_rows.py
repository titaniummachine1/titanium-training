"""Pure helpers for constructing Cycle 1 horizon labels."""
from __future__ import annotations

from typing import Any


def ply_move(
    *,
    index: int,
    oracle_index: int,
    oracle_move: str | None,
    deploy_move: str | None,
    proven_move: str | None,
    exact: bool,
) -> str | None:
    """Return the move belonging to this ply, never an unrelated entry move."""
    if index == oracle_index:
        return oracle_move
    if proven_move is not None:
        return proven_move
    if exact:
        return deploy_move
    return None


def initial_target_move(*, index: int, oracle_index: int, oracle_move: str | None) -> str | None:
    """Only the exact entry has an oracle move before ancestor proofing."""
    return oracle_move if index == oracle_index else None


def build_horizon_row(
    *,
    position_id: str,
    game_id: str,
    lineage_id: str,
    index: int,
    oracle_index: int,
    band: int,
    label_class: str,
    primary: bool,
    oracle_wdl: str,
    oracle_proven: bool,
    backed_proven: bool,
    deploy: dict[str, Any],
    best_move: str | None,
    needs_learning_value: bool,
    needs_learning_reasons: list[str],
    ladder: list[dict],
    weights_sha256: str,
    engine_sha256: str,
    packed_state_hex: str,
) -> dict:
    """Build one serializable label row from already-computed evidence."""
    return {
        "position_id": position_id,
        "packed_state_hex": packed_state_hex,
        "game_id": game_id,
        "lineage_id": lineage_id,
        "ply": index,
        "plies_to_oracle_entry": oracle_index - index,
        "band": band,
        "label_class": label_class,
        "primary": primary,
        "oracle_wdl": oracle_wdl,
        "oracle_proven": oracle_proven,
        "backed_proven": backed_proven,
        "score": deploy.get("score"),
        "selected_move": best_move,
        "best_move": best_move,
        "needs_learning": needs_learning_value,
        "needs_learning_reasons": needs_learning_reasons,
        "ladder": ladder,
        "weights_sha256": weights_sha256,
        "engine_sha256": engine_sha256,
        "book_move_used": False,
        "evaluation_only": False,
    }
