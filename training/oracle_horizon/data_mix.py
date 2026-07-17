"""Conservative source and provenance caps for pilot sampling."""
from __future__ import annotations


def pilot_mix() -> dict[str, float]:
    return {"general": 0.80, "oracle_horizon": 0.10, "anchors": 0.10}


def caps(
    *,
    source_game: int | str | None = None,
    proof_lineage: int | str | None = None,
    phase: int | str | None = None,
    wall_stock: int | str | None = None,
    stm: int | str | None = None,
    win_loss: int | str | None = None,
) -> dict[str, int | str | None]:
    return {
        "source_game": source_game, "proof_lineage": proof_lineage,
        "phase": phase, "wall_stock": wall_stock, "stm": stm, "win_loss": win_loss,
    }
