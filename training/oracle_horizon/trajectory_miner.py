"""Pure trajectory-to-horizon-row helpers; no engine or filesystem required."""
from __future__ import annotations

from .bands import assign_band


def find_first_oracle_index(trajectory_oracle_flags: list[bool]) -> int | None:
    return next((index for index, flag in enumerate(trajectory_oracle_flags) if flag), None)


def walk_backward(n_plies: int, first_oracle_idx: int) -> list[tuple[int, int]]:
    if n_plies < 0 or not 0 <= first_oracle_idx < n_plies:
        raise ValueError("first_oracle_idx must identify an index in the trajectory")
    return [(index, first_oracle_idx - index) for index in range(first_oracle_idx, -1, -1)]


def build_horizon_row(
    *,
    plies_to_oracle_entry: int,
    oracle_wdl_stm: str | int,
    exact_dtm_or_null: int | None,
    best_move: str,
    proof_lineage_id: str,
    proof_completeness_class: str,
    book_move_used: bool = False,
    **extra: object,
) -> dict:
    row = {
        "plies_to_oracle_entry": int(plies_to_oracle_entry),
        "oracle_wdl_stm": oracle_wdl_stm,
        "exact_dtm_or_null": exact_dtm_or_null,
        "best_move": best_move,
        "proof_lineage_id": proof_lineage_id,
        "proof_completeness_class": proof_completeness_class,
        "book_move_used": bool(book_move_used),
        "band": assign_band(int(plies_to_oracle_entry)),
    }
    row.update(extra)
    return row


def reject_book_assisted(row: dict) -> dict:
    if row.get("book_move_used") is True:
        raise ValueError("book-assisted trajectory is not eligible for training ingestion")
    return row
