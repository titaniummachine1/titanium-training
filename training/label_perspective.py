"""Label perspective for packed teacher rows vs JSON move-prefix rows.

Packed eval (`titanium_game_from_packed`) flips the numeric ``turn`` field and swaps
pawn identities relative to dataset ``side_to_move``.  That coordinate transform
preserves semantic side-to-move advantage — engine features stay engine-canonical
and dataset-STM teacher labels stay unchanged.
"""
from __future__ import annotations

LABEL_PERSPECTIVE_CONVENTION = "dataset_stm_unchanged_v1"


def value_i16_to_dataset_stm(value_i16: int) -> float:
    return float(value_i16) / 100.0


def stm_to_target_prob(value_stm: float) -> float:
    return (float(value_stm) + 1.0) / 2.0


def packed_row_target_prob(
    *,
    value_dataset_stm: float,
    engine_turn: int | None = None,
    dataset_side_to_move: int | None = None,
) -> float:
    """Packed rows: dataset-STM label unchanged (engine_turn ignored)."""
    del engine_turn, dataset_side_to_move
    return stm_to_target_prob(value_dataset_stm)


def dataset_stm_to_outcome_p0(
    value_dataset_stm: float,
    dataset_side_to_move: int,
) -> float:
    """Map dataset-STM normalized value to P0 outcome for QuoridorDataset round-trip."""
    if int(dataset_side_to_move) == 0:
        return float(value_dataset_stm)
    return -float(value_dataset_stm)


def packed_row_outcome_p0(
    *,
    value_dataset_stm: float,
    engine_turn: int | None = None,
    dataset_side_to_move: int,
) -> float:
    del engine_turn
    return dataset_stm_to_outcome_p0(value_dataset_stm, dataset_side_to_move)


def json_row_target_prob(value_dataset_stm: float) -> float:
    """Move-prefix JSON rows: turn already matches dataset STM."""
    return stm_to_target_prob(value_dataset_stm)
