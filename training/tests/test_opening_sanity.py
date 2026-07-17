from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from titanium_training.validation.opening_sanity import (
    EXPECTED_OPENING,
    OpeningSanityError,
    assert_opening_sanity,
)


def test_opening_sanity_accepts_center_pawn_sequence(tmp_path: Path) -> None:
    weights = tmp_path / "net_weights.bin"
    weights.write_bytes(b"x")
    with patch(
        "titanium_training.validation.opening_sanity.opening_sequence",
        return_value=EXPECTED_OPENING,
    ):
        assert assert_opening_sanity(weights) == EXPECTED_OPENING


def test_opening_sanity_rejects_wall_first_sequence(tmp_path: Path) -> None:
    weights = tmp_path / "net_weights.bin"
    weights.write_bytes(b"x")
    with patch(
        "titanium_training.validation.opening_sanity.opening_sequence",
        return_value=("g7h", "e8", "e3", "e7"),
    ):
        with pytest.raises(OpeningSanityError, match="collapsed opening"):
            assert_opening_sanity(weights)
