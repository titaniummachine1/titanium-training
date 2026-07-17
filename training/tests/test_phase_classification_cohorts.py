"""Phase classification and cohort phase-coverage tests for CATv5 training."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from canonical_sampling import (
    ENDGAME_BUCKET_FRAC,
    MIDGAME_BUCKET_FRAC,
    OPENING_BUCKET_FRAC,
    select_phase_balanced,
)
from label_weights import game_phase_from_packed, game_phase_from_record, game_phase_from_walls
from streaming_db_loader import EpochCohorts, interleave_epoch_cohorts


def _packed(wl0: int, wl1: int) -> bytes:
    # schema version + cells + walls + stm + pad + empty wall masks
    head = bytes([1, 4, 76, wl0, wl1, 0, 0, 0])
    return head + (0).to_bytes(8, "little") + (0).to_bytes(8, "little")


def test_game_phase_from_walls_thresholds() -> None:
    assert game_phase_from_walls(10, 10) == "opening"
    assert game_phase_from_walls(7, 6) == "opening"  # placed = 7
    assert game_phase_from_walls(6, 6) == "midgame"  # placed = 8
    assert game_phase_from_walls(2, 8) == "endgame"
    assert game_phase_from_walls(9, 1) == "endgame"


def test_game_phase_from_packed_matches_record() -> None:
    packed = _packed(9, 9)
    assert game_phase_from_packed(packed) == "opening"
    assert game_phase_from_record({"wl0": 9, "wl1": 9}) == "opening"
    packed_end = _packed(2, 5)
    assert game_phase_from_packed(packed_end) == "endgame"
    assert game_phase_from_record({"wl0": 2, "wl1": 5}) == "endgame"


def test_select_phase_balanced_preserves_count_and_covers_phases() -> None:
    import tempfile

    with tempfile.TemporaryDirectory(prefix="phase_bal_") as tmp:
        db = Path(tmp) / "labels.db"
        con = sqlite3.connect(str(db))
        con.executescript(
            """
            CREATE TABLE teacher_positions(
                position_key BLOB PRIMARY KEY,
                packed_state BLOB NOT NULL,
                side_to_move INTEGER NOT NULL
            );
            CREATE TABLE positions(pos_key TEXT PRIMARY KEY, position_data BLOB);
            CREATE TABLE labels(pos_key TEXT, source TEXT, value_stm REAL, n_samples INTEGER);
            """
        )
        keys: list[str] = []
        # Enough per phase that 28/47/25 quotas can be filled without leftovers.
        for i, (wl0, wl1) in enumerate([(10, 10)] * 60 + [(5, 5)] * 60 + [(2, 8)] * 60):
            pk = i.to_bytes(16, "little")
            con.execute(
                "INSERT INTO teacher_positions(position_key, packed_state, side_to_move) VALUES (?,?,0)",
                (pk, _packed(wl0, wl1)),
            )
            keys.append("teacher:" + pk.hex())
        con.commit()
        con.close()

        picked = select_phase_balanced(keys, db, count=100, seed=7)
        assert len(picked) == 100
        assert len(set(picked)) == 100

        con = sqlite3.connect(str(db))
        phases = []
        for key in picked:
            pk = bytes.fromhex(key[8:])
            packed = con.execute(
                "SELECT packed_state FROM teacher_positions WHERE position_key=?",
                (pk,),
            ).fetchone()[0]
            phases.append(game_phase_from_packed(bytes(packed)))
        con.close()
        counts = {p: phases.count(p) for p in ("opening", "midgame", "endgame")}
        n_open = int(round(100 * OPENING_BUCKET_FRAC))
        n_mid = int(round(100 * MIDGAME_BUCKET_FRAC))
        n_end = 100 - n_open - n_mid
        assert counts["opening"] == n_open
        assert counts["midgame"] == n_mid
        assert counts["endgame"] == n_end
        assert abs(ENDGAME_BUCKET_FRAC - n_end / 100.0) < 0.02


def test_epoch_phase_coverage_inside_cohort_interleave() -> None:
    """Per-batch cohort composition stays exact; phases may vary by batch."""
    cohorts = EpochCohorts(
        fresh=[f"f{i}" for i in range(80)],
        recent=[f"r{i}" for i in range(10)],
        anchor=[f"a{i}" for i in range(10)],
    )
    keys = interleave_epoch_cohorts(cohorts, batch_size=10, seed=3)
    assert len(keys) == 100
    for start in range(0, 100, 10):
        batch = keys[start : start + 10]
        assert sum(k.startswith("a") for k in batch) == 1
        assert sum(k.startswith("r") for k in batch) == 1
        assert sum(k.startswith("f") for k in batch) == 8


def test_interleave_exact_for_batch_512() -> None:
    cohorts = EpochCohorts(
        fresh=[f"f{i}" for i in range(960)],
        recent=[f"r{i}" for i in range(120)],
        anchor=[f"a{i}" for i in range(120)],
    )
    keys = interleave_epoch_cohorts(cohorts, batch_size=512, seed=9)
    assert len(keys) == 1200
    batch = keys[:512]
    assert abs(sum(k.startswith("f") for k in batch) / 512 - 0.8) <= 1 / 512 + 1e-12
    assert abs(sum(k.startswith("r") for k in batch) / 512 - 0.1) <= 1 / 512 + 1e-12
    assert abs(sum(k.startswith("a") for k in batch) / 512 - 0.1) <= 1 / 512 + 1e-12
