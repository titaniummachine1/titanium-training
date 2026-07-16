from __future__ import annotations

import numpy as np

from build_feature_cache import FV_LEN
from streaming_db_loader import EpochCohorts, interleave_epoch_cohorts, mirror_feature_rows_lr
from titanium_training.models.field_planes import compact_catv5_precise_vectors


MIRC = [(8 - i // 9) * 9 + (8 - i % 9) for i in range(81)]


def _record(turn: int = 0) -> dict:
    witness0 = [0] * 81
    witness1 = [0] * 81
    witness0[10] = 4
    witness1[70] = 3
    prop0 = [0] * 81
    prop1 = [0] * 81
    prop0[10] = 200
    prop1[70] = 100
    combined = [a + b for a, b in zip(prop0, prop1, strict=True)]
    return {
        "turn": turn,
        "cat_witness_p0_field": witness0,
        "cat_witness_p1_field": witness1,
        "cat_propagated_p0_field": prop0,
        "cat_propagated_p1_field": prop1,
        "cat_propagated_field": combined,
    }


def test_cat_inputs_are_five_bounded_planes_and_role_rotated() -> None:
    p0 = compact_catv5_precise_vectors(_record(0), MIRC)
    p1 = compact_catv5_precise_vectors(_record(1), MIRC)
    assert len(p0) == len(p1) == 5
    assert all(len(plane) == 81 for plane in p0 + p1)
    assert all(0.0 <= value <= 1.0 for plane in p0 + p1 for value in plane)
    assert p0[0][10] == 1.0
    assert p0[1][70] == 0.75
    assert p0[2][10] == 1.0
    assert p0[3][70] == 0.5
    assert p1[0][MIRC.index(70)] == 0.75
    assert p1[1][MIRC.index(10)] == 1.0


def test_lr_feature_mirror_is_an_involution() -> None:
    rng = np.random.default_rng(7)
    features = rng.normal(size=(4, FV_LEN)).astype(np.float32)
    features[:, 950] = [0, 10, 40, 80]
    features[:, 951] = [80, 70, 40, 0]
    features[:, 949] = [(int(cell) // 9 // 3) * 3 + (int(cell) % 9) // 3 for cell in features[:, 950]]
    mask = np.ones(4, dtype=bool)
    twice = mirror_feature_rows_lr(mirror_feature_rows_lr(features, mask), mask)
    np.testing.assert_array_equal(twice, features)


def test_cohorts_are_spread_across_every_full_batch() -> None:
    cohorts = EpochCohorts(
        fresh=[f"f{i}" for i in range(80)],
        recent=[f"r{i}" for i in range(10)],
        anchor=[f"a{i}" for i in range(10)],
    )
    keys = interleave_epoch_cohorts(cohorts, batch_size=10, seed=3)
    for start in range(0, 100, 10):
        batch = keys[start : start + 10]
        assert sum(key.startswith("a") for key in batch) == 1
        assert sum(key.startswith("r") for key in batch) == 1
        assert sum(key.startswith("f") for key in batch) == 8
