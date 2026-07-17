#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

TRAINING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING))

from streaming_val_split import split_streaming_epoch_keys
from titanium_training.data.split import _split_bucket


def _key_in_bucket(*, seed: int, val_fraction: float, selected: bool) -> str:
    for i in range(1, 100_000):
        key = f"{i:032x}"
        if (_split_bucket(bytes.fromhex(key), seed) < val_fraction) == selected:
            return key
    raise AssertionError("could not find a deterministic split key")


def test_same_canonical_position_not_in_both_splits(tmp_path):
    labels_db = tmp_path / "labels.db"
    # use nonexistent games.db — orphan hash split path
    keys = ["json:aa", "json:bb", "json:cc", "json:dd", "json:ee"]
    train, val = split_streaming_epoch_keys(
        keys,
        labels_db=labels_db,
        val_fraction=0.2,
        seed=42,
    )
    train_hex = {k[5:] for k in train}
    val_hex = {k[5:] for k in val}
    assert not (train_hex & val_hex)
    assert len(train) + len(val) == len(keys)


def test_new_streaming_positions_use_hash_not_epoch_first_row(tmp_path):
    labels_db = tmp_path / "labels.db"
    seed = 42
    fraction = 0.2
    initial = _key_in_bucket(seed=seed, val_fraction=fraction, selected=False)
    first_non_val = _key_in_bucket(seed=seed, val_fraction=fraction, selected=False)
    hashed_val = _key_in_bucket(seed=seed, val_fraction=fraction, selected=True)
    # Avoid accidentally using the same generated key for both non-validation
    # positions, which would make the membership assertion ambiguous.
    if first_non_val == initial:
        for i in range(100_000, 200_000):
            candidate = f"{i:032x}"
            if _split_bucket(bytes.fromhex(candidate), seed) >= fraction:
                first_non_val = candidate
                break

    split_streaming_epoch_keys(
        [f"json:{initial}"], labels_db=labels_db, val_fraction=fraction, seed=seed
    )
    epoch = [f"json:{first_non_val}", f"json:{hashed_val}"]
    train, val = split_streaming_epoch_keys(
        epoch, labels_db=labels_db, val_fraction=fraction, seed=seed
    )
    train2, val2 = split_streaming_epoch_keys(
        epoch, labels_db=labels_db, val_fraction=fraction, seed=seed
    )

    assert f"json:{hashed_val}" in val
    assert f"json:{first_non_val}" in train
    assert train == train2
    assert val == val2
