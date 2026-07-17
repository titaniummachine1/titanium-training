from __future__ import annotations

import tempfile
from pathlib import Path

from cache_val_split import save_val_manifest
from incremental_feature_cache import _assign_incremental_split


def test_incremental_split_holds_out_at_least_one_row_for_small_cache():
    with tempfile.TemporaryDirectory() as td:
        cache_dir = Path(td)
        row_keys = [f"key-{i}" for i in range(12)]

        train, val, manifest = _assign_incremental_split(cache_dir, row_keys)

        assert len(val) >= 1
        assert len(train) + len(val) == len(row_keys)
        assert set(train).isdisjoint(set(val))
        assert manifest["split_algorithm"] == "incremental_position_key_hash"
        assert (cache_dir / "val_manifest.json").is_file()


def test_incremental_split_extends_existing_manifest_for_new_keys():
    with tempfile.TemporaryDirectory() as td:
        cache_dir = Path(td)
        save_val_manifest(
            cache_dir,
            {
                "split_algorithm": "incremental_position_key_hash",
                "split_seed": 42,
                "val_fraction": 1.0,
                "val_position_keys_hex": ["aa"],
            },
        )

        train, val, manifest = _assign_incremental_split(cache_dir, ["aa", "bb"])

        assert len(train) == 0
        assert set(val.tolist()) == {0, 1}
        assert manifest["val_position_keys_hex"] == ["aa", "bb"]
