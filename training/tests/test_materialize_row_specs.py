"""Regression tests for featurize spec materialization contract."""
from __future__ import annotations

from build_feature_cache import (
    FEAT_META_RECORD,
    PACKED_ROW_BYTES,
    _featurize_spec_paths,
    materialize_row_specs_to_disk,
    read_chunk_sidecar,
)


def sample_specs() -> list[tuple[bytes, bytes, float, int, int]]:
    return [
        (b"k1", b"\x00" * PACKED_ROW_BYTES, 0.5, 0, 2),
        (b"key2long", b"\x01" * PACKED_ROW_BYTES, 0.25, 1, 1),
    ]


def test_materialize_row_specs_contract(tmp_path):
    specs = sample_specs()
    count = materialize_row_specs_to_disk(tmp_path, specs)

    assert isinstance(count, int)
    assert count == len(specs)

    packed, meta, keys, keys_idx = _featurize_spec_paths(tmp_path)

    assert packed.exists()
    assert meta.exists()
    assert keys.exists()
    assert keys_idx.exists()
    assert packed.stat().st_size == count * PACKED_ROW_BYTES
    assert meta.stat().st_size == count * FEAT_META_RECORD.size

    side = read_chunk_sidecar(meta, keys, keys_idx, 0, count)
    assert len(side) == count
    assert side[0][0] == b"k1"
    assert side[1][3] == 1


def test_materialize_returns_int_not_paths(tmp_path):
    """Caller must not unpack the return value as paths."""
    count = materialize_row_specs_to_disk(tmp_path, sample_specs())
    assert type(count) is int
    packed, *_ = _featurize_spec_paths(tmp_path)
    assert packed.is_file()
