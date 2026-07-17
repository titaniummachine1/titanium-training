"""Teacher dataset scan audit — explains manifest vs cache row gaps."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from titanium_training.data.teacher_value import load_manifest
from titanium_training.paths import ACTIVE_TEACHER_DATASET, REPO_ROOT

PACKED_STATE_LEN = 24


def audit_teacher_dataset_scan(
    dataset_dir: Path | None = None,
    *,
    root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Reconcile manifest position count with build_feature_cache Pass-1 unique keys.

    The cache build iterates labels and joins to positions; positions without any
    label row are excluded. Labels with null value_i16 or missing position rows
    are also excluded or counted separately.
    """
    dataset_dir = dataset_dir or ACTIVE_TEACHER_DATASET
    manifest = load_manifest(dataset_dir)
    labels_rel = manifest["parts"]["labels"][0]
    positions_rel = manifest["parts"]["positions"][0]
    labels_path = (root / labels_rel).resolve()
    positions_path = (root / positions_rel).resolve()

    positions = pq.read_table(
        positions_path,
        columns=["position_key", "packed_state", "side_to_move"],
    )
    labels = pq.read_table(
        labels_path,
        columns=["position_key", "value_i16", "source_cohort"],
    )

    pos_by_key: dict[Any, tuple[bytes, int]] = {}
    invalid_packed = 0
    read_errors = 0
    for i in range(positions.num_rows):
        try:
            key = positions.column("position_key")[i].as_py()
            packed = positions.column("packed_state")[i].as_py()
            stm = int(positions.column("side_to_move")[i].as_py())
            if packed is None:
                read_errors += 1
                continue
            packed_b = bytes(packed)
            if len(packed_b) != PACKED_STATE_LEN:
                invalid_packed += 1
                continue
            pos_by_key[key] = (packed_b, stm)
        except Exception:
            read_errors += 1

    skip_reasons: Counter[str] = Counter()
    unique_label_keys: set[Any] = set()
    n_labels_scanned = 0
    duplicate_label_keys = 0
    cohort_skipped = 0

    for i in range(labels.num_rows):
        n_labels_scanned += 1
        value_i16 = labels.column("value_i16")[i].as_py()
        pos_key = labels.column("position_key")[i].as_py()
        if value_i16 is None:
            skip_reasons["label_null_value_i16"] += 1
            continue
        if pos_key not in pos_by_key:
            skip_reasons["label_missing_position_row"] += 1
            continue
        packed_b, _stm = pos_by_key[pos_key]
        if len(packed_b) != PACKED_STATE_LEN:
            skip_reasons["invalid_packed_state_length"] += 1
            continue
        if pos_key in unique_label_keys:
            duplicate_label_keys += 1
        else:
            unique_label_keys.add(pos_key)

    positions_parquet_rows = positions.num_rows
    labels_parquet_rows = labels.num_rows
    manifest_positions = int(manifest["counts"]["positions"])
    manifest_labels = int(manifest["counts"]["labels"])

    orphan_positions = len(set(pos_by_key.keys()) - unique_label_keys)
    cache_unique_estimate = len(unique_label_keys)

    explained_gap = manifest_positions - cache_unique_estimate
    reconciliation = {
        "orphan_positions_no_label": orphan_positions,
        "label_missing_position_row": skip_reasons["label_missing_position_row"],
        "label_null_value_i16": skip_reasons["label_null_value_i16"],
        "invalid_packed_state_length": skip_reasons["invalid_packed_state_length"],
        "position_read_errors": read_errors,
        "position_invalid_packed": invalid_packed,
    }
    unexplained = explained_gap - sum(
        reconciliation[k]
        for k in (
            "orphan_positions_no_label",
            "label_missing_position_row",
            "label_null_value_i16",
            "invalid_packed_state_length",
        )
    )

    return {
        "dataset": str(dataset_dir),
        "manifest_positions": manifest_positions,
        "manifest_labels": manifest_labels,
        "positions_parquet_rows": positions_parquet_rows,
        "labels_parquet_rows": labels_parquet_rows,
        "unique_position_keys_in_positions": len(pos_by_key),
        "unique_position_keys_from_labels": cache_unique_estimate,
        "manifest_minus_cache_unique": explained_gap,
        "skip_reasons": dict(skip_reasons),
        "reconciliation": reconciliation,
        "duplicate_label_keys_extra_rows": duplicate_label_keys,
        "labels_scanned": n_labels_scanned,
        "cohort_skipped": cohort_skipped,
        "unexplained_gap": unexplained,
        "cache_build_would_emit": cache_unique_estimate,
        "fully_explained": unexplained == 0,
    }


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    report = audit_teacher_dataset_scan(args.dataset)
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0 if report["fully_explained"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
