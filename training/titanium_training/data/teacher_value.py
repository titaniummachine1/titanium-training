"""Promoted teacher-dataset value training samples (Parquet + eval-batch featurization)."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import pyarrow.parquet as pq

from titanium_training.models.field_planes import (
    CHOKE_P0,
    CHOKE_P1,
    CONTESTED,
    CORRIDOR_DELTA_P0,
    CORRIDOR_DELTA_P1,
    GOAL_INV_P0,
    GOAL_INV_P1,
    PATH_CROSS_P0,
    PATH_CROSS_P1,
    PAWN_FWD_P0,
    PAWN_FWD_P1,
    rec_field,
)
from titanium_training.paths import ACTIVE_MANIFEST_SHA256_DEFAULT, REPO_ROOT
from titanium_training.store.config import GAME_STORE_DB

TARGET_DEFINITION = "normalized_teacher_value_i16_to_win_prob"
LOSS_FUNCTION = "binary_cross_entropy_on_sigmoid(eval_cp/scale)"
FEATURE_SOURCE = "titanium eval-batch via game_store move-prefix index"


class TeacherFeaturizationError(RuntimeError):
    """Raised when real teacher rows cannot be featurized without synthetic fallback."""


@dataclass(frozen=True)
class TeacherValueSample:
    position_key: bytes
    packed_state: bytes
    side_to_move: int
    value_i16: int
    source_cohort: str
    move_prefix: tuple[str, ...]


def resolve_dataset_dir(path: Path, *, root: Path = REPO_ROOT) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = (root / p).resolve()
    if not (p / "manifest.json").is_file():
        raise FileNotFoundError(f"teacher dataset manifest missing: {p / 'manifest.json'}")
    return p


def load_manifest(dataset_dir: Path) -> dict[str, Any]:
    return json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))


def verify_manifest_identity(manifest: dict[str, Any], expected_sha256: str | None = None) -> None:
    expected = expected_sha256 or ACTIVE_MANIFEST_SHA256_DEFAULT
    actual = manifest.get("manifest_hash")
    if actual != expected:
        raise ValueError(
            f"teacher dataset manifest mismatch: got {actual!r}, expected {expected!r}"
        )


def teacher_value_target(value_i16: int) -> float:
    """Map normalized teacher value [-100,100] to win-probability target in [0,1]."""
    value = float(value_i16) / 100.0
    if not -1.0 <= value <= 1.0:
        raise ValueError(f"teacher value out of range: {value_i16}")
    return (value + 1.0) / 2.0


def iter_value_only_rows(
    dataset_dir: Path,
    *,
    root: Path = REPO_ROOT,
    max_scan: int | None = None,
) -> Iterator[dict[str, Any]]:
    manifest = load_manifest(dataset_dir)
    labels_rel = manifest["parts"]["labels"][0]
    positions_rel = manifest["parts"]["positions"][0]
    labels_path = (root / labels_rel).resolve()
    positions_path = (root / positions_rel).resolve()

    labels = pq.read_table(
        labels_path,
        columns=["position_key", "value_i16", "has_policy", "source_cohort"],
    )
    positions = pq.read_table(
        positions_path,
        columns=["position_key", "packed_state", "side_to_move"],
    )
    pos_by_key = {
        positions.column("position_key")[i].as_py(): i for i in range(positions.num_rows)
    }

    scanned = 0
    for i in range(labels.num_rows):
        if max_scan is not None and scanned >= max_scan:
            break
        if bool(labels.column("has_policy")[i].as_py()):
            continue
        value_i16 = labels.column("value_i16")[i].as_py()
        if value_i16 is None:
            continue
        pos_key = labels.column("position_key")[i].as_py()
        pos_i = pos_by_key.get(pos_key)
        if pos_i is None:
            continue
        scanned += 1
        yield {
            "position_key": pos_key,
            "packed_state": positions.column("packed_state")[pos_i].as_py(),
            "side_to_move": int(positions.column("side_to_move")[pos_i].as_py()),
            "value_i16": int(value_i16),
            "source_cohort": str(labels.column("source_cohort")[i].as_py() or ""),
        }


def collect_featurizable_samples(
    dataset_dir: Path,
    prefix_index: dict[bytes, tuple[str, ...]],
    *,
    max_samples: int,
    seed: int = 0,
    root: Path = REPO_ROOT,
    max_scan: int = 500_000,
) -> list[TeacherValueSample]:
    rng = random.Random(seed)
    rows = list(iter_value_only_rows(dataset_dir, root=root, max_scan=max_scan))
    rng.shuffle(rows)
    out: list[TeacherValueSample] = []
    for row in rows:
        packed = bytes(row["packed_state"])
        prefix = prefix_index.get(packed)
        if prefix is None:
            continue
        out.append(
            TeacherValueSample(
                position_key=bytes(row["position_key"]),
                packed_state=packed,
                side_to_move=int(row["side_to_move"]),
                value_i16=int(row["value_i16"]),
                source_cohort=str(row["source_cohort"]),
                move_prefix=prefix,
            )
        )
        if len(out) >= max_samples:
            break
    return out


def _eval_record_to_training_row(rec: dict[str, Any], *, outcome: float, src: str) -> dict[str, Any]:
    gi0 = rec_field(rec, GOAL_INV_P0)
    gi1 = rec_field(rec, GOAL_INV_P1)
    p0 = rec.get("pawn0", 0)
    p1 = rec.get("pawn1", 0)
    d0 = gi0[p0] if gi0 and p0 < len(gi0) else rec.get("d0", 0)
    d1 = gi1[p1] if gi1 and p1 < len(gi1) else rec.get("d1", 0)
    if "legal_wall_count" not in rec:
        raise RuntimeError("eval-batch record missing legal_wall_count")
    return {
        "_src": src,
        "ply": len(rec.get("_move_prefix", [])),
        "turn": rec.get("turn", 0),
        "outcome": float(outcome),
        "pawn0": p0,
        "pawn1": p1,
        "wl0": rec.get("wl0", 0),
        "wl1": rec.get("wl1", 0),
        "d0": d0,
        "d1": d1,
        "legal_wall_count": int(rec["legal_wall_count"]),
        GOAL_INV_P0: gi0,
        GOAL_INV_P1: gi1,
        PAWN_FWD_P0: rec_field(rec, PAWN_FWD_P0),
        PAWN_FWD_P1: rec_field(rec, PAWN_FWD_P1),
        CORRIDOR_DELTA_P0: rec_field(rec, CORRIDOR_DELTA_P0),
        CORRIDOR_DELTA_P1: rec_field(rec, CORRIDOR_DELTA_P1),
        PATH_CROSS_P0: rec_field(rec, PATH_CROSS_P0),
        PATH_CROSS_P1: rec_field(rec, PATH_CROSS_P1),
        CHOKE_P0: rec_field(rec, CHOKE_P0),
        CHOKE_P1: rec_field(rec, CHOKE_P1),
        CONTESTED: rec_field(rec, CONTESTED),
        "corridor_width0": sum(1 for v in gi0 if v == d0),
        "corridor_width1": sum(1 for v in gi1 if v == d1),
        "hw": rec.get("hw", []),
        "vw": rec.get("vw", []),
    }


def featurize_teacher_samples(
    samples: list[TeacherValueSample],
) -> list[dict[str, Any]]:
    if not samples:
        raise TeacherFeaturizationError("no teacher samples to featurize")
    from tools.datagen.datagen import eval_batch

    prefixes = [list(s.move_prefix) for s in samples]
    evals = eval_batch(prefixes)
    if len(evals) != len(samples):
        raise TeacherFeaturizationError(
            f"eval-batch count mismatch: {len(evals)} vs {len(samples)}"
        )

    records: list[dict[str, Any]] = []
    for sample, rec in zip(samples, evals):
        if int(rec.get("turn", -1)) != sample.side_to_move:
            raise TeacherFeaturizationError(
                "side_to_move mismatch between teacher parquet and eval-batch"
            )
        value = float(sample.value_i16) / 100.0
        row = _eval_record_to_training_row(
            rec,
            outcome=value,
            src=f"teacher:{sample.source_cohort}",
        )
        records.append(row)
    return records


def load_teacher_value_training_records(
    dataset_dir: Path,
    *,
    game_store_db: Path = GAME_STORE_DB,
    max_samples: int = 2000,
    min_samples: int = 64,
    seed: int = 0,
    root: Path = REPO_ROOT,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load featurized teacher-value records for HalfPW training."""
    dataset_dir = resolve_dataset_dir(dataset_dir, root=root)
    manifest = load_manifest(dataset_dir)
    verify_manifest_identity(manifest)

    from titanium_training.data.move_prefix_index import build_game_store_prefix_index

    prefix_index = build_game_store_prefix_index(game_store_db)
    samples = collect_featurizable_samples(
        dataset_dir,
        prefix_index,
        max_samples=max_samples,
        seed=seed,
        root=root,
    )
    if len(samples) < min_samples:
        raise TeacherFeaturizationError(
            f"only {len(samples)} teacher rows matched game_store move prefixes "
            f"(need >={min_samples}); pathless friend positions require engine packed-state eval"
        )

    records = featurize_teacher_samples(samples)
    meta = {
        "dataset_path": str(dataset_dir.relative_to(root)).replace("\\", "/"),
        "dataset_manifest_sha256": manifest.get("manifest_hash"),
        "game_store_index": str(game_store_db.relative_to(root)).replace("\\", "/"),
        "prefix_index_positions": len(prefix_index),
        "featurized_samples": len(records),
        "target_definition": TARGET_DEFINITION,
        "loss_function": LOSS_FUNCTION,
        "feature_source": FEATURE_SOURCE,
        "synthetic_fallback_used": False,
    }
    return records, meta
