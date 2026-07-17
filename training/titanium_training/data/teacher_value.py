"""Promoted teacher-dataset value training samples (Parquet + eval-packed-batch)."""
from __future__ import annotations

import json
import random
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

_TRAINING = Path(__file__).resolve().parents[2]
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

import pyarrow.parquet as pq

from titanium_training.data.eval_packed import FEATURE_SCHEMA, eval_packed_batch_allow_errors
from label_perspective import dataset_stm_to_outcome_p0, value_i16_to_dataset_stm
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
from titanium_training.store.state import POSITION_SCHEMA_VERSION, PositionState

TARGET_DEFINITION = "normalized_teacher_value_i16_to_win_prob"
LOSS_FUNCTION = "binary_cross_entropy_on_sigmoid(eval_cp/scale)"
FEATURIZATION_MODE = "packed-state-direct"
FEATURE_SOURCE = "titanium eval-packed-batch on canonical packed_state"
DEFAULT_COVERAGE_MIN = 0.999
PACKED_STATE_LEN = 24


class TeacherFeaturizationError(RuntimeError):
    """Raised when real teacher rows cannot be featurized without synthetic fallback."""


class TeacherCoverageError(TeacherFeaturizationError):
    """Raised when featurization coverage is below the required gate."""


@dataclass
class FeaturizationStats:
    candidate_labels: int = 0
    positions_requested: int = 0
    successfully_featurized: int = 0
    decode_failures: int = 0
    missing_positions: int = 0
    duplicate_positions: int = 0
    side_to_move_mismatch: int = 0
    filtered_records: int = 0
    failure_categories: dict[str, int] = field(default_factory=dict)

    @property
    def coverage_percentage(self) -> float:
        if self.positions_requested <= 0:
            return 0.0
        return 100.0 * self.successfully_featurized / self.positions_requested

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_labels": self.candidate_labels,
            "positions_requested": self.positions_requested,
            "successfully_featurized": self.successfully_featurized,
            "decode_failures": self.decode_failures,
            "missing_positions": self.missing_positions,
            "duplicate_positions": self.duplicate_positions,
            "side_to_move_mismatch": self.side_to_move_mismatch,
            "filtered_records": self.filtered_records,
            "coverage_percentage": round(self.coverage_percentage, 4),
            "failure_categories": dict(self.failure_categories),
        }


@dataclass(frozen=True)
class TeacherValueSample:
    position_key: bytes
    packed_state: bytes
    side_to_move: int
    value_i16: int
    source_cohort: str


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


def engine_commit_identity() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT / "engine"), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


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
        columns=["position_key", "value_i16", "source_cohort"],
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
        value_i16 = labels.column("value_i16")[i].as_py()
        if value_i16 is None:
            continue
        pos_key = labels.column("position_key")[i].as_py()
        pos_i = pos_by_key.get(pos_key)
        if pos_i is None:
            yield {"_missing_position": True, "position_key": pos_key}
            continue
        scanned += 1
        yield {
            "position_key": pos_key,
            "packed_state": positions.column("packed_state")[pos_i].as_py(),
            "side_to_move": int(positions.column("side_to_move")[pos_i].as_py()),
            "value_i16": int(value_i16),
            "source_cohort": str(labels.column("source_cohort")[i].as_py() or ""),
        }


def _validate_packed_python(packed: bytes) -> None:
    PositionState.unpack_state(packed)


def _eval_record_to_training_row(
    rec: dict[str, Any],
    *,
    outcome: float,
    src: str,
    position_key: bytes,
) -> dict[str, Any]:
    gi0 = rec_field(rec, GOAL_INV_P0)
    gi1 = rec_field(rec, GOAL_INV_P1)
    p0 = rec.get("pawn0", 0)
    p1 = rec.get("pawn1", 0)
    d0 = gi0[p0] if gi0 and p0 < len(gi0) else rec.get("d0", 0)
    d1 = gi1[p1] if gi1 and p1 < len(gi1) else rec.get("d1", 0)
    if "legal_wall_count" not in rec:
        raise RuntimeError("eval record missing legal_wall_count")
    return {
        "_src": src,
        "_position_key": position_key,
        "ply": 0,
        "turn": rec.get("turn", 0),
        "outcome": float(outcome),
        "pawn0": p0,
        "pawn1": p1,
        "wl0": rec.get("wl0", 0),
        "wl1": rec.get("wl1", 0),
        "d0": d0,
        "d1": d1,
        "legal_wall_count": int(rec["legal_wall_count"]),
        "legal_path_cross_p0": int(rec.get("legal_path_cross_p0", 0)),
        "legal_path_cross_p1": int(rec.get("legal_path_cross_p1", 0)),
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


def featurize_packed_samples(
    samples: list[TeacherValueSample],
    *,
    batch_size: int = 256,
    stats: FeaturizationStats | None = None,
) -> list[dict[str, Any]]:
    if not samples:
        raise TeacherFeaturizationError("no teacher samples to featurize")
    st = stats or FeaturizationStats()
    records: list[dict[str, Any]] = []
    for start in range(0, len(samples), batch_size):
        chunk = samples[start : start + batch_size]
        st.positions_requested += len(chunk)
        items = [(i, bytes(s.packed_state)) for i, s in enumerate(chunk)]
        evals = eval_packed_batch_allow_errors(items)
        for sample, rec in zip(chunk, evals):
            if not rec.get("ok", False):
                st.decode_failures += 1
                err = str(rec.get("error", "unknown"))
                st.failure_categories[err] = st.failure_categories.get(err, 0) + 1
                continue
            if int(rec.get("turn", -1)) not in (0, 1):
                st.side_to_move_mismatch += 1
                continue
            value_dataset = value_i16_to_dataset_stm(int(sample.value_i16))
            value_p0 = dataset_stm_to_outcome_p0(
                value_dataset,
                int(sample.side_to_move),
            )
            row = _eval_record_to_training_row(
                rec,
                outcome=value_p0,
                src=f"teacher:{sample.source_cohort}",
                position_key=bytes(sample.position_key),
            )
            records.append(row)
            st.successfully_featurized += 1
    return records


def collect_teacher_samples(
    dataset_dir: Path,
    *,
    max_samples: int,
    seed: int = 0,
    root: Path = REPO_ROOT,
    max_scan: int | None = None,
    stats: FeaturizationStats | None = None,
) -> list[TeacherValueSample]:
    rng = random.Random(seed)
    st = stats or FeaturizationStats()
    rows = [r for r in iter_value_only_rows(dataset_dir, root=root, max_scan=max_scan)]
    st.candidate_labels = len(rows)
    st.missing_positions = sum(1 for r in rows if r.get("_missing_position"))
    candidates = [r for r in rows if not r.get("_missing_position")]
    rng.shuffle(candidates)
    seen_packed: set[bytes] = set()
    out: list[TeacherValueSample] = []
    for row in candidates:
        packed = bytes(row["packed_state"])
        if packed in seen_packed:
            st.duplicate_positions += 1
            continue
        seen_packed.add(packed)
        try:
            _validate_packed_python(packed)
        except ValueError as e:
            st.filtered_records += 1
            st.failure_categories[str(e)] = st.failure_categories.get(str(e), 0) + 1
            continue
        out.append(
            TeacherValueSample(
                position_key=bytes(row["position_key"]),
                packed_state=packed,
                side_to_move=int(row["side_to_move"]),
                value_i16=int(row["value_i16"]),
                source_cohort=str(row["source_cohort"]),
            )
        )
        if len(out) >= max_samples:
            break
    return out


def scan_packed_state_coverage(
    dataset_dir: Path,
    *,
    max_scan: int = 50_000,
    batch_size: int = 512,
    root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Bounded coverage scan without training."""
    stats = FeaturizationStats()
    samples = collect_teacher_samples(
        dataset_dir,
        max_samples=max_scan,
        seed=0,
        root=root,
        max_scan=max_scan,
        stats=stats,
    )
    featurize_packed_samples(samples, batch_size=batch_size, stats=stats)
    return stats.to_dict()


def load_teacher_value_training_records(
    dataset_dir: Path,
    *,
    max_samples: int = 2000,
    min_samples: int = 64,
    seed: int = 0,
    root: Path = REPO_ROOT,
    batch_size: int = 512,
    coverage_min: float | None = DEFAULT_COVERAGE_MIN,
    require_full_coverage: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load featurized teacher-value records via direct packed-state eval."""
    dataset_dir = resolve_dataset_dir(dataset_dir, root=root)
    manifest = load_manifest(dataset_dir)
    verify_manifest_identity(manifest)

    stats = FeaturizationStats()
    samples = collect_teacher_samples(
        dataset_dir,
        max_samples=max_samples,
        seed=seed,
        root=root,
        max_scan=None if require_full_coverage else max(max_samples * 4, 50_000),
        stats=stats,
    )
    records = featurize_packed_samples(samples, batch_size=batch_size, stats=stats)

    if len(records) < min_samples:
        raise TeacherFeaturizationError(
            f"only {len(records)} teacher rows featurized (need >={min_samples}); "
            f"stats={stats.to_dict()}"
        )

    if coverage_min is not None and stats.positions_requested > 0:
        ratio = stats.successfully_featurized / stats.positions_requested
        if ratio + 1e-9 < coverage_min:
            raise TeacherCoverageError(
                f"coverage {ratio:.4%} below gate {coverage_min:.4%}; stats={stats.to_dict()}"
            )

    meta = {
        "dataset_path": str(dataset_dir.relative_to(root)).replace("\\", "/"),
        "dataset_manifest_sha256": manifest.get("manifest_hash"),
        "featurization_mode": FEATURIZATION_MODE,
        "engine": "titanium",
        "engine_version": "v15",
        "engine_commit": engine_commit_identity(),
        "feature_schema": FEATURE_SCHEMA,
        "position_schema_version": POSITION_SCHEMA_VERSION,
        "packed_state_bytes": PACKED_STATE_LEN,
        "target_definition": TARGET_DEFINITION,
        "loss_function": LOSS_FUNCTION,
        "feature_source": FEATURE_SOURCE,
        "synthetic_fallback_used": False,
        **stats.to_dict(),
        "featurized_samples": len(records),
    }
    return records, meta


def featurize_via_move_prefix(
    samples_with_prefix: list[tuple[TeacherValueSample, tuple[str, ...]]],
) -> list[dict[str, Any]]:
    """Diagnostic path: featurize via move-prefix eval-batch (equivalence tests)."""
    from tools.datagen.datagen import eval_batch

    prefixes = [list(pfx) for _, pfx in samples_with_prefix]
    evals = eval_batch(prefixes)
    records: list[dict[str, Any]] = []
    for (sample, _), rec in zip(samples_with_prefix, evals):
        value = float(sample.value_i16) / 100.0
        records.append(
            _eval_record_to_training_row(
                rec,
                outcome=value,
                src=f"teacher:{sample.source_cohort}",
                position_key=bytes(sample.position_key),
            )
        )
    return records
