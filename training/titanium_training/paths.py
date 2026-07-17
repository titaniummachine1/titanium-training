"""Repository and training path resolution (single source of truth)."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = REPO_ROOT / "training"
DATA_DIR = TRAINING_ROOT / "data"
CONFIGS_DIR = TRAINING_ROOT / "configs"
RUNS_DIR = TRAINING_ROOT / "runs"
CHECKPOINTS_DIR = TRAINING_ROOT / "checkpoints"
_GOOD_TEACHER = DATA_DIR / "teacher_dataset_good"
_CANDIDATE_V9_TEACHER = DATA_DIR / "teacher_dataset_candidate_v9"
_LEGACY_TEACHER = DATA_DIR / "teacher_dataset"
_ACTIVE_TEACHER_OVERRIDE = os.environ.get("TITANIUM_ACTIVE_TEACHER_DATASET")
ACTIVE_TEACHER_DATASET = Path(_ACTIVE_TEACHER_OVERRIDE) if _ACTIVE_TEACHER_OVERRIDE else (
    _GOOD_TEACHER
    if (_GOOD_TEACHER / "manifest.json").is_file()
    else (
        _CANDIDATE_V9_TEACHER
        if (_CANDIDATE_V9_TEACHER / "manifest.json").is_file()
        else _LEGACY_TEACHER
    )
)
ACTIVE_MANIFEST_PATH = ACTIVE_TEACHER_DATASET / "manifest.json"

# Verification default for promoted v10 (read-only checks; do not rewrite manifests).
ACTIVE_MANIFEST_SHA256_DEFAULT = (
    "810fe8c5db540447aafd89399c5dcc3d8916ec800ade5bc97759a1bfd45bb08d"
)

ENGINE_BIN = Path(
    os.environ.get(
        "TITANIUM_ENGINE_BIN",
        str(
            REPO_ROOT
            / "engine"
            / "target"
            / "release"
            / ("titanium.exe" if os.name == "nt" else "titanium")
        ),
    )
)
WEIGHTS_BIN = REPO_ROOT / "engine" / "src" / "titanium" / "net_weights.bin"

# Re-export canonical store paths (defined in store.config after relocation).
from titanium_training.store.config import (  # noqa: E402
    GAME_STORE_DB,
    REPORT_DIR,
    ROOT,
    TEACHER_STORE_DB,
)

__all__ = [
    "REPO_ROOT",
    "TRAINING_ROOT",
    "DATA_DIR",
    "CONFIGS_DIR",
    "RUNS_DIR",
    "CHECKPOINTS_DIR",
    "ACTIVE_TEACHER_DATASET",
    "ACTIVE_MANIFEST_PATH",
    "ACTIVE_MANIFEST_SHA256_DEFAULT",
    "ENGINE_BIN",
    "WEIGHTS_BIN",
    "GAME_STORE_DB",
    "REPORT_DIR",
    "ROOT",
    "TEACHER_STORE_DB",
]
