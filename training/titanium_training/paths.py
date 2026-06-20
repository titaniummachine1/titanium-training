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

ACTIVE_TEACHER_DATASET = DATA_DIR / "teacher_dataset"
ACTIVE_MANIFEST_PATH = ACTIVE_TEACHER_DATASET / "manifest.json"

# Verification default for promoted v10 (read-only checks; do not rewrite manifests).
ACTIVE_MANIFEST_SHA256_DEFAULT = (
    "31a422f25a8c701ebfa72410f59fab9dff52c2717e30985a3f8e6929be007d02"
)

ENGINE_BIN = REPO_ROOT / "engine" / "target" / "release" / (
    "titanium.exe" if os.name == "nt" else "titanium"
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
