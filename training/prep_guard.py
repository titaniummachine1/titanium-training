"""Global preparation-only guard for DIVERSITY_SPEC_V1 rollout.

When TRAINING_PREP_ONLY=1 (default during engine-semantics churn), all real
generation, labeling, training, gating, and deployment entry points must refuse
to run. Dry-run planners, schema validation, and synthetic tests are allowed.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import NoReturn

PREP_ENV = "TRAINING_PREP_ONLY"
DRY_RUN_LOG_DIR = Path(__file__).resolve().parent / "data" / "overnight_logs"
ALLOWED_DRY_RUN_PREFIXES = (
    "prepare_diversity_plan",
    "test_",
    "pytest",
)

# Real-work command categories blocked under prep-only mode.
BLOCKED_CATEGORIES = frozenset(
    {
        "local_self_play_pool",
        "oracle_factory",
        "corpus_generation",
        "labeling",
        "dataset_finalization",
        "optimizer_training",
        "candidate_gating",
        "deployment",
    }
)


def prep_only_enabled() -> bool:
    raw = os.environ.get(PREP_ENV, "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def refuse_real_work(
    category: str,
    *,
    argv0: str | None = None,
    detail: str | None = None,
) -> NoReturn:
    """Exit non-zero with an explicit refusal message."""
    if category not in BLOCKED_CATEGORIES:
        raise ValueError(f"unknown blocked category: {category}")
    cmd = argv0 or (sys.argv[0] if sys.argv else "<unknown>")
    msg = (
        f"REFUSED [{category}]: TRAINING_PREP_ONLY=1 blocks real work while engine "
        f"semantics are in flux. Command: {cmd}"
    )
    if detail:
        msg += f" ({detail})"
    msg += (
        f"\nAllowed now: dry-run planning, fixtures, schema validation, unit tests."
        f"\nTo run real work later: set {PREP_ENV}=0 plus launch-gate approval."
    )
    print(msg, file=sys.stderr)
    raise SystemExit(2)


def guard_real_work(category: str, *, detail: str | None = None) -> None:
    if prep_only_enabled():
        refuse_real_work(category, detail=detail)


def assert_dry_run_allowed() -> None:
    """Marker for dry-run modules that only write under overnight_logs."""
    return None


def validate_dry_run_output_path(path: Path) -> None:
    """Dry-run reports must stay under training/data/overnight_logs/."""
    try:
        path.resolve().relative_to(DRY_RUN_LOG_DIR.resolve())
    except ValueError as exc:
        raise SystemExit(
            f"dry-run output must be under {DRY_RUN_LOG_DIR}, got {path}"
        ) from exc
