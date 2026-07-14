"""Training corpus opening gate — reject garbage, not diverse starts.

DIVERSITY_SPEC_V1 (2026-07-11): training diversity comes from seeded starts,
population cross-play, forks, and seam positions — not from locking every game
to one four-ply trunk or from move-selection temperature.

TEMPORARY_GARBAGE_FILTER_NOT_DIVERSITY_COMPLIANCE: central pawn plies 0–1 only.
This filter does NOT satisfy N_eff(2) >= 16 and must not be cited as diversity
compliance. Seeded opening centroids are required for that floor.

Deploy collapse detection (promoted weights always play e2 e8 e3 e7) lives in
``titanium_training.validation.opening_sanity`` — evaluation/deploy only.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TEMPORARY_GARBAGE_FILTER_NOT_DIVERSITY_COMPLIANCE = True

# Minimum training gate: central pawn development, not walls or off-board junk.
WHITE_OPENING_PAWNS = frozenset({"d2", "e2", "f2"})
BLACK_OPENING_PAWNS = frozenset({"d8", "e8", "f8"})
TRAINING_OPENING_MIN_PREFIX = ("e2", "e8")  # canonical centroid; not the only legal pair
MIN_SANE_PAWN_PLIES = 2

# Deploy-only collapse signature (do NOT use as a training corpus filter).
DEPLOY_COLLAPSE_OPENING = ("e2", "e8", "e3", "e7")

# Back-compat alias for reports that referenced the old four-ply training rule.
OPENING_SANITY_PREFIX = DEPLOY_COLLAPSE_OPENING

_LOG_DIR = Path(__file__).resolve().parent / "data" / "overnight_logs"
_REJECT_LOG = _LOG_DIR / "opening_sanity_rejections.jsonl"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def first_two_plies_forward_pawns(moves: list[str]) -> bool:
    """Temporary garbage filter — not diversity compliance."""
    if len(moves) < MIN_SANE_PAWN_PLIES:
        return False
    return moves[0] in WHITE_OPENING_PAWNS and moves[1] in BLACK_OPENING_PAWNS


def training_opening_ok(moves: list[str]) -> bool:
    return first_two_plies_forward_pawns(moves)


def opening_sanity_ok(moves: list[str]) -> bool:
    return training_opening_ok(moves)


def contributes_to_n_eff_two_floor() -> bool:
    """Explicit: temporary filter must not be used to claim N_eff(2) compliance."""
    return False


def log_rejected_game(payload: dict[str, Any]) -> None:
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"recorded_at": _utc_now(), **payload}
    with _REJECT_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, separators=(",", ":")) + "\n")
