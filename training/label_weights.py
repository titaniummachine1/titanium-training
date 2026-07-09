"""Per-position training weights: position frequency × source reliability × phase."""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

from label_resolution import _env_float

FREQUENCY_LOG_SCALE = _env_float("LABEL_FREQUENCY_LOG_SCALE", 0.25)
FREQUENCY_WEIGHT_CAP = _env_float("LABEL_FREQUENCY_WEIGHT_CAP", 3.0)
PHASE_OPENING_WEIGHT = _env_float("LABEL_PHASE_OPENING_WEIGHT", 1.25)
PHASE_MIDGAME_WEIGHT = _env_float("LABEL_PHASE_MIDGAME_WEIGHT", 1.00)
PHASE_ENDGAME_WEIGHT = _env_float("LABEL_PHASE_ENDGAME_WEIGHT", 1.00)

TITANIUM_ANCHORED_LOSS_CAP = _env_float("LABEL_TITANIUM_ANCHORED_LOSS_CAP", 0.60)
TITANIUM_OUTCOME_LOSS_CAP = _env_float("LABEL_TITANIUM_OUTCOME_LOSS_CAP", 0.75)
OPENING_WALLS_PLACED_MAX = 7
ENDGAME_WALLS_REMAINING_MAX = 2


@dataclass(frozen=True)
class ResolvedLabel:
    target: float
    loss_weight: float
    source_tier: str
    game_phase: str
    position_occurrence_count: int
    source_n_samples: int
    source_confidence: float
    frequency_weight: float
    phase_weight: float

    @property
    def occurrence_count(self) -> int:
        """Backward-compatible alias for position-level occurrence."""
        return self.position_occurrence_count


def game_phase_from_record(rec: dict) -> str:
    """Classify opening / midgame / endgame from wall counts in eval JSON."""
    try:
        w0 = float(rec.get("wl0", 10))
        w1 = float(rec.get("wl1", 10))
    except (TypeError, ValueError):
        return "midgame"
    placed = (10.0 - w0) + (10.0 - w1)
    if placed <= OPENING_WALLS_PLACED_MAX:
        return "opening"
    if min(w0, w1) <= ENDGAME_WALLS_REMAINING_MAX:
        return "endgame"
    return "midgame"


def phase_weight(phase: str) -> float:
    if phase == "opening":
        return PHASE_OPENING_WEIGHT
    if phase == "endgame":
        return PHASE_ENDGAME_WEIGHT
    return PHASE_MIDGAME_WEIGHT


def position_frequency_weight(position_occurrence_count: int) -> float:
    """How common this canonical position is (capped)."""
    n = max(1, int(position_occurrence_count))
    return min(FREQUENCY_WEIGHT_CAP, 1.0 + FREQUENCY_LOG_SCALE * math.log2(n))


def frequency_weight(occurrence_count: int) -> float:
    """Backward-compatible alias."""
    return position_frequency_weight(occurrence_count)


def sample_support_confidence(source_n_samples: int) -> float:
    """Rises with independent observations supporting the resolved source."""
    n = max(1, int(source_n_samples))
    return min(1.0, math.log2(n + 1) / 8.0)


def anchor_confidence(anchor_abs: float) -> float:
    a = abs(float(anchor_abs))
    if a < 0.05:
        return 0.0
    if a < 0.20:
        return 0.35
    if a < 0.60:
        return 0.45
    return 0.50


def trusted_tier_confidence(tier: str) -> float:
    tier = tier.lower()
    if tier == "ishtar":
        return 1.00
    if tier in ("zeroink_soft", "zeroink_nn", "zeroink_engine"):
        return 0.90
    if tier in ("zeroink_training", "friend", "friend_outcome", "zeroink_outcome"):
        return 0.80
    if tier in ("ka", "pool_vs_ka"):
        return 0.70
    return 0.25


def pool_mixed_source_confidence(*, anchor_abs: float, outcome_mean: float) -> float:
    agreement = abs(float(outcome_mean))
    anchor_conf = anchor_confidence(anchor_abs)
    if anchor_conf <= 0.0:
        return 0.0
    return anchor_conf * (0.75 + 0.25 * agreement)


def pool_unanimous_source_confidence(source_n_samples: int) -> float:
    sample_conf = sample_support_confidence(source_n_samples)
    return 0.15 + 0.25 * sample_conf


def effective_frequency_weight(frequency_weight: float, source_confidence: float) -> float:
    """Scale the frequency boost by label confidence — common ≠ trustworthy."""
    boost = max(0.0, float(frequency_weight) - 1.0)
    return 1.0 + boost * max(0.0, float(source_confidence))


def cap_tier_loss_weight(raw_weight: float, source_tier: str) -> float:
    tier = source_tier.lower()
    if tier == "titanium_anchored":
        return min(float(raw_weight), TITANIUM_ANCHORED_LOSS_CAP)
    if tier == "titanium_outcome":
        return min(float(raw_weight), TITANIUM_OUTCOME_LOSS_CAP)
    return float(raw_weight)


def combine_loss_weight(
    *,
    source_tier: str,
    position_occurrence_count: int,
    source_n_samples: int,
    game_phase: str,
    mixed_pool: bool = False,
    outcome_mean: float | None = None,
    anchor_abs: float | None = None,
) -> tuple[float, float, float, float]:
    freq = position_frequency_weight(position_occurrence_count)
    phase = phase_weight(game_phase)

    if mixed_pool:
        if outcome_mean is None or anchor_abs is None:
            return 0.0, 0.0, freq, phase
        conf = pool_mixed_source_confidence(
            anchor_abs=float(anchor_abs),
            outcome_mean=float(outcome_mean),
        )
    elif source_tier in ("titanium_outcome", "titanium_anchored"):
        if source_tier == "titanium_anchored":
            if anchor_abs is None:
                return 0.0, 0.0, freq, phase
            conf = pool_mixed_source_confidence(
                anchor_abs=float(anchor_abs),
                outcome_mean=float(outcome_mean or 0.0),
            )
        else:
            conf = pool_unanimous_source_confidence(source_n_samples)
    else:
        conf = trusted_tier_confidence(source_tier)
        # Softly reward repeated trusted observations without conflating with position frequency.
        conf *= 0.85 + 0.15 * sample_support_confidence(source_n_samples)

    if conf <= 0.0:
        return 0.0, conf, freq, phase

    eff_freq = effective_frequency_weight(freq, conf)
    raw = eff_freq * conf * phase
    loss = cap_tier_loss_weight(raw, source_tier)
    return loss, conf, freq, phase
