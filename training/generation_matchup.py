"""Generation matchup selection: continuous 30% prior-epoch self-play.

At all times during training, STREAM_PRIOR_EPOCH_FRACTION (~30%) of games
are current weights vs the immediately previous accepted weights, same
engine, same node/time budget -- not a phase, not a one-time check, just an
ongoing fraction of every batch of games for as long as training runs. The
remaining ~70% are current-vs-current self-play. All games count as training
data regardless of matchup kind.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MATCHUP_SELFPLAY = "selfplay"
MATCHUP_PRIOR_EPOCH = "prior_epoch"


def _candidate_engine() -> str:
    return os.environ.get("TITANIUM_GENERATION_ENGINE", "titanium-v17").strip() or "titanium-v17"


def _prior_epoch_fraction() -> float:
    raw = os.environ.get("STREAM_PRIOR_EPOCH_FRACTION", "0.30")
    try:
        return max(0.0, min(1.0, float(raw)))
    except ValueError:
        return 0.30


def uses_weight_override(engine: str) -> bool:
    return engine.startswith("titanium-v")


@dataclass(frozen=True)
class GenerationMatchup:
    kind: str
    engine_p0: str
    engine_p1: str
    weights_p0: Path | None
    weights_p1: Path | None
    current_is_p0: bool
    opponent_engine: str | None
    opening_exploration: bool
    metadata: dict[str, Any]


def choose_generation_matchup(
    rng: random.Random,
    *,
    current_weights: Path,
    previous_weights: Path | None,
) -> GenerationMatchup:
    """~30% of games at all times: current vs immediately previous accepted."""
    cur_eng = _candidate_engine()
    prior_frac = _prior_epoch_fraction()
    has_prior = (
        previous_weights is not None
        and previous_weights.is_file()
        and previous_weights.resolve() != current_weights.resolve()
    )

    if has_prior and rng.random() < prior_frac:
        current_is_p0 = rng.random() < 0.5
        if current_is_p0:
            w_p0, w_p1 = current_weights, previous_weights
        else:
            w_p0, w_p1 = previous_weights, current_weights
        return GenerationMatchup(
            kind=MATCHUP_PRIOR_EPOCH,
            engine_p0=cur_eng,
            engine_p1=cur_eng,
            weights_p0=w_p0,
            weights_p1=w_p1,
            current_is_p0=current_is_p0,
            opponent_engine=None,
            opening_exploration=False,
            metadata={"prior_epoch_fraction": prior_frac},
        )

    return GenerationMatchup(
        kind=MATCHUP_SELFPLAY,
        engine_p0=cur_eng,
        engine_p1=cur_eng,
        weights_p0=current_weights,
        weights_p1=current_weights,
        current_is_p0=True,
        opponent_engine=None,
        opening_exploration=False,
        metadata={"prior_epoch_fraction": prior_frac},
    )
