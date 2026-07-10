"""Resolve one training target per position from multi-source labels.db rows.

Priority (highest first):
  1. Ishtar eval / labels
  2. Zero.ink soft NN eval (``*_nn`` from zero.ink games)
  3. Zero.ink / friend training labels (outcomes from imported AZ corpus)
  4. Ka (pool vs Ka) game outcomes
  5. Pool self-play — per-source running mean on ``*_outcome`` rows only:
     - ``|mean| ≈ 1`` → unanimous game results, train toward ±1
     - ``|mean| < 1`` → mixed winners (never a Quoridor draw); use engine
       anchor if ``|anchor| >= 0.05``, else skip

Never cross-source ``AVG(value_stm)`` — that collapses contradictory outcomes to ~0.
"""
from __future__ import annotations

import os

_POOL_SELFPLAY_OUTCOME = "pool_selfplay_outcome"
_UNANIMOUS_OUTCOME_EPS = 0.999

# Lower rank = higher training priority.
_SOURCE_RANK: tuple[tuple[str, int], ...] = (
    ("ishtar_nn", 5),
    ("ishtar_engine", 8),
    ("ishtar_outcome", 10),
    ("zeroink_nn", 20),
    ("zeroink_engine", 22),
    ("friend_nn", 30),
    ("friend_outcome", 32),
    ("zeroink_outcome", 35),
    ("pool_vs_ka_outcome", 50),
    ("pool_vs_ka_engine", 52),
    ("wallz_outcome", 55),
    ("overnight_selfplay_outcome", 70),
    ("pool_selfplay_engine", 85),
    ("pool_selfplay_outcome", 90),
    ("pool_generation_selfplay_outcome", 92),
    ("oracle_selfplay_outcome", 95),
    ("selfplay_train_outcome", 100),
    ("selfplay_verify_outcome", 105),
    ("pool_generation_outcome", 110),
)

_POOL_SELFPLAY_SOURCES: frozenset[str] = frozenset(
    {
        "pool_selfplay_outcome",
        "pool_generation_selfplay_outcome",
        "overnight_selfplay_outcome",
        "oracle_selfplay_outcome",
        "selfplay_train_outcome",
        "selfplay_verify_outcome",
        "pool_generation_outcome",
    }
)

_EXCLUDED_OUTCOME_SOURCES: frozenset[str] = frozenset(
    {
        "oracle_mixed_outcome",
        "pool_prior_epoch_outcome",
        "pool_mixed_opponent_outcome",
        "overnight_mixed_outcome",
        "pool_generation_mixed_outcome",
    }
)

_EVAL_SCALE_CP = 400.0
LabelRow = tuple[str, float, int]


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


POOL_LOW_CONFIDENCE = _env_float("LABEL_POOL_LOW_CONFIDENCE", 0.05)

# Stockfish-style score/outcome blend (nnue-pytorch train.py's --lambda:
# "1.0 means use purely the score, 0.0 means use purely the outcome"). Applies
# ONLY when a genuine EXTERNAL score-derived signal (ishtar_nn/zeroink_nn/
# friend_nn -- an independent evaluation, not our own self-play engine's
# anchor) AND a genuine resolved outcome both exist for the same position;
# previously this system picked exactly one via the tier list above and threw
# the other away. Deliberately does NOT touch the pool's own engine-anchor
# logic (_pool_engine_anchor / resolve_pool_selfplay_target) -- that anchor is
# a cheap disambiguation heuristic for mixed-winner rows, not an independent
# score signal, so blending it the same way would add noise, not information,
# unlike Stockfish's own search score which is a real deep search.
LABEL_LAMBDA_BLEND = _env_float("LABEL_LAMBDA_BLEND", 0.5)
# Ishtar is measurably stronger than our own engine (see this session's
# match results), so its score is trusted well above a neutral blend with
# our own game outcome -- a higher per-tier lambda than the flat default.
# zeroink_nn/friend_nn aren't as proven, so they stay at the neutral default
# unless overridden. Keyed by _tier_for_source's output, not raw source name.
_TIER_LAMBDA: dict[str, float] = {
    "ishtar": _env_float("LABEL_LAMBDA_BLEND_ISHTAR", 0.8),
}
# Positions from the opening are heavily repeated across many games (same
# book lines recur constantly), so the outcome side of the blend there is
# built from a much larger, more stable sample than a rarer mid/endgame
# position gets -- not comparable enough to the external score signal to mix
# safely. Blend only outside the opening; opening positions keep the
# existing pure tiered-priority resolution untouched.
_BLEND_EXCLUDED_PHASES: frozenset[str] = frozenset({"opening"})
_EXTERNAL_SOFT_SOURCES: frozenset[str] = frozenset({"ishtar_nn", "zeroink_nn", "friend_nn"})


def _lambda_for_tier(tier: str) -> float:
    return _TIER_LAMBDA.get(tier, LABEL_LAMBDA_BLEND)


def stm_from_eval_cp(eval_cp: float, *, scale: float = _EVAL_SCALE_CP) -> float:
    """Map engine centipawn eval to [-1, +1] (matches WDL training scale)."""
    import math

    x = float(eval_cp) / float(scale)
    return max(-1.0, min(1.0, math.tanh(x)))


def source_rank(source: str) -> int:
    if "ishtar" in source:
        if source.endswith("_nn"):
            return 5
        if source.endswith("_engine"):
            return 8
        return 10
    if source.endswith("_nn"):
        return 20
    if source.endswith("_engine"):
        return 22
    if source.startswith("friend"):
        return 32 if source.endswith("_outcome") else 30
    for prefix, rank in _SOURCE_RANK:
        if source == prefix:
            return rank
    return 999


def is_excluded_outcome_source(source: str) -> bool:
    return source in _EXCLUDED_OUTCOME_SOURCES


def is_pool_selfplay_source(source: str) -> bool:
    return source in _POOL_SELFPLAY_SOURCES


def is_pool_engine_source(source: str) -> bool:
    return source.endswith("_engine") and is_pool_selfplay_source(source[: -len("_engine")] + "_outcome")


def _clamp_stm(value: float) -> float:
    return max(-1.0, min(1.0, float(value)))


def _normalize_labels(
    labels: list[tuple[str, float] | tuple[str, float, int]],
) -> list[LabelRow]:
    out: list[LabelRow] = []
    for row in labels:
        if len(row) == 2:
            out.append((str(row[0]), float(row[1]), 1))
        else:
            out.append((str(row[0]), float(row[1]), max(1, int(row[2]))))
    return out


def _split_labels(
    labels: list[LabelRow],
) -> tuple[list[LabelRow], list[LabelRow]]:
    soft: list[LabelRow] = []
    outcomes: list[LabelRow] = []
    for source, value, n_samples in labels:
        if source.endswith("_outcome"):
            if is_excluded_outcome_source(source):
                continue
            outcomes.append((source, float(value), n_samples))
        elif source.endswith("_nn") or source.endswith("_engine"):
            soft.append((source, float(value), n_samples))
    soft.sort(key=lambda row: (source_rank(row[0]), row[0]))
    outcomes.sort(key=lambda row: (source_rank(row[0]), row[0]))
    return soft, outcomes


def _best_soft(soft: list[LabelRow]) -> float | None:
    if not soft:
        return None
    return _clamp_stm(soft[0][1])


def _best_external_soft(soft: list[LabelRow]) -> float | None:
    """Best genuinely-external score-derived value (ishtar_nn/zeroink_nn/
    friend_nn only) -- excludes the pool's own engine-anchor, which is a
    disambiguation heuristic, not an independent score. `soft` is already
    sorted by rank; the external sources always rank above every outcome
    source (5/20/30 vs 50-110), so the first external match is the best one."""
    for source, value, _n in soft:
        if source in _EXTERNAL_SOFT_SOURCES:
            return _clamp_stm(value)
    return None


def _pool_engine_anchor(soft: list[LabelRow], *, explicit: float | None) -> float | None:
    if explicit is not None:
        return _clamp_stm(explicit)
    for source, value, _n in soft:
        if is_pool_engine_source(source):
            return _clamp_stm(value)
    return _best_soft(soft)


def _best_pool_outcome_row(outcomes: list[LabelRow]) -> LabelRow | None:
    pool = [row for row in outcomes if is_pool_selfplay_source(row[0])]
    if not pool:
        return None
    pool.sort(key=lambda row: (source_rank(row[0]), row[0]))
    return pool[0]


def is_unanimous_pool_outcome(outcome_mean: float) -> bool:
    """True when every stored observation for this source agrees on the winner."""
    return abs(_clamp_stm(outcome_mean)) >= _UNANIMOUS_OUTCOME_EPS


def resolve_pool_selfplay_target(
    outcome_mean: float,
    *,
    anchor: float | None,
) -> float | None:
    """Resolve one pool outcome row (source-local running mean, not cross-source)."""
    mean = _clamp_stm(outcome_mean)
    if is_unanimous_pool_outcome(mean):
        return 1.0 if mean > 0 else -1.0

    # Mixed winners — 0.0 is a 50/50 split, not a Quoridor draw.
    if anchor is not None and abs(_clamp_stm(anchor)) >= POOL_LOW_CONFIDENCE:
        return _clamp_stm(anchor)
    return None


def _tier_for_source(source: str) -> str:
    if "ishtar" in source:
        return "ishtar"
    if source.endswith("_nn") or source.endswith("_engine"):
        if "zeroink" in source or source.startswith("friend"):
            return "zeroink_soft"
        return "zeroink_soft"
    if source.startswith("friend") or source in ("zeroink_outcome",):
        return "zeroink_training"
    if source == "pool_vs_ka_outcome":
        return "ka"
    if is_pool_selfplay_source(source):
        return "titanium_outcome"
    return "titanium_outcome"


def _position_occurrence_count(rows: list[LabelRow]) -> int:
    """Best estimate of how often this position was seen (avoid summing correlated sources)."""
    outcome_ns = [n for source, _value, n in rows if source.endswith("_outcome")]
    if not outcome_ns:
        return 1
    # Multiple pool outcome source tags often reference the same underlying games.
    return max(outcome_ns)


def resolve_position_label_bundle(
    labels: list[tuple[str, float] | tuple[str, float, int]],
    *,
    engine_eval_stm: float | None = None,
    position_occurrence_count: int | None = None,
    game_phase: str = "midgame",
) -> "ResolvedLabel | None":
    """Resolve target and per-position loss weight for training."""
    from label_weights import ResolvedLabel, combine_loss_weight

    rows = _normalize_labels(labels)
    if not rows:
        return None

    soft, outcomes = _split_labels(rows)
    anchor = _pool_engine_anchor(soft, explicit=engine_eval_stm)
    pool_rank = source_rank(_POOL_SELFPLAY_OUTCOME)

    tier = "titanium_outcome"
    source_n_samples = 1
    position_count = max(1, int(position_occurrence_count or _position_occurrence_count(rows)))
    target: float | None = None
    pool_outcome_mean: float | None = None
    mixed_pool = False

    def _resolve_outcome_target() -> float | None:
        """Trusted (non-pool) outcome only -- deliberately does NOT fall
        through to pool self-play outcomes the way the tiered fallback below
        does. Pool self-play is the LEAST trusted source in this whole system
        (confidence 0.25, last resort by design) -- diluting an external
        trusted score signal (ishtar_nn/zeroink_nn/friend_nn) with a noisy
        pool outcome would hurt the target, not improve it, unlike Stockfish
        where both blend sides are comparably trustworthy (its own deep
        search + its own real game result)."""
        for src, val, _n in outcomes:
            if not is_pool_selfplay_source(src):
                return _clamp_stm(val)
        return None

    for source, value, n in soft:
        if is_pool_engine_source(source):
            continue
        if source_rank(source) < pool_rank:
            target = _clamp_stm(value)
            tier = _tier_for_source(source)
            source_n_samples = n
            # Stockfish-style blend: an external score signal just won
            # outright by priority -- if a genuine outcome ALSO exists for
            # this position (and it's not an opening -- see
            # _BLEND_EXCLUDED_PHASES), blend them instead of discarding the
            # outcome, weighted by that source tier's own trust level.
            if game_phase not in _BLEND_EXCLUDED_PHASES:
                outcome_target = _resolve_outcome_target()
                if outcome_target is not None:
                    lam = _lambda_for_tier(tier)
                    target = _clamp_stm(lam * target + (1.0 - lam) * outcome_target)
                    tier = f"{tier}_blend"
            break
    else:
        for source, value, n in outcomes:
            if not is_pool_selfplay_source(source):
                target = _clamp_stm(value)
                tier = _tier_for_source(source)
                source_n_samples = n
                break
        else:
            pool_row = _best_pool_outcome_row(outcomes)
            if pool_row is None:
                if soft:
                    target = _best_soft(soft)
                    tier = _tier_for_source(soft[0][0])
                    source_n_samples = soft[0][2]
                else:
                    return None
            else:
                _source, outcome_mean, n = pool_row
                source_n_samples = n
                pool_outcome_mean = outcome_mean
                target = resolve_pool_selfplay_target(outcome_mean, anchor=anchor)
                if target is None:
                    return None
                unanimous = is_unanimous_pool_outcome(outcome_mean)
                mixed_pool = not unanimous
                tier = "titanium_outcome" if unanimous else "titanium_anchored"

    if target is None:
        return None

    loss_weight, conf, freq, phase_w = combine_loss_weight(
        source_tier=tier,
        position_occurrence_count=position_count,
        source_n_samples=source_n_samples,
        game_phase=game_phase,
        mixed_pool=mixed_pool,
        outcome_mean=pool_outcome_mean,
        anchor_abs=abs(anchor) if anchor is not None else None,
    )
    if loss_weight <= 0.0:
        return None

    return ResolvedLabel(
        target=_clamp_stm(target),
        loss_weight=loss_weight,
        source_tier=tier,
        game_phase=game_phase,
        position_occurrence_count=position_count,
        source_n_samples=source_n_samples,
        source_confidence=conf,
        frequency_weight=freq,
        phase_weight=phase_w,
    )


def merge_outcome_sample(prior_value: float, prior_n: int, new_value: float) -> float:
    """Running mean for duplicate import rows (``value_stm`` may be fractional)."""
    n = int(prior_n)
    if n <= 0:
        return float(new_value)
    return (float(prior_value) * n + float(new_value)) / (n + 1)


def resolve_position_labels(
    labels: list[tuple[str, float] | tuple[str, float, int]],
    *,
    engine_eval_stm: float | None = None,
    position_occurrence_count: int | None = None,
    game_phase: str = "midgame",
) -> float | None:
    """Return one STM value in [-1, +1] for training, or None if unusable."""
    bundle = resolve_position_label_bundle(
        labels,
        engine_eval_stm=engine_eval_stm,
        position_occurrence_count=position_occurrence_count,
        game_phase=game_phase,
    )
    return None if bundle is None else bundle.target


def pick_outcome_near_engine(
    prior_value: float,
    new_value: float,
    engine_eval_stm: float,
) -> float:
    """Deprecated import helper — kept for callers; prefer ``merge_outcome_sample``."""
    del engine_eval_stm
    return merge_outcome_sample(float(prior_value), 1, float(new_value))
