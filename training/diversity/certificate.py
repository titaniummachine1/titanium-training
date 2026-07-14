"""Full DIVERSITY_SPEC_V1 collapse certificate — PASS / BLOCK / INVALID."""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable

from diversity.canonical import (
    CanonicalStateRow,
    deduplicate_finalized_rows,
    reflection_canonical_four_ply,
    reflection_canonical_two_ply,
)
from diversity.lanes import DiversityLane
from diversity.quota import QUOTA_TOLERANCE_PP, validate_quota_shares
from diversity.seam_anchors import ExactAnchorRecord, validate_exact_anchor_balance

MAX_TWO_PLY_PREFIX_MASS = 0.10
MIN_N_EFF_2 = 16
MIN_N_EFF_4 = 64
MAX_ROWS_PER_SOURCE_GAME = 32
MAX_ROWS_PER_FORK_LINEAGE = 128
MIN_NOVELTY_FRACTION = 0.25
MIN_CELL_SHARE = 0.05
STM_MIN = 0.45
STM_MAX = 0.55


class CertificateStatus(str, Enum):
    PASS = "PASS"
    BLOCK = "BLOCK"
    INVALID = "INVALID"


@dataclass(frozen=True)
class CorpusSampleRow:
    metadata_lane: DiversityLane
    source_game_id: str
    fork_lineage_id: str | None
    side_to_move: int
    canonical: CanonicalStateRow
    is_exact: bool = False
    phase: str = "opening"
    tension: str = "default"


@dataclass(frozen=True)
class CertificateMeasurementContext:
    diversity_spec_version: str
    engine_semantics_hash: str
    corpus_generation_id: str
    prefixes_trusted: bool = False
    prior_corpora_trusted: bool = False


@dataclass(frozen=True)
class FullCollapseCertificate:
    status: CertificateStatus
    game_count: int
    n_eff_2: float
    n_eff_4: float
    max_two_ply_mass: float
    duplicate_canonical_states: int
    max_rows_per_source_game: int
    max_rows_per_fork_lineage: int
    effective_source_games: int
    novelty_fraction: float | None
    stm_white_fraction: float
    reasons: tuple[str, ...]
    measurement_errors: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.status == CertificateStatus.PASS

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "game_count": self.game_count,
            "n_eff_2": round(self.n_eff_2, 4),
            "n_eff_4": round(self.n_eff_4, 4),
            "max_two_ply_mass": round(self.max_two_ply_mass, 4),
            "duplicate_canonical_states": self.duplicate_canonical_states,
            "max_rows_per_source_game": self.max_rows_per_source_game,
            "max_rows_per_fork_lineage": self.max_rows_per_fork_lineage,
            "effective_source_games": self.effective_source_games,
            "novelty_fraction": self.novelty_fraction,
            "stm_white_fraction": round(self.stm_white_fraction, 4),
            "reasons": list(self.reasons),
            "measurement_errors": list(self.measurement_errors),
        }


def effective_support(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    probs = [c / total for c in counts.values() if c > 0]
    inv_sum = sum(p * p for p in probs)
    if inv_sum <= 0.0:
        return 0.0
    return 1.0 / inv_sum


def prefix_stats(prefixes: list[tuple[str, ...]]) -> tuple[float, float, float]:
    two_ply: Counter[str] = Counter()
    four_ply: Counter[str] = Counter()
    eligible = 0
    for moves in prefixes:
        if len(moves) < 2:
            continue
        eligible += 1
        two_ply[reflection_canonical_two_ply(moves[0], moves[1])] += 1
        if len(moves) >= 4:
            four_ply[reflection_canonical_four_ply(tuple(moves[:4]))] += 1
    n_eff_2 = effective_support(two_ply)
    n_eff_4 = effective_support(four_ply) if four_ply else 0.0
    max_two = max((c / eligible for c in two_ply.values()), default=0.0) if eligible else 1.0
    return n_eff_2, n_eff_4, max_two


def _validate_metadata(ctx: CertificateMeasurementContext | None) -> list[str]:
    if ctx is None:
        return ["missing certificate measurement context"]
    errors: list[str] = []
    for field in ("diversity_spec_version", "engine_semantics_hash", "corpus_generation_id"):
        if not getattr(ctx, field, "").strip():
            errors.append(f"missing metadata field: {field}")
    if ctx.engine_semantics_hash in ("", "prep", "unknown"):
        errors.append("untrustworthy engine_semantics_hash")
    return errors


def validate_full_certificate(
    rows: Iterable[CorpusSampleRow],
    *,
    prefixes: list[tuple[str, ...]] | None = None,
    prior_canonical_keys: set[str] | None = None,
    prior_canonical_keys_2: set[str] | None = None,
    per_lane_counts: dict[DiversityLane, int] | None = None,
    total_rows: int | None = None,
    measurement_context: CertificateMeasurementContext | None = None,
    exact_anchors: list[ExactAnchorRecord] | None = None,
) -> FullCollapseCertificate:
    measurement_errors = _validate_metadata(measurement_context)
    row_list = list(rows)
    if not row_list and not measurement_errors:
        measurement_errors.append("empty corpus sample")

    if measurement_errors:
        return FullCollapseCertificate(
            status=CertificateStatus.INVALID,
            game_count=0,
            n_eff_2=0.0,
            n_eff_4=0.0,
            max_two_ply_mass=1.0,
            duplicate_canonical_states=0,
            max_rows_per_source_game=0,
            max_rows_per_fork_lineage=0,
            effective_source_games=0,
            novelty_fraction=None,
            stm_white_fraction=0.0,
            reasons=(),
            measurement_errors=tuple(measurement_errors),
        )

    if prefixes is None or not measurement_context or not measurement_context.prefixes_trusted:
        measurement_errors.append("opening prefix panel missing or untrusted")

    if measurement_errors:
        return FullCollapseCertificate(
            status=CertificateStatus.INVALID,
            game_count=len({r.source_game_id for r in row_list}),
            n_eff_2=0.0,
            n_eff_4=0.0,
            max_two_ply_mass=1.0,
            duplicate_canonical_states=0,
            max_rows_per_source_game=0,
            max_rows_per_fork_lineage=0,
            effective_source_games=0,
            novelty_fraction=None,
            stm_white_fraction=0.0,
            reasons=(),
            measurement_errors=tuple(measurement_errors),
        )

    canonical_rows = [r.canonical for r in row_list]
    unique, dupes = deduplicate_finalized_rows(canonical_rows)

    per_game = Counter(r.source_game_id for r in row_list)
    per_fork = Counter(
        r.fork_lineage_id for r in row_list if r.fork_lineage_id is not None
    )
    max_game = max(per_game.values(), default=0)
    max_fork = max(per_fork.values(), default=0)
    n_rows = len(row_list)
    effective_games = len(per_game)

    stm_white = sum(1 for r in row_list if r.side_to_move == 0)
    stm_frac = stm_white / n_rows if n_rows else 0.0

    n_eff_2, n_eff_4, max_two = prefix_stats(prefixes or [])

    novelty_fraction: float | None = None
    if prior_canonical_keys is not None and prior_canonical_keys_2 is not None:
        prior_union = prior_canonical_keys | prior_canonical_keys_2
        novel = sum(1 for r in unique if r.canonical_key() not in prior_union)
        novelty_fraction = novel / len(unique) if unique else 0.0

    block_reasons: list[str] = []
    if dupes > 0:
        block_reasons.append(f"{dupes} duplicate canonical states in finalized sample")
    if max_two > MAX_TWO_PLY_PREFIX_MASS:
        block_reasons.append(f"two-ply mass {max_two:.3f} > {MAX_TWO_PLY_PREFIX_MASS}")
    if n_eff_2 < MIN_N_EFF_2:
        block_reasons.append(f"N_eff(2)={n_eff_2:.2f} < {MIN_N_EFF_2}")
    if n_eff_4 < MIN_N_EFF_4:
        block_reasons.append(f"N_eff(4)={n_eff_4:.2f} < {MIN_N_EFF_4}")
    if max_game > MAX_ROWS_PER_SOURCE_GAME:
        block_reasons.append(f"source game rows {max_game} > {MAX_ROWS_PER_SOURCE_GAME}")
    if max_fork > MAX_ROWS_PER_FORK_LINEAGE:
        block_reasons.append(f"fork lineage rows {max_fork} > {MAX_ROWS_PER_FORK_LINEAGE}")
    min_games = math.ceil(n_rows / MAX_ROWS_PER_SOURCE_GAME) if n_rows else 0
    if effective_games < min_games:
        block_reasons.append(
            f"effective source games {effective_games} < N/32 ({min_games})"
        )
    if novelty_fraction is not None and novelty_fraction < MIN_NOVELTY_FRACTION:
        block_reasons.append(
            f"novelty fraction {novelty_fraction:.3f} < {MIN_NOVELTY_FRACTION}"
        )
    if not (STM_MIN <= stm_frac <= STM_MAX):
        block_reasons.append(f"STM white fraction {stm_frac:.3f} outside [{STM_MIN}, {STM_MAX}]")

    if per_lane_counts and total_rows:
        block_reasons.extend(validate_quota_shares(per_lane_counts, total_rows))
        non_exact = sum(
            c for lane, c in per_lane_counts.items() if lane != DiversityLane.EXACT_ANCHORS
        )
        if non_exact > 0:
            for lane, count in per_lane_counts.items():
                if lane == DiversityLane.EXACT_ANCHORS:
                    continue
                share = count / non_exact
                if share < MIN_CELL_SHARE - 1e-9:
                    block_reasons.append(f"cell {lane.value} share {share:.3f} < {MIN_CELL_SHARE}")

    phase_tension = Counter((r.phase, r.tension) for r in row_list if not r.is_exact)
    if phase_tension and n_rows:
        non_exact_n = sum(1 for r in row_list if not r.is_exact)
        for cell, count in phase_tension.items():
            share = count / non_exact_n
            if share < MIN_CELL_SHARE - 1e-9:
                block_reasons.append(
                    f"phase×tension cell {cell} share {share:.3f} < {MIN_CELL_SHARE}"
                )

    if exact_anchors:
        for err in validate_exact_anchor_balance(exact_anchors):
            block_reasons.append(err)

    status = CertificateStatus.PASS if not block_reasons else CertificateStatus.BLOCK
    return FullCollapseCertificate(
        status=status,
        game_count=len(per_game),
        n_eff_2=n_eff_2,
        n_eff_4=n_eff_4,
        max_two_ply_mass=max_two,
        duplicate_canonical_states=dupes,
        max_rows_per_source_game=max_game,
        max_rows_per_fork_lineage=max_fork,
        effective_source_games=effective_games,
        novelty_fraction=novelty_fraction,
        stm_white_fraction=stm_frac,
        reasons=tuple(block_reasons),
        measurement_errors=(),
    )
