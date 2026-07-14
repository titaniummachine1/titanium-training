"""DIVERSITY_SPEC_V1 — backward-compatible re-exports.

Full certificate: ``diversity.certificate``
Quota floors: ``diversity.quota``
"""
from __future__ import annotations

from diversity.canonical import reflection_canonical_four_ply, reflection_canonical_two_ply
from diversity.certificate import (
    MAX_TWO_PLY_PREFIX_MASS,
    MIN_N_EFF_2,
    MIN_N_EFF_4,
    effective_support,
)
from diversity.quota import (
    QUOTA_ADAPTIVE_RESIDUAL,
    QUOTA_BEHAVIORAL_CROSSPLAY,
    QUOTA_CLOSED_LOOP_POP,
    QUOTA_EXACT_ANCHORS,
    QUOTA_FORKS,
    QUOTA_SOLVER_SEAM,
)
from pathlib import Path

from diversity.certificate import prefix_stats


# Legacy CollapseCertificate shape for coordinator logging.
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CollapseCertificate:
    passed: bool
    game_count: int
    n_eff_2: float
    n_eff_4: float
    max_two_ply_mass: float
    two_ply_counts: dict[str, int]
    block_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "game_count": self.game_count,
            "n_eff_2": round(self.n_eff_2, 4),
            "n_eff_4": round(self.n_eff_4, 4),
            "max_two_ply_mass": round(self.max_two_ply_mass, 4),
            "block_reason": self.block_reason,
        }


def load_game_opening_prefixes(
    games_db: Path,
    *,
    min_plies: int = 2,
    max_games: int | None = None,
) -> list[tuple[str, ...]]:
    import sqlite3

    if not games_db.is_file():
        return []
    con = sqlite3.connect(games_db)
    try:
        has_moves = con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='game_moves'"
        ).fetchone()
        if not has_moves:
            return []
        limit_sql = f" LIMIT {int(max_games)}" if max_games else ""
        game_ids = [
            str(row[0])
            for row in con.execute(
                f"SELECT game_id FROM games ORDER BY imported_at DESC{limit_sql}"
            )
        ]
        prefixes: list[tuple[str, ...]] = []
        for game_id in game_ids:
            rows = con.execute(
                "SELECT move_alg FROM game_moves WHERE game_id = ? ORDER BY move_num",
                (game_id,),
            ).fetchall()
            moves = tuple(str(r[0]) for r in rows)
            if len(moves) >= min_plies:
                prefixes.append(moves)
        return prefixes
    finally:
        con.close()


def collapse_certificate_from_prefixes(prefixes: list[tuple[str, ...]]) -> CollapseCertificate:
    n_eff_2, n_eff_4, max_two = prefix_stats(prefixes)
    reasons: list[str] = []
    if not prefixes:
        reasons.append("no eligible games")
    if max_two > MAX_TWO_PLY_PREFIX_MASS:
        reasons.append(f"two-ply mass {max_two:.3f} > {MAX_TWO_PLY_PREFIX_MASS}")
    if n_eff_2 < MIN_N_EFF_2:
        reasons.append(f"N_eff(2)={n_eff_2:.2f} < {MIN_N_EFF_2}")
    if n_eff_4 < MIN_N_EFF_4:
        reasons.append(f"N_eff(4)={n_eff_4:.2f} < {MIN_N_EFF_4}")
    return CollapseCertificate(
        passed=not reasons,
        game_count=len(prefixes),
        n_eff_2=n_eff_2,
        n_eff_4=n_eff_4,
        max_two_ply_mass=max_two,
        two_ply_counts={},
        block_reason="; ".join(reasons) if reasons else None,
    )


def collapse_certificate(games_db: Path, *, max_games: int | None = None) -> CollapseCertificate:
    return collapse_certificate_from_prefixes(
        load_game_opening_prefixes(games_db, max_games=max_games)
    )
