"""Quota-aware seed selection (fixture / deterministic planner only).

No real generation. Selection is deterministic for a fixed planner seed.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

from diversity.seed_bank_schema import SeedRecord, validate_seed_for_selection

SEED_SELECTION_POLICY_VERSION = "seed-selection-policy-v1"


@dataclass
class SeedSelectionConfig:
    cooldown_batches: int = 1
    max_per_family_per_batch: int = 2
    max_standard_start_share: float = 0.15
    planner_seed: int = 0
    policy_version: str = SEED_SELECTION_POLICY_VERSION


@dataclass
class SelectionDecision:
    seed_id: str
    accepted: bool
    reason: str
    rank_score: float = 0.0


@dataclass
class SeedUsageTracker:
    """Tracks usage / cooldowns across batches (in-memory fixture tracker)."""

    usage_count: Counter[str] = field(default_factory=Counter)
    last_batch_used: dict[str, int] = field(default_factory=dict)
    family_usage: Counter[str] = field(default_factory=Counter)
    batch_index: int = 0

    def note_selected(self, seed: SeedRecord) -> None:
        self.usage_count[seed.seed_id] += 1
        self.family_usage[seed.seed_family_id] += 1
        self.last_batch_used[seed.seed_id] = self.batch_index


def _stable_rank(seed: SeedRecord, cfg: SeedSelectionConfig, tracker: SeedUsageTracker) -> float:
    """Lower is better. Deterministic given planner_seed."""
    blob = f"{cfg.planner_seed}|{seed.seed_id}|{seed.seed_family_id}|{seed.phase}|{seed.tension_class}"
    h = hashlib.sha256(blob.encode()).hexdigest()
    base = int(h[:12], 16) / float(16**12)
    # Prefer underused seeds / families.
    base += 0.01 * tracker.usage_count[seed.seed_id]
    base += 0.005 * tracker.family_usage[seed.seed_family_id]
    return base


def select_seeds_for_batch(
    candidates: Iterable[SeedRecord],
    *,
    batch_size: int,
    cfg: SeedSelectionConfig,
    tracker: SeedUsageTracker,
    phase_tension_deficit: dict[tuple[str, str], float] | None = None,
    stm_deficit: dict[int, float] | None = None,
) -> tuple[list[SeedRecord], list[SelectionDecision]]:
    """Select up to batch_size seeds with safeguards. Deterministic for fixed planner_seed."""
    decisions: list[SelectionDecision] = []
    selected: list[SeedRecord] = []
    seen_canonical: set[str] = set()
    family_in_batch: Counter[str] = Counter()
    standard_start_count = 0
    deficit = phase_tension_deficit or {}
    stm_def = stm_deficit or {}

    ranked: list[tuple[float, SeedRecord]] = []
    for seed in candidates:
        reasons = validate_seed_for_selection(seed)
        if reasons:
            decisions.append(
                SelectionDecision(seed.seed_id, False, "reject:" + ",".join(reasons))
            )
            continue
        last = tracker.last_batch_used.get(seed.seed_id)
        if last is not None and (tracker.batch_index - last) < cfg.cooldown_batches:
            decisions.append(
                SelectionDecision(seed.seed_id, False, "reject:cooldown")
            )
            continue
        score = _stable_rank(seed, cfg, tracker)
        # Priority boost for deficit cells / STM.
        score -= 0.1 * deficit.get((seed.phase, seed.tension_class), 0.0)
        score -= 0.05 * stm_def.get(seed.side_to_move, 0.0)
        ranked.append((score, seed))

    ranked.sort(key=lambda x: (x[0], x[1].seed_id))

    for score, seed in ranked:
        if len(selected) >= batch_size:
            decisions.append(
                SelectionDecision(seed.seed_id, False, "reject:batch_full", score)
            )
            continue
        if seed.reflection_canonical_state_key in seen_canonical:
            decisions.append(
                SelectionDecision(seed.seed_id, False, "reject:duplicate_canonical_in_batch", score)
            )
            continue
        if family_in_batch[seed.seed_family_id] >= cfg.max_per_family_per_batch:
            decisions.append(
                SelectionDecision(seed.seed_id, False, "reject:family_cap", score)
            )
            continue
        if seed.origin_source == "standard_start":
            if (standard_start_count + 1) / max(1, batch_size) > cfg.max_standard_start_share:
                decisions.append(
                    SelectionDecision(seed.seed_id, False, "reject:standard_start_cap", score)
                )
                continue
            standard_start_count += 1

        selected.append(seed)
        seen_canonical.add(seed.reflection_canonical_state_key)
        family_in_batch[seed.seed_family_id] += 1
        tracker.note_selected(seed)
        decisions.append(
            SelectionDecision(seed.seed_id, True, "accept:quota_aware", score)
        )

    tracker.batch_index += 1
    return selected, decisions


def selection_log_payload(
    selected: list[SeedRecord],
    decisions: list[SelectionDecision],
    cfg: SeedSelectionConfig,
) -> dict[str, Any]:
    return {
        "policy_version": cfg.policy_version,
        "planner_seed": cfg.planner_seed,
        "selected_seed_ids": [s.seed_id for s in selected],
        "decisions": [
            {"seed_id": d.seed_id, "accepted": d.accepted, "reason": d.reason, "rank_score": d.rank_score}
            for d in decisions
        ],
    }
