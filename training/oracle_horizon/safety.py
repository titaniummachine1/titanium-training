"""Fail-closed safety gates for the supervised pilot."""
from __future__ import annotations

import math

CRITICAL_FLAGS = {
    "protocol_error", "semantic_hash_changed", "feature_hash_changed",
    "incomplete_provenance", "eval_leakage", "oracle_contradiction",
    "no_accepted_improvement_2_cycles", "frozen_anchor_regression",
    "negative_proof_horizon_gain", "queue_limit", "disk_limit",
    "stale_producer", "match_integrity_fail", "unverifiable_accepted_hash",
}


def should_pause(events: dict) -> tuple[bool, list[str]]:
    events = events or {}
    aliases = {
        "protocol_errors": "protocol_error", "semantic_hash_change": "semantic_hash_changed",
        "feature_hash_change": "feature_hash_changed", "queue_over_limit": "queue_limit",
        "disk_over_limit": "disk_limit", "match_integrity": "match_integrity_fail",
    }
    reasons = [
        aliases.get(name, name) for name in sorted(CRITICAL_FLAGS)
        if events.get(name) is True
    ]
    reasons.extend(reason for key, reason in aliases.items() if events.get(key) is True)
    unknown = events.get("unknown_critical_flags", [])
    if unknown is True:
        reasons.append("unknown_critical_flag")
    elif unknown:
        reasons.extend(f"unknown_critical_flag:{name}" for name in unknown)
    reasons.extend(f"unknown_critical_flag:{key}" for key, value in events.items()
                   if key.startswith("unknown_") and value is True and key != "unknown_critical_flags")
    loss = events.get("loss")
    if loss is not None and (not isinstance(loss, (int, float)) or not math.isfinite(float(loss))):
        reasons.append("non_finite_loss")
    if events.get("non_finite_loss") is True:
        reasons.append("non_finite_loss")
    return bool(reasons), list(dict.fromkeys(reasons))
