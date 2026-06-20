"""Schema and conservative labels for Titanium +1 LMR counterfactuals."""

from __future__ import annotations

import hashlib
import math

SCHEMA = "titanium-reduction-counterfactual-v1"
FEATURE_SCHEMA = "halfpw-hidden32-search-context5-v1"
# v2 adds total_legal_moves, history_score, and rank_percentile to the probe event and context
FEATURE_SCHEMA_V2 = "halfpw-hidden32-search-context7-v2"
SIDECAR_SCHEMA = "titanium-reduction-sidecar-v1"
STATUSES = {"SAFE", "UNSAFE", "UNKNOWN"}


def bound_class(score: int, alpha: int, beta: int) -> str:
    if score <= alpha:
        return "FAIL_LOW"
    if score >= beta:
        return "FAIL_HIGH"
    return "EXACT"


def pipeline_decision(event: dict) -> dict:
    score = int(event["score"])
    alpha = int(event["alpha"])
    beta = int(event["beta"])
    return {
        "score": score,
        "bound": bound_class(score, alpha, beta),
        "alpha_improved": score > alpha,
        "cutoff": score >= beta,
        "verification_triggered": bool(event["verification_triggered"]),
        "nodes": int(event["nodes"]),
    }


def comparable_events(baseline: dict, counterfactual: dict) -> tuple[bool, str]:
    identity = (
        "ordinal", "parent_hash", "child_hash", "move", "depth", "ply",
        "alpha", "beta", "move_index", "base_reduction",
    )
    for key in identity:
        if baseline.get(key) != counterfactual.get(key):
            return False, f"context_mismatch:{key}"
    if baseline.get("extra_reduction") or not counterfactual.get("extra_reduction"):
        return False, "wrong_probe_modes"
    if len(baseline.get("hidden", [])) != 32:
        return False, "missing_hidden32"
    return True, ""


def classify_pair(
    baseline: dict,
    counterfactual: dict,
    *,
    minimum_nodes_saved: int,
    minimum_savings_ratio: float,
) -> dict:
    comparable, reason = comparable_events(baseline, counterfactual)
    if not comparable:
        return {
            "sample_status": "UNKNOWN",
            "status_reason": reason,
            "decision_preserved": False,
            "safe_plus_one_reduction": False,
            "worthwhile_net_savings": False,
            "activate_plus_one": False,
        }

    base = pipeline_decision(baseline)
    cf = pipeline_decision(counterfactual)
    same_control = (
        base["bound"] == cf["bound"]
        and base["alpha_improved"] == cf["alpha_improved"]
        and base["cutoff"] == cf["cutoff"]
    )
    # Exact-window results carry an exact value. Fail-low/high scouts carry only
    # a bound, so differing numeric scores inside the same bound are acceptable.
    same_exact_score = base["bound"] != "EXACT" or base["score"] == cf["score"]
    decision_preserved = same_control and same_exact_score
    net_nodes_saved = base["nodes"] - cf["nodes"]
    net_savings_ratio = net_nodes_saved / max(1, base["nodes"])
    worthwhile = (
        net_nodes_saved >= minimum_nodes_saved
        and net_savings_ratio >= minimum_savings_ratio
    )
    return {
        "sample_status": "SAFE" if decision_preserved else "UNSAFE",
        "status_reason": "decision_preserved" if decision_preserved else "decision_changed",
        "decision_preserved": decision_preserved,
        "safe_plus_one_reduction": decision_preserved,
        "worthwhile_net_savings": worthwhile,
        "activate_plus_one": decision_preserved and worthwhile,
        "baseline_final": base,
        "counterfactual_final": cf,
        "verification_triggered": cf["verification_triggered"],
        "baseline_nodes": base["nodes"],
        "counterfactual_nodes": cf["nodes"],
        "net_nodes_saved": net_nodes_saved,
        "net_savings_ratio": net_savings_ratio,
    }


def rank_percentile(move_index: int, total_legal_moves: int) -> float:
    """Soft ordering-rank feature: position of this move in the ordered list.

    0.0 = first move tried, 1.0 = last move tried.
    Costs nothing extra if total_legal_moves is already available.
    """
    return move_index / max(total_legal_moves - 1, 1)


def context_features_v2(event: dict) -> list[float]:
    """7-element context vector (FEATURE_SCHEMA_V2 / context7).

    Extends context5 with history_score (ordering confidence) and rank_percentile
    (branching context). Both fields come from v2 probe events.
    """
    move = str(event["move"])
    mi = int(event["move_index"])
    n = int(event.get("total_legal_moves", 128))
    raw_history = int(event.get("history_score", 0))
    history_norm = max(0.0, min(1.0, (raw_history + 10000) / 20000.0))
    rp = rank_percentile(mi, n)
    return [
        min(max((int(event["depth"]) - 1) / 30.0, 0.0), 1.0),
        min(mi / 128.0, 1.0),
        min(int(event["base_reduction"]) / 4.0, 1.0),
        1.0 if move.endswith("h") else 0.0,
        1.0 if move.endswith("v") else 0.0,
        history_norm,
        rp,
    ]


def stable_partition(game_key: str, seed: int) -> str:
    """Stable game-disjoint train/calibration/test assignment."""
    value = int.from_bytes(hashlib.sha256(f"{seed}:{game_key}".encode()).digest()[:8], "big")
    fraction = value / float(1 << 64)
    if fraction < 0.15:
        return "final_test"
    if fraction < 0.30:
        return "calibration"
    return "train"


def wilson_lower(successes: int, total: int, z: float = 1.959963984540054) -> float:
    if total <= 0:
        return 0.0
    p = successes / total
    den = 1.0 + z * z / total
    centre = p + z * z / (2.0 * total)
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total)
    return max(0.0, (centre - margin) / den)


def validate_row(row: dict) -> None:
    if row.get("schema") != SCHEMA:
        raise ValueError(f"unsupported schema {row.get('schema')!r}")
    if row.get("sample_status") not in STATUSES:
        raise ValueError("invalid sample_status")
    if row["sample_status"] != "UNKNOWN":
        for key in ("baseline_nodes", "counterfactual_nodes", "net_nodes_saved", "net_savings_ratio"):
            if key not in row:
                raise ValueError(f"missing {key}")
    if row.get("activate_plus_one") and not row.get("safe_plus_one_reduction"):
        raise ValueError("activation cannot be unsafe")

