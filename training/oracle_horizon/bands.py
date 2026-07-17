"""Proof-horizon bands and conservative expansion gate."""
from __future__ import annotations


def assign_band(plies_to_oracle: int) -> int:
    if plies_to_oracle < 0:
        raise ValueError("plies_to_oracle must be non-negative")
    if plies_to_oracle == 0:
        return 0
    if plies_to_oracle <= 2:
        return 1
    if plies_to_oracle <= 4:
        return 2
    if plies_to_oracle <= 8:
        return 3
    if plies_to_oracle <= 16:
        return 4
    return 5


def active_bands_for_pilot() -> set[int]:
    return {0, 1, 2, 3}


def expand_band_allowed(mastery_metrics: dict | None) -> bool:
    """Expand only after explicit frontier mastery; missing metrics fail closed."""
    metrics = mastery_metrics or {}
    if metrics.get("frontier_mastered") is True:
        return True
    required = ("accuracy", "wdl_accuracy", "move_accuracy", "min_samples")
    return (
        all(key in metrics for key in required)
        and float(metrics["accuracy"]) >= 0.95
        and float(metrics["wdl_accuracy"]) >= 0.98
        and float(metrics["move_accuracy"]) >= 0.90
        and int(metrics["min_samples"]) >= 100
    )
