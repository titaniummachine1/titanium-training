"""Bounded deployment-budget search ladder and telemetry schema."""
from __future__ import annotations

DEFAULT_LADDER = (1, 4, 16, 64)
DEFAULT_DEPLOYMENT_NODES = 50_000


def ladder_stages() -> list[int]:
    return list(DEFAULT_LADDER)


def next_budget(stage: int, deployment_nodes: int = DEFAULT_DEPLOYMENT_NODES) -> int | None:
    if stage < 0:
        raise ValueError("stage must be non-negative")
    if stage >= len(DEFAULT_LADDER):
        return None
    return int(DEFAULT_LADDER[stage] * deployment_nodes)


def stage_record(**values: object) -> dict[str, object]:
    """Create a stable record, retaining all required stage telemetry fields."""
    fields = {
        "position_id", "stage", "budget_nodes", "deployment_budget_nodes",
        "elapsed_cpu_seconds", "elapsed_wall_seconds", "best_move", "wdl",
        "oracle_wdl", "score", "nodes", "nps", "completed", "protocol_ok",
        "semantic_hash", "feature_hash", "provenance_complete",
        "provenance_hash", "move_match", "proof_horizon",
    }
    record = {field: values.get(field) for field in sorted(fields)}
    record.update({key: value for key, value in values.items() if key not in record})
    return record
