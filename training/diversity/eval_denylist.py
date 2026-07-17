"""Evaluation-only asset denylist / leakage prevention."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable

EVAL_DENYLIST_VERSION = "eval-denylist-v1"

from diversity.canonical import reflection_canonical_position_key


@dataclass(frozen=True)
class EvaluationAsset:
    asset_id: str
    kind: str
    canonical_keys: frozenset[str]
    lineage_ids: frozenset[str]


def evaluation_registry_content_hash(
    registry: Iterable[EvaluationAsset] | None = None,
) -> str:
    """Stable hash of the evaluation denylist registry for launch-gate approval."""
    reg = registry or default_evaluation_registry()
    payload = [
        {
            "asset_id": asset.asset_id,
            "kind": asset.kind,
            "canonical_keys": sorted(asset.canonical_keys),
            "lineage_ids": sorted(asset.lineage_ids),
        }
        for asset in reg
    ]
    blob = json.dumps(
        {"eval_denylist_version": EVAL_DENYLIST_VERSION, "assets": payload},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode()).hexdigest()


def default_evaluation_registry() -> tuple[EvaluationAsset, ...]:
    theory24_key = reflection_canonical_position_key({"asset": "theory-24", "version": "eval-only"})
    gate_battery_key = reflection_canonical_position_key(
        {"asset": "frozen_gate_battery", "version": "eval-only"}
    )
    composite_key = reflection_canonical_position_key(
        {"asset": "anchor_alainrinder_pavlosdais_composite", "version": "eval-only"}
    )
    style_panel_key = reflection_canonical_position_key(
        {"asset": "frozen_10k_style_eligibility_panel", "version": "eval-only"}
    )
    promotion_panel_key = reflection_canonical_position_key(
        {"asset": "frozen_promotion_panel", "version": "eval-only"}
    )
    assets = [
        EvaluationAsset("theory-24", "opening_battery", frozenset({theory24_key}), frozenset()),
        EvaluationAsset(
            "frozen_gate_battery", "opening_battery", frozenset({gate_battery_key}), frozenset()
        ),
        EvaluationAsset(
            "anchor_alainrinder_pavlosdais",
            "eval_composite",
            frozenset({composite_key}),
            frozenset(),
        ),
        EvaluationAsset(
            "frozen_10k_style_panel",
            "style_eligibility",
            frozenset({style_panel_key}),
            frozenset({"style_panel_v1"}),
        ),
        EvaluationAsset(
            "frozen_promotion_panel",
            "promotion_panel",
            frozenset({promotion_panel_key}),
            frozenset({"promotion_panel_v1"}),
        ),
    ]
    assets.append(load_claustrophobia_clean_v1_asset())
    return tuple(assets)


def load_claustrophobia_clean_v1_asset() -> EvaluationAsset:
    """Load the immutable clean_v1-derived denylist, failing closed."""
    root = Path(__file__).resolve().parents[1] / "external_sources" / "claustrophobia" / "eval_games" / "clean_v1"
    path = root / "EVAL_DENYLIST_KEYS.json"
    if not path.is_file():
        return EvaluationAsset("claustrophobia_clean_v1", "frozen_evaluation_games",
                               frozenset({"clean_v1:missing"}), frozenset({"clean_v1"}))
    data = json.loads(path.read_text(encoding="utf-8"))
    opening_hashes = list(data.get("opening_hashes", []))
    opening_hashes += [x.rsplit(":", 1)[-1] for x in opening_hashes if ":" in x]
    return EvaluationAsset(
        data.get("asset_id", "claustrophobia_clean_v1"),
        "frozen_evaluation_games",
        frozenset(data.get("canonical_keys", []) + opening_hashes +
                  data.get("opening_ids", [])),
        frozenset(data.get("lineage_ids", []) + data.get("opening_ids", []) + ["clean_v1"]),
    )


def is_evaluation_leakage(
    *,
    canonical_key: str | None = None,
    lineage_id: str | None = None,
    registry: Iterable[EvaluationAsset] | None = None,
) -> tuple[bool, str | None]:
    reg = registry or default_evaluation_registry()
    for asset in reg:
        if canonical_key and canonical_key in asset.canonical_keys:
            return True, asset.asset_id
        if lineage_id and lineage_id in asset.lineage_ids:
            return True, asset.asset_id
    return False, None


def reject_if_evaluation_leakage(
    *,
    canonical_key: str | None = None,
    lineage_id: str | None = None,
    registry: Iterable[EvaluationAsset] | None = None,
) -> None:
    leaked, asset_id = is_evaluation_leakage(
        canonical_key=canonical_key, lineage_id=lineage_id, registry=registry
    )
    if leaked:
        raise ValueError(f"evaluation leakage: matches denylisted asset {asset_id}")
