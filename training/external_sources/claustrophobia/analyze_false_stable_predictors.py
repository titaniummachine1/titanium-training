#!/usr/bin/env python3
"""Measure cheap predictors of 1s/4s false stability in the priority set."""
from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from label_quality_common import load_json, move_type, phase_bucket, rows_from, wall_bucket, write_json


def normalize(row: dict) -> str:
    value = row.get("classification")
    return {"TRIPLE_STABLE": "SURVIVES_16S", "LOW_CONFIDENCE": "LOW_MARGIN_16S",
            "FALSE_STABLE": "FLIPS_AT_16S"}.get(value, value)


def info_at(row: dict, index: int) -> dict:
    searches = row.get("searches") or row.get("titanium_searches") or []
    return (searches[index].get("info") or {}) if len(searches) > index else {}


def score_margin(info: dict) -> float | None:
    moves = info.get("rootMoves", info.get("root_moves", [])) or []
    if len(moves) < 2:
        return None
    values = []
    for item in moves[:2]:
        score = item.get("score") if isinstance(item, dict) else None
        if isinstance(score, dict):
            score = score.get("cp", score.get("centipawns", score.get("value")))
        if not isinstance(score, (int, float)):
            return None
        values.append(float(score))
    return abs(values[0] - values[1])


def nodes(info: dict) -> int | None:
    for key in ("nodes", "totalNodes", "mainThreadNodes"):
        value = info.get(key)
        if isinstance(value, (int, float)):
            return int(value)
    return None


def pv(info: dict) -> list[str]:
    value = info.get("pv", info.get("principalVariation", []))
    return [str(item) for item in value] if isinstance(value, list) else []


def pv_divergence(row: dict) -> str:
    a, b = pv(info_at(row, 0)), pv(info_at(row, 1))
    if not a or not b:
        return "unavailable"
    for index, (left, right) in enumerate(zip(a, b)):
        if left != right:
            return f"index_{index}"
    return "same_prefix" if len(a) == len(b) else f"prefix_{min(len(a), len(b))}"


def tertiles(values: list[float]) -> list[float]:
    if not values:
        return []
    ordered = sorted(values)
    return [ordered[max(0, math.ceil(len(ordered) * fraction) - 1)] for fraction in (1 / 3, 2 / 3)]


def numeric_bucket(value: float | None, cuts: list[float]) -> str:
    if value is None:
        return "unknown"
    if not cuts or value <= cuts[0]:
        return "low"
    if value <= cuts[1]:
        return "middle"
    return "high"


def summarize(rows: list[dict], field: str) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(field, "unknown"))].append(row)
    result = {}
    for key, group in sorted(groups.items()):
        flips = sum(normalize(item) == "FLIPS_AT_16S" for item in group)
        result[key] = {"n": len(group), "flips": flips, "survives_or_low_margin": len(group) - flips,
                       "flip_rate": flips / len(group) if group else None}
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deep-check", type=Path, required=True)
    ap.add_argument("--completion", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    merged = {str(row.get("root_id")): row for row in rows_from(load_json(args.deep_check, {}))}
    merged.update({str(row.get("root_id")): row for row in rows_from(load_json(args.completion, {}))})
    rows = [row for row in merged.values() if normalize(row) in {"SURVIVES_16S", "LOW_MARGIN_16S", "FLIPS_AT_16S"}]
    margin_1 = [value for value in (score_margin(info_at(row, 0)) for row in rows) if value is not None]
    margin_4 = [value for value in (score_margin(info_at(row, 1)) for row in rows) if value is not None]
    node_1 = [value for value in (nodes(info_at(row, 0)) for row in rows) if value is not None]
    node_4 = [value for value in (nodes(info_at(row, 1)) for row in rows) if value is not None]
    enriched = []
    for row in rows:
        item = dict(row)
        item["outcome"] = normalize(row)
        item["move_type"] = row.get("move_type", move_type(row))
        item["phase"] = phase_bucket(row)
        item["wall_count"] = wall_bucket(row)
        item["paired_fork"] = "fork" if row.get("paired_fork", row.get("titanium_best") != row.get("claustrophobia_action")) else "aligned"
        item["score_margin_1s"] = numeric_bucket(score_margin(info_at(row, 0)), tertiles(margin_1))
        item["score_margin_4s"] = numeric_bucket(score_margin(info_at(row, 1)), tertiles(margin_4))
        item["nodes_1s"] = numeric_bucket(nodes(info_at(row, 0)), tertiles(node_1))
        item["nodes_4s"] = numeric_bucket(nodes(info_at(row, 1)), tertiles(node_4))
        item["pv_prefix_divergence"] = pv_divergence(row)
        enriched.append(item)
    fields = ("score_margin_1s", "score_margin_4s", "nodes_1s", "nodes_4s",
              "move_type", "phase", "wall_count", "style", "paired_fork",
              "pv_prefix_divergence")
    signals = []
    for field in fields:
        breakdown = summarize(enriched, field)
        rates = [value["flip_rate"] for value in breakdown.values() if value["n"]]
        signals.append({"feature": field, "rate_range": max(rates) - min(rates) if rates else None,
                        "breakdown": breakdown})
    signals.sort(key=lambda item: item["rate_range"] if item["rate_range"] is not None else -1, reverse=True)
    write_json(args.out, {
        "purpose": "cheap_triage_signal_measurement",
        "n_priority_rows": len(enriched),
        "outcome_counts": summarize([{"outcome": item["outcome"]} for item in enriched], "outcome"),
        "numeric_bucket_method": "sample tertiles; descriptive only, not policy thresholds",
        "signals_ranked_by_rate_range": signals,
        "features": list(fields),
        "training_eligible": False,
        "policy_note": "Rates are measured candidates only; no hardcoded eligibility threshold is authorized.",
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
