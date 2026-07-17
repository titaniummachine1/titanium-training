#!/usr/bin/env python3
"""Create the non-promotional Claustrophobia confidence policy report."""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from label_quality_common import load_json, phase_bucket, rows_from, sha256, write_json


def rates(rows: list[dict], survival: set[str] | None = None) -> dict:
    counts = Counter(row.get("classification", "UNKNOWN") for row in rows)
    n = len(rows)
    result = {"n": n, "counts": dict(counts)}
    if survival is not None:
        good = sum(row.get("classification") in survival for row in rows)
        result["survivors"] = good
        result["survival_rate"] = good / n if n else None
    return result


def breakdown(rows: list[dict], field: str, survival: set[str]) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        value = phase_bucket(row) if field == "phase" else str(row.get(field, "unknown"))
        groups[value].append(row)
    return {key: rates(group, survival) for key, group in sorted(groups.items())}


def verify_hashes(root: Path, frozen: dict) -> dict:
    expected = {}
    expected.update(frozen.get("discovery_artifact_sha256") or {})
    expected.update(frozen.get("audit_artifact_sha256") or {})
    actual, mismatches = {}, {}
    for name, value in expected.items():
        candidates = [root / name, root / "label_quality_audit" / name]
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        actual[name] = sha256(path) if path else None
        if actual[name] != value:
            mismatches[name] = {"expected": value, "actual": actual[name]}
    return {"expected_sha256": expected, "actual_sha256": actual,
            "mismatches": mismatches, "matches": not mismatches}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-dir", type=Path, required=True)
    ap.add_argument("--deep-check", type=Path, required=True)
    ap.add_argument("--completion", type=Path, required=True)
    ap.add_argument("--survives-64-results", type=Path, required=True)
    ap.add_argument("--frozen-hashes", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--titanium-bin", type=Path, default=None)
    ap.add_argument("--titanium-weights", type=Path, default=None)
    args = ap.parse_args()
    deep = rows_from(load_json(args.deep_check, {}))
    completion = rows_from(load_json(args.completion, {}))
    merged = {str(row.get("root_id")): row for row in deep}
    merged.update({str(row.get("root_id")): row for row in completion})
    priority = list(merged.values())
    sixteen_survival = {"SURVIVES_16S", "LOW_MARGIN_16S", "TRIPLE_STABLE", "LOW_CONFIDENCE"}
    sixtyfour_survival = {"SURVIVES_64S", "LOW_MARGIN_64S"}
    sixteens = [row for row in priority if row.get("classification") in sixteen_survival or
                row.get("classification") == "FLIPS_AT_16S"]
    sixtyfours = rows_from(load_json(args.survives_64_results, {}))
    frozen = load_json(args.frozen_hashes, {}) or {}
    node_values = []
    for row in sixteens:
        for search in (row.get("searches") or [])[2:3]:
            info = search.get("info") or {}
            value = info.get("nodes", info.get("totalNodes"))
            if isinstance(value, (int, float)):
                node_values.append(value)
    for row in sixtyfours:
        for search in (row.get("searches") or [])[3:4]:
            info = search.get("info") or {}
            value = info.get("nodes", info.get("totalNodes"))
            if isinstance(value, (int, float)):
                node_values.append(value)
    report = {
        "report_kind": "claustrophobia_label_confidence_policy",
        "pilot": "mining_pilot_v1",
        "training_eligible": False,
        "training_started": False,
        "proposed_eligible_row_count": 0,
        "priority_set_16s": rates(sixteens, sixteen_survival),
        "survives_16s_to_64s_sample": rates(sixtyfours, sixtyfour_survival),
        "breakdowns": {
            "priority_16s_by_phase": breakdown(sixteens, "phase", sixteen_survival),
            "priority_16s_by_style": breakdown(sixteens, "style", sixteen_survival),
            "priority_16s_by_move_type": breakdown(sixteens, "move_type", sixteen_survival),
            "sample_64s_by_phase": breakdown(sixtyfours, "phase", sixtyfour_survival),
            "sample_64s_by_style": breakdown(sixtyfours, "style", sixtyfour_survival),
            "sample_64s_by_move_type": breakdown(sixtyfours, "move_type", sixtyfour_survival),
        },
        "confidence_ladder": {
            "PROVISIONAL_STABLE": "1s bestmove equals 4s bestmove; mining signal only.",
            "DEEP_STABLE": "PROVISIONAL_STABLE and 16s bestmove agrees; not eligible by itself.",
            "VERIFIED_DEEP_STABLE": "DEEP_STABLE and 64s calibration agrees; requires the defined calibration evidence.",
            "EXACT": "Verified label independently confirmed by an exact or exhaustive reference; not established by this pilot.",
            "UNSTABLE": "Any missing/disagreeing search result, or an unresolved low-margin result.",
        },
        "policy_decision": {
            "authorize_training_lane": False,
            "proposed_eligible_row_count": 0,
            "confirmation_eligibility": False,
            "training": "none",
            "stable_search_status": "retired_for_eligibility",
            "note": "STABLE_SEARCH is retired for eligibility; 1s+4s is PROVISIONAL_STABLE only.",
        },
        "cpu_cost_estimates": {
            "verified_deep_stable_search_seconds": 16.0 + 64.0,
            "including_provisional_1s_4s_seconds": 1.0 + 4.0 + 16.0 + 64.0,
            "observed_nodes_across_available_16s_64s_searches": sum(node_values) if node_values else None,
            "node_observation_count": len(node_values),
            "note": "Wall-clock estimates exclude process startup and are descriptive, not a training authorization.",
        },
        "heldout_games": {
            "path": str(args.pilot_dir / "label_quality_audit" / "heldout_games.json"),
            "excluded_from_calibration": True,
        },
        "frozen_audit_verification": verify_hashes(args.pilot_dir, frozen),
    }
    write_json(args.out, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
