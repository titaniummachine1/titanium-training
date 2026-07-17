#!/usr/bin/env python3
"""Finalize the non-promotional Claustrophobia label-quality audit report."""
from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from label_quality_common import load_json, rows_from, sha256, write_json


def rates(rows: list[dict]) -> dict:
    counts = Counter(row.get("classification", "UNKNOWN") for row in rows)
    n = len(rows)
    return {"n": n, "counts": dict(counts),
            "rates": {key: (value / n if n else None) for key, value in counts.items()}}


def breakdown(rows: list[dict], field: str) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[str(row.get(field, "unknown"))].append(row)
    return {key: rates(value) for key, value in sorted(groups.items())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--discovery-sha256", type=Path, required=True)
    ap.add_argument("--discovery-dir", type=Path, required=True)
    args = ap.parse_args()

    deep = rows_from(load_json(args.audit_dir / "deep_check_results.json", []))
    random_rows = rows_from(load_json(args.audit_dir / "random_bias_results.json", []))
    heldout = load_json(args.audit_dir / "heldout_games.json", {}) or {}
    frozen = load_json(args.discovery_sha256, {}) or {}
    actual = {}
    mismatches = {}
    for name, expected in (frozen.get("artifact_sha256") or {}).items():
        path = args.discovery_dir / name
        value = sha256(path) if path.exists() else None
        actual[name] = value
        if value != expected:
            mismatches[name] = {"expected": expected, "actual": value}

    def implied_14(rows: list[dict]) -> dict:
        usable = [r for r in rows if len(r.get("searches") or []) >= 2]
        stable = [r for r in usable if r["searches"][0].get("bestmove") and
                  r["searches"][0].get("bestmove") == r["searches"][1].get("bestmove")]
        return {"n": len(usable), "stable_1s_4s": len(stable),
                "rate": len(stable) / len(usable) if usable else None}

    combined = deep + random_rows
    report = {
        "report_kind": "label_quality_audit",
        "pilot": "mining_pilot_v1",
        "training_eligible": False,
        "training_started": False,
        "proposed_eligible_row_count": 0,
        "heldout": {
            "source_game_ids": heldout.get("source_game_ids", []),
            "lineage_ids": heldout.get("lineage_ids", []),
            "source_game_count": heldout.get("n_source_games", 0),
            "lineage_count": heldout.get("n_lineages", 0),
        },
        "deep": rates(deep),
        "random": rates(random_rows),
        "combined": rates(combined),
        "breakdowns": {
            "deep_by_phase": breakdown(deep, "phase"),
            "deep_by_style": breakdown(deep, "style"),
            "deep_by_move_type": breakdown(deep, "move_type"),
            "random_by_phase": breakdown(random_rows, "phase"),
            "random_by_style": breakdown(random_rows, "style"),
            "random_by_move_type": breakdown(random_rows, "move_type"),
        },
        "priority_head_comparison": {
            "priority_head_implied_1s_4s": implied_14(deep),
            "random_sample_1s_4s": implied_14(random_rows),
            "priority_head_triple": rates(deep),
            "random_sample_triple": rates(random_rows),
        },
        "decision_gate": {
            "thresholds": {
                "false_stable_lt": 0.10,
                "triple_among_proposed_gte": 0.90,
                "style_max_lte": 0.60,
                "phase_max_lte": 0.70,
            },
            "proposed_count": 0,
            "false_stable_pass": (rates(combined)["rates"].get("FALSE_STABLE", 0) < 0.10),
            "triple_among_proposed_pass": True,
            "style_pass": max((v["n"] / len(combined) for v in breakdown(combined, "style").values()), default=0) <= 0.60,
            "phase_pass": max((v["n"] / len(combined) for v in breakdown(combined, "phase").values()), default=0) <= 0.70,
            "promotion_decision": "DEFERRED_PROPOSED_COUNT_ZERO",
        },
        "frozen_discovery_verification": {
            "expected_sha256": frozen.get("artifact_sha256", {}),
            "actual_sha256": actual, "mismatches": mismatches,
            "matches": not mismatches,
        },
    }
    write_json(args.out, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
