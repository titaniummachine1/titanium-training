#!/usr/bin/env python3
"""Complete the frozen 1s/4s priority audit with a resumable 16s pass."""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
from engine_session import EngineSession
from label_quality_common import load_json, margin_cp, move_type, result_score, rows_from, write_json


def classify(searches: list[dict]) -> tuple[str, str | None]:
    if len(searches) < 3:
        return "UNSTABLE_16S", "missing_search"
    first_search, fourth_search, deep_search = searches[:3]
    first, fourth, deep = (item.get("bestmove") for item in (first_search, fourth_search, deep_search))
    if not first or not fourth or not deep or first != fourth:
        return "UNSTABLE_16S", "not_provisional_stable"
    if deep != fourth:
        return "FLIPS_AT_16S", None
    margin = margin_cp((deep_search or {}).get("info"))
    if margin is not None and margin < 50.0:
        return "LOW_MARGIN_16S", None
    note = "margin_unavailable" if margin is None else None
    return "SURVIVES_16S", note


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--priority", type=Path, required=True)
    ap.add_argument("--deep-check", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--titanium-bin", type=Path, required=True)
    ap.add_argument("--titanium-weights", type=Path, required=True)
    ap.add_argument("--deep-sec", type=float, default=16.0)
    ap.add_argument("--max", type=int, default=0)
    args = ap.parse_args()
    os.environ["TITANIUM_BOOK_MODE"] = "off"
    source_rows = rows_from(load_json(args.priority, []))
    deep_rows = rows_from(load_json(args.deep_check, {})) if args.deep_check else []
    by_id = {str(row.get("root_id")): row for row in deep_rows}
    rows = []
    for row in source_rows:
        merged = dict(row)
        prior = by_id.get(str(row.get("root_id")))
        if prior:
            merged.update(prior)
        rows.append(merged)
    existing = load_json(args.out, {}) or {}
    results = {}
    for row in deep_rows:
        if len(row.get("searches") or []) >= 3:
            old = row.get("classification")
            mapped = {"TRIPLE_STABLE": "SURVIVES_16S",
                      "LOW_CONFIDENCE": "LOW_MARGIN_16S",
                      "FALSE_STABLE": "FLIPS_AT_16S"}.get(old, old)
            copied = dict(row)
            copied["classification"] = mapped
            copied["deep_16s_move"] = (row.get("searches") or [None, None, {}])[2].get("bestmove")
            copied["training_eligible"] = False
            results[str(row.get("root_id"))] = copied
    results.update({str(row.get("root_id")): row for row in rows_from(existing)})
    todo = [row for row in rows if str(row.get("root_id")) not in results]
    if args.max:
        todo = todo[:args.max]
    started = time.perf_counter()
    session = EngineSession("titanium-v17", args.titanium_weights, engine_bin=args.titanium_bin)
    try:
        for row in todo:
            frozen = row.get("searches") or row.get("titanium_searches") or []
            searches = list(frozen[:2])
            status = "ok"
            if len(searches) < 2:
                status = "missing_frozen_1s_4s"
            elif not session.sync(row.get("prefix_moves") or []):
                status = "protocol_error"
            else:
                searches.append(session.go_detailed(args.deep_sec))
            classification, note = classify(searches) if status == "ok" else ("UNSTABLE_16S", status)
            results[str(row.get("root_id"))] = {
                "root_id": row.get("root_id"),
                "source_game_id": row.get("source_game_id"),
                "fork_lineage_id": row.get("fork_lineage_id"),
                "style": row.get("style"),
                "phase": row.get("phase"),
                "wall_count": row.get("wall_count"),
                "side_to_move": row.get("side_to_move"),
                "move_type": row.get("move_type", move_type(row)),
                "paired_fork": row.get("paired_fork", row.get("titanium_best") != row.get("claustrophobia_action")),
                "prefix_moves": row.get("prefix_moves") or [],
                "searches": searches,
                "classification": classification,
                "classification_note": note,
                "status": status,
                "frozen_1s_move": searches[0].get("bestmove") if searches else None,
                "frozen_4s_move": searches[1].get("bestmove") if len(searches) > 1 else None,
                "deep_16s_move": searches[2].get("bestmove") if len(searches) > 2 else None,
                "scores": [result_score(s.get("info")) for s in searches],
                "training_eligible": False,
            }
            write_json(args.out, {
                "budget_sec": args.deep_sec, "engine": "titanium-v17",
                "opening_book": "off", "weights": str(args.titanium_weights),
                "rows": [results[key] for key in sorted(results)],
                "n_rows": len(results), "training_eligible": False,
                "resumable": True,
            })
    finally:
        session.close()
    output_rows = [results[key] for key in sorted(results)]
    counts: dict[str, int] = {}
    for item in output_rows:
        counts[item["classification"]] = counts.get(item["classification"], 0) + 1
    write_json(args.out, {
        "budget_sec": args.deep_sec, "engine": "titanium-v17",
        "opening_book": "off", "weights": str(args.titanium_weights),
        "rows": output_rows, "n_rows": len(output_rows), "summary": counts,
        "elapsed_sec": time.perf_counter() - started, "training_eligible": False,
        "resumable": True,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
