#!/usr/bin/env python3
"""Run the 16-second continuation audit for the deep sample."""
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
from label_quality_common import classify_searches, load_json, rows_from, result_score, write_json


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--titanium-bin", type=Path, required=True)
    ap.add_argument("--titanium-weights", type=Path, required=True)
    ap.add_argument("--deep-sec", type=float, default=16.0)
    ap.add_argument("--max", type=int, default=0)
    args = ap.parse_args()
    os.environ["TITANIUM_BOOK_MODE"] = "off"
    rows = rows_from(load_json(args.sample, []))
    existing = load_json(args.out, {}) or {}
    prior = {str(row.get("root_id")): row for row in rows_from(existing)}
    todo = [row for row in rows if str(row.get("root_id")) not in prior]
    if args.max:
        todo = todo[:args.max]
    results = dict(prior)
    started = time.perf_counter()
    session = EngineSession("titanium-v17", args.titanium_weights, engine_bin=args.titanium_bin)
    try:
        for row in todo:
            frozen = row.get("titanium_searches") or []
            searches = list(frozen[:2])
            status = "ok"
            if len(searches) < 2:
                status = "missing_frozen_1s_4s"
            elif not session.sync(row.get("prefix_moves") or []):
                status = "protocol_error"
            else:
                searches.append(session.go_detailed(args.deep_sec))
            classification = classify_searches(searches) if len(searches) == 3 else "UNSTABLE"
            results[str(row.get("root_id"))] = {
                "root_id": row.get("root_id"), "source_game_id": row.get("source_game_id"),
                "fork_lineage_id": row.get("fork_lineage_id"), "style": row.get("style"),
                "phase": row.get("phase"), "side_to_move": row.get("side_to_move"),
                "move_type": "wall" if len(str(row.get("claustrophobia_action") or "")) >= 3 else "pawn",
                "paired_fork": row.get("titanium_best") != row.get("claustrophobia_action"),
                "searches": searches, "classification": classification, "status": status,
                "frozen_1s_move": searches[0].get("bestmove") if searches else None,
                "frozen_4s_move": searches[1].get("bestmove") if len(searches) > 1 else None,
                "deep_move": searches[2].get("bestmove") if len(searches) > 2 else None,
                "scores": [result_score(s.get("info")) for s in searches],
                "training_eligible": False,
            }
    finally:
        session.close()
    output_rows = [results[key] for key in sorted(results)]
    counts = {}
    for item in output_rows:
        counts[item["classification"]] = counts.get(item["classification"], 0) + 1
    write_json(args.out, {
        "budget_sec": args.deep_sec, "engine": "titanium-v17",
        "opening_book": "off", "weights": str(args.titanium_weights),
        "rows": output_rows, "n_rows": len(output_rows), "summary": counts,
        "elapsed_sec": time.perf_counter() - started, "training_eligible": False,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
