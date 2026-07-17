#!/usr/bin/env python3
"""Calibrate a stratified, held-out-free sample of 16s survivors at 64s."""
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
from label_quality_common import load_json, margin_cp, move_type, result_score, rows_from, stratified_sample, write_json


def classify(sixteen: dict, sixtyfour: dict) -> tuple[str, str | None]:
    if sixteen.get("bestmove") != sixtyfour.get("bestmove"):
        return "FLIPS_AT_64S", None
    margin = margin_cp((sixtyfour.get("info") or {}))
    if margin is not None and margin < 50.0:
        return "LOW_MARGIN_64S", None
    return "SURVIVES_64S", "margin_unavailable" if margin is None else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--priority-results", type=Path, required=True)
    ap.add_argument("--heldout", type=Path, required=True)
    ap.add_argument("--sample-out", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--titanium-bin", type=Path, required=True)
    ap.add_argument("--titanium-weights", type=Path, required=True)
    ap.add_argument("--deep-sec", type=float, default=64.0)
    ap.add_argument("--sample-size", type=int, default=50)
    ap.add_argument("--max", type=int, default=0)
    args = ap.parse_args()
    os.environ["TITANIUM_BOOK_MODE"] = "off"
    priority = rows_from(load_json(args.priority_results, {}))
    heldout = load_json(args.heldout, {}) or {}
    blocked_games = set(heldout.get("source_game_ids", []))
    blocked_lineages = set(heldout.get("lineage_ids", []))
    eligible = [
        row for row in priority
        if row.get("classification") == "SURVIVES_16S"
        and row.get("source_game_id") not in blocked_games
        and row.get("fork_lineage_id") not in blocked_lineages
    ]
    sample_payload = load_json(args.sample_out, {}) or {}
    sample = rows_from(sample_payload)
    if not sample:
        sample, strata = stratified_sample(
            eligible, args.sample_size,
            ("style", "source_kind", "phase", "move_type", "paired_fork", "score_margin"),
        )
        write_json(args.sample_out, {
            "purpose": "survives_16s_calibration",
            "sample_size_requested": args.sample_size,
            "heldout_excluded": True,
            "strata": strata,
            "rows": sample,
            "n_rows": len(sample),
            "training_eligible": False,
        })
    selected = {str(row.get("root_id")): row for row in sample}
    existing = load_json(args.out, {}) or {}
    results = {str(row.get("root_id")): row for row in rows_from(existing)}
    todo = [row for row in sample if str(row.get("root_id")) not in results]
    if args.max:
        todo = todo[:args.max]
    started = time.perf_counter()
    session = EngineSession("titanium-v17", args.titanium_weights, engine_bin=args.titanium_bin)
    try:
        for row in todo:
            searches = list(row.get("searches") or [])
            status = "ok"
            if len(searches) < 3:
                status = "missing_16s_search"
            elif not session.sync(row.get("prefix_moves") or []):
                status = "protocol_error"
            else:
                sixtyfour = session.go_detailed(args.deep_sec)
                classification, note = classify(searches[2], sixtyfour)
                searches.append(sixtyfour)
            if status != "ok":
                classification, note = "UNAVAILABLE_64S", status
            results[str(row.get("root_id"))] = {
                "root_id": row.get("root_id"),
                "source_game_id": row.get("source_game_id"),
                "fork_lineage_id": row.get("fork_lineage_id"),
                "style": row.get("style"),
                "phase": row.get("phase"),
                "wall_count": row.get("wall_count"),
                "move_type": row.get("move_type", move_type(row)),
                "paired_fork": row.get("paired_fork"),
                "searches": searches,
                "sixteen_move": searches[2].get("bestmove") if len(searches) > 2 else None,
                "sixtyfour_move": searches[3].get("bestmove") if len(searches) > 3 else None,
                "scores": [result_score(s.get("info")) for s in searches],
                "classification": classification,
                "classification_note": note,
                "status": status,
                "training_eligible": False,
            }
            write_json(args.out, {
                "budget_sec": args.deep_sec, "engine": "titanium-v17",
                "opening_book": "off", "weights": str(args.titanium_weights),
                "sample_path": str(args.sample_out),
                "rows": [results[key] for key in sorted(results)],
                "n_rows": len(results), "resumable": True,
                "training_eligible": False,
            })
    finally:
        session.close()
    counts: dict[str, int] = {}
    for row in results.values():
        counts[row["classification"]] = counts.get(row["classification"], 0) + 1
    write_json(args.out, {
        "budget_sec": args.deep_sec, "engine": "titanium-v17",
        "opening_book": "off", "weights": str(args.titanium_weights),
        "sample_path": str(args.sample_out),
        "rows": [results[key] for key in sorted(results)],
        "n_rows": len(results), "summary": counts,
        "elapsed_sec": time.perf_counter() - started, "resumable": True,
        "training_eligible": False,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
