#!/usr/bin/env python3
"""Queue Titanium deep-search / exact-oracle relabel for disagreement roots.

Evaluation roots stay evaluation-only until this script marks them relabeled.
Does not write into labels.db. Outputs a capped external-lane candidate JSONL
with training_eligible=false until a human/coordinator accepts the lane.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "training"))

from engine_session import EngineSession  # noqa: E402
from diversity.claustrophobia_rows import (
    MAX_ROWS_PER_FORK_LINEAGE,
    MAX_ROWS_PER_SOURCE_GAME,
    enforce_pilot_caps,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", type=Path, required=True)
    ap.add_argument("--titanium-bin", type=Path, required=True)
    ap.add_argument("--titanium-weights", type=Path, required=True)
    ap.add_argument("--time-sec", type=float, default=5.0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-roots", type=int, default=32)
    args = ap.parse_args()

    data = json.loads(args.roots.read_text(encoding="utf-8"))
    roots = data.get("roots") or []
    sess = EngineSession("titanium-v17", args.titanium_weights, threads=1, engine_bin=args.titanium_bin)
    out_rows = []
    per_game: dict[str, int] = {}
    try:
        for i, root in enumerate(roots[: args.max_roots]):
            prefix = list(root.get("prefix_moves") or [])
            gid = str(root.get("game_idx"))
            if per_game.get(gid, 0) >= MAX_ROWS_PER_SOURCE_GAME:
                continue
            if not sess.sync(prefix):
                out_rows.append({**root, "relabeling_status": "sync_failed", "training_eligible": False})
                continue
            mv = sess.go(args.time_sec)
            row = {
                **root,
                "dataset_kind": "relabeled_forks",
                "titanium_best_move": mv,
                "action_disagreement": mv is not None
                and mv != root.get("epoch2_move_at_diverge")
                and mv != root.get("candidate_move_at_diverge"),
                "relabeling_status": "titanium_deep_search_done" if mv else "search_failed",
                "search_time_sec": args.time_sec,
                "evaluation_eligible": False,
                "training_eligible": False,  # requires separate acceptance into capped lane
                "provenance_complete": True,
            }
            caps = enforce_pilot_caps(
                total_pilot_rows=max(100, len(out_rows) * 20),
                claustrophobia_rows=len(out_rows) + 1,
                rows_for_source_game=per_game.get(gid, 0) + 1,
                rows_for_fork_lineage=1,
            )
            row["pilot_cap_blockers"] = caps
            out_rows.append(row)
            per_game[gid] = per_game.get(gid, 0) + 1
    finally:
        sess.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "purpose": "claustrophobia_external_lane_candidates",
        "do_not_import_to_labels_db_until_accepted": True,
        "max_rows_per_source_game": MAX_ROWS_PER_SOURCE_GAME,
        "max_rows_per_fork_lineage": MAX_ROWS_PER_FORK_LINEAGE,
        "pilot_cap_fraction": 0.05,
        "n_rows": len(out_rows),
        "rows": out_rows,
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"n_rows": len(out_rows), "out": str(args.out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
