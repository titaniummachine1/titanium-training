#!/usr/bin/env python3
"""Build deterministic, non-training label-quality audit samples."""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from label_quality_common import SEED, load_json, rows_from, stratified_sample, write_json


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mined-roots", type=Path, required=True)
    ap.add_argument("--relabeled-roots", type=Path, required=True)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--resume-boundary", type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--heldout-fraction", type=float, default=0.20)
    ap.add_argument("--deep-size", type=int, default=100)
    ap.add_argument("--random-size", type=int, default=200)
    args = ap.parse_args()

    roots = rows_from(load_json(args.mined_roots, []))
    relabeled = rows_from(load_json(args.relabeled_roots, []))
    # Read results intentionally: it is the source-game inventory fallback and
    # ensures sampling never silently ignores a completed game manifest.
    results = []
    if args.results.exists():
        with args.results.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    value = json.loads(line)
                    if isinstance(value, dict):
                        results.append(value)
                except json.JSONDecodeError:
                    continue
    source_groups: dict[str, list[dict]] = defaultdict(list)
    for row in roots:
        source_groups[str(row.get("source_game_id", ""))].append(row)
    source_ids = sorted(k for k in source_groups if k)
    rng = random.Random(SEED)
    shuffled = list(source_ids)
    rng.shuffle(shuffled)
    heldout_n = max(1, round(len(source_ids) * args.heldout_fraction))
    initially_selected = set(shuffled[:heldout_n])
    selected_lineages = {
        str(row.get("fork_lineage_id") or row.get("source_game_id"))
        for source_id in initially_selected for row in source_groups[source_id]
    }
    # Expand to whole lineage components so a fork can never straddle the
    # held-out/training-proposal boundary.
    heldout_ids = sorted(
        source_id for source_id in source_ids
        if source_id in initially_selected or any(
            str(row.get("fork_lineage_id") or row.get("source_game_id")) in selected_lineages
            for row in source_groups[source_id]
        )
    )
    heldout_lineage = sorted({
        str(row.get("fork_lineage_id") or row.get("source_game_id"))
        for source_id in heldout_ids for row in source_groups[source_id]
    })
    heldout_set = set(heldout_ids)
    heldout_rows = [row for source_id in heldout_ids for row in source_groups[source_id]]
    relabeled_ids = {str(row.get("root_id")) for row in relabeled}

    stable = [row for row in relabeled if row.get("label_kind") == "STABLE_SEARCH"]
    deep_pool = [row for row in stable if str(row.get("source_game_id")) not in heldout_set]
    deep_note = None
    if len(deep_pool) < args.deep_size:
        deep_note = f"Only {len(deep_pool)} stable rows remain after held-out exclusion; filled from held-out rows."
        deep_pool += [row for row in stable if str(row.get("source_game_id")) in heldout_set]
    deep_sample, deep_strata = stratified_sample(
        deep_pool, args.deep_size, ("style", "phase", "side_to_move", "move_type", "score_margin", "paired_fork", "wall_count")
    )
    for row in deep_sample:
        row["training_eligible"] = False
        row["audit_sample"] = "deep_check"
        row["paired_fork"] = row.get("titanium_best") != row.get("claustrophobia_action")

    random_pool = [
        row for row in roots
        if str(row.get("root_id")) not in relabeled_ids
        and str(row.get("source_game_id")) not in heldout_set
    ]
    random_sample, random_strata = stratified_sample(
        random_pool, args.random_size, ("phase", "style", "move_type", "side_to_move", "wall_count")
    )
    for row in random_sample:
        row["training_eligible"] = False
        row["audit_sample"] = "random_bias"

    write_json(args.out_dir / "heldout_games.json", {
        "seed": SEED, "fraction": args.heldout_fraction,
        "source_game_ids": heldout_ids, "lineage_ids": heldout_lineage,
        "rows": heldout_rows, "n_source_games": len(heldout_ids),
        "n_lineages": len(heldout_lineage), "training_eligible": False,
    })
    write_json(args.out_dir / "deep_check_sample.json", {
        "seed": SEED, "requested_size": args.deep_size, "rows": deep_sample,
        "n_rows": len(deep_sample), "strata_counts": deep_strata,
        "heldout_exclusion_note": deep_note, "training_eligible": False,
    })
    write_json(args.out_dir / "random_bias_sample.json", {
        "seed": SEED, "requested_size": args.random_size, "rows": random_sample,
        "n_rows": len(random_sample), "pool_size": len(random_pool),
        "strata_counts": random_strata, "training_eligible": False,
    })
    boundary = load_json(args.resume_boundary, None) if args.resume_boundary else None
    write_json(args.out_dir / "SAMPLING_PLAN.json", {
        "seed": SEED, "heldout_fraction": args.heldout_fraction,
        "source_game_count": len(source_ids), "results_record_count": len(results),
        "heldout_source_game_ids": heldout_ids, "heldout_lineage_ids": heldout_lineage,
        "deep_size": len(deep_sample), "deep_strata_counts": deep_strata,
        "deep_pool_size": len(deep_pool), "random_size": len(random_sample),
        "random_pool_size": len(random_pool), "random_strata_counts": random_strata,
        "resume_boundary_loaded": boundary is not None,
        "training_eligible": False,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
