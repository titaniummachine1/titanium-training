#!/usr/bin/env python3
"""Collect complete-pipeline +1 LMR counterfactual labels from native Titanium."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))

from datagen import DB_PATH, load_games_from_db  # noqa: E402
from engine_identity import assert_engine_ready  # noqa: E402
from move_codec import pack_moves  # noqa: E402
from reduction_counterfactual_schema import (  # noqa: E402
    FEATURE_SCHEMA,
    FEATURE_SCHEMA_V2,
    SCHEMA,
    classify_pair,
    context_features_v2,
    rank_percentile,
    stable_partition,
)

BIN = ROOT / "engine" / "target" / "release" / "titanium.exe"
DEFAULT_OUT = ROOT / "training" / "data" / "reduction_counterfactuals.jsonl"
WEIGHTS = ROOT / "engine" / "src" / "acev13" / "net_weights.bin"


def engine_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT / "engine", capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def run_probe(moves: list[str], depth: int, limit: int, target: int | None = None, min_event_depth: int = 0) -> tuple[list[dict], dict]:
    command = [str(BIN), "reduction-probe", "--depth", str(depth), "--limit", str(limit)]
    if target is not None:
        command.extend(("--target", str(target)))
    if min_event_depth > 0:
        command.extend(("--min-event-depth", str(min_event_depth)))
    command.extend(moves)
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=300)
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout)[-2000:])
    rows = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
    events = [row for row in rows if row.get("schema") == "reduction-probe-event-v1"]
    roots = [row for row in rows if row.get("schema") == "reduction-probe-root-v1"]
    if not roots:
        raise RuntimeError("probe produced no root record")
    return events, roots[-1]


def context_features(event: dict) -> list[float]:
    """5-element context vector (context5 / FEATURE_SCHEMA v1)."""
    move = str(event["move"])
    return [
        min(max((int(event["depth"]) - 1) / 30.0, 0.0), 1.0),
        min(int(event["move_index"]) / 128.0, 1.0),
        min(int(event["base_reduction"]) / 4.0, 1.0),
        1.0 if move.endswith("h") else 0.0,
        1.0 if move.endswith("v") else 0.0,
    ]




def candidate_prefixes(games, *, count: int, min_ply: int, max_ply: int, seed: int):
    rng = random.Random(seed)
    candidates = []
    for moves, outcome, source in games:
        game_key = hashlib.sha256(pack_moves(moves)).hexdigest()[:20]
        hi = min(len(moves) - 1, max_ply)
        for ply in range(min_ply, hi + 1):
            candidates.append((moves[:ply], outcome, source, game_key))
    rng.shuffle(candidates)
    return candidates[:count]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DB_PATH))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--positions", type=int, default=8)
    parser.add_argument("--samples-per-position", type=int, default=2)
    parser.add_argument("--event-scan-limit", type=int, default=128)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--min-ply", type=int, default=8)
    parser.add_argument("--max-ply", type=int, default=70)
    parser.add_argument("--minimum-nodes-saved", type=int, default=8)
    parser.add_argument("--minimum-savings-ratio", type=float, default=0.05)
    parser.add_argument("--min-event-depth", type=int, default=0,
                        help="Only record probe events at local LMR depth >= this value. "
                             "Use 5+ to avoid post-order filling with leaf-level events.")
    parser.add_argument("--population", choices=("natural", "stratified"), default="natural")
    parser.add_argument("--proposal-source", default="native-runtime")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    assert_engine_ready(write_if_missing=True, parity=False)
    if not BIN.exists():
        raise SystemExit(f"missing {BIN}; build release binary first")
    games = load_games_from_db(Path(args.data))
    prefixes = candidate_prefixes(
        games, count=args.positions, min_ply=args.min_ply, max_ply=args.max_ply, seed=args.seed
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    commit = engine_commit()
    trunk_hash = hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()
    binary_hash = hashlib.sha256(BIN.read_bytes()).hexdigest()
    rng = random.Random(args.seed ^ 0x52454455)
    written = 0

    with out.open("a", encoding="utf-8") as handle:
        for moves, outcome, source, game_key in prefixes:
            try:
                baseline_events, baseline_root = run_probe(
                    moves, args.depth, args.event_scan_limit,
                    min_event_depth=args.min_event_depth,
                )
            except Exception as exc:
                print(f"skip baseline ply={len(moves)}: {exc}", file=sys.stderr)
                continue
            if not baseline_events:
                continue
            choices = list(baseline_events)
            if args.population == "stratified":
                # Oversample pipelines with real scout work; one-node scouts
                # cannot repay inference and dominate the natural population.
                expensive = sorted(choices, key=lambda row: int(row["nodes"]), reverse=True)
                high_count = max(1, args.samples_per_position // 2)
                selected = expensive[:high_count]
                remaining = [row for row in choices if row not in selected]
                rng.shuffle(remaining)
                choices = selected + remaining
            else:
                rng.shuffle(choices)
            for baseline in choices[: args.samples_per_position]:
                try:
                    cf_events, cf_root = run_probe(moves, args.depth, 1, int(baseline["ordinal"]),
                                                    min_event_depth=args.min_event_depth)
                    counterfactual = cf_events[0] if cf_events else {}
                    labels = classify_pair(
                        baseline,
                        counterfactual,
                        minimum_nodes_saved=args.minimum_nodes_saved,
                        minimum_savings_ratio=args.minimum_savings_ratio,
                    )
                except Exception as exc:
                    counterfactual = {}
                    cf_root = {}
                    labels = {
                        "sample_status": "UNKNOWN",
                        "status_reason": f"probe_error:{type(exc).__name__}",
                        "decision_preserved": False,
                        "safe_plus_one_reduction": False,
                        "worthwhile_net_savings": False,
                        "activate_plus_one": False,
                    }
                move = str(baseline["move"])
                has_v2_fields = "total_legal_moves" in baseline and "history_score" in baseline
                row = {
                    "schema": SCHEMA,
                    "feature_schema": FEATURE_SCHEMA_V2 if has_v2_fields else FEATURE_SCHEMA,
                    "moves_bin": base64.b64encode(pack_moves(moves)).decode("ascii"),
                    "source_game_key": game_key,
                    "source": source,
                    "outcome": outcome,
                    "position_ply": len(moves),
                    "parent_hash": baseline["parent_hash"],
                    "child_hash": baseline["child_hash"],
                    "move": move,
                    "depth": baseline["depth"],
                    "search_ply": baseline["ply"],
                    "alpha": baseline["alpha"],
                    "beta": baseline["beta"],
                    "node_type": "ROOT" if int(baseline["ply"]) == 0 else "INTERNAL",
                    "move_index": baseline["move_index"],
                    "base_reduction": baseline["base_reduction"],
                    "move_class": "wall_h" if move.endswith("h") else "wall_v",
                    "legal_move_bucket": None,
                    "path_cutting": None,
                    "bottleneck": None,
                    "total_legal_moves": baseline.get("total_legal_moves"),
                    "history_score": baseline.get("history_score"),
                    "rank_percentile": (
                        rank_percentile(int(baseline["move_index"]), int(baseline["total_legal_moves"]))
                        if has_v2_fields else None
                    ),
                    "hidden32": baseline["hidden"],
                    "context5": context_features(baseline),
                    "context7": context_features_v2(baseline) if has_v2_fields else None,
                    "baseline": baseline,
                    "counterfactual": counterfactual,
                    "baseline_root": baseline_root,
                    "counterfactual_root": cf_root,
                    "proposal_source": args.proposal_source,
                    "zero_ink": None,
                    "population": args.population,
                    "split": stable_partition(game_key, args.seed),
                    "engine_commit": commit,
                    "engine_binary_sha256": binary_hash,
                    "trunk_sha256": trunk_hash,
                    "collection": {
                        "depth": args.depth,
                        "fixed_tt_bits": 18,
                        "minimum_nodes_saved": args.minimum_nodes_saved,
                        "minimum_savings_ratio": args.minimum_savings_ratio,
                        "seed": args.seed,
                    },
                    **labels,
                }
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
                handle.flush()
                written += 1
                print(
                    f"{written:4d} {row['sample_status']:<7} activate={int(row['activate_plus_one'])} "
                    f"move={move} d={row['depth']} i={row['move_index']} "
                    f"saved={row.get('net_nodes_saved', 0):+d}"
                )
    print(f"wrote {written} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
