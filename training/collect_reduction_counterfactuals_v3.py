#!/usr/bin/env python3
"""Large-scale grouped LMR counterfactual collector — Phase 3.

Collects a minimum of 5 000 natural events (target 10 000–25 000) by
running realistic searches from diverse position phases.  Every row is
tagged with:

  source_tag: "natural" | "hard_negative"
  phase_tag:  "out_of_book" | "wall_heavy_mid" | "race_transition" |
              "late_mid" | "wall_endgame" | "complex_path"

Family identity for grouped split is source_game_key (SHA-256 of the
packed move sequence), so all events from the same game stay in one split.

Hard-negative mining runs a SEPARATE second pass over the natural event
file looking for:
  - decision changes (UNSAFE)
  - large counterfactual node explosions (cf_nodes > 5 × bl_nodes)
  - bound changes (EXACT ↔ FAIL_LOW / FAIL_HIGH)
  - high-depth, low-move-index candidates that look safe superficially

These are written to a SEPARATE output file; they must not be mixed with
the natural stream for calibration or prevalence estimation.

Isolation guarantee
-------------------
Each A/B pair uses:
  - a fresh 18-bit TT (engine enforces this internally)
  - zeroed history, killers, countermoves, counters, and stop state
  - no information transferred from A to B

This is identical to the Phase-2 isolator already in the engine.

Usage
-----
python training/collect_reduction_counterfactuals_v3.py \\
    --natural-target 10000 \\
    --out-dir training/data/lmr_phase3 \\
    --depth 8 \\
    --min-event-depth 6 \\
    --min-ply 11 \\
    --seed 777

python training/collect_reduction_counterfactuals_v3.py \\
    --hard-negative-pass \\
    --natural-file training/data/lmr_phase3/natural.jsonl \\
    --out-dir training/data/lmr_phase3 \\
    --depth 8 \\
    --min-event-depth 6 \\
    --seed 777
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Iterator

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))

from datagen import DB_PATH, load_games_from_db  # noqa: E402
from engine_identity import assert_engine_ready  # noqa: E402
from move_codec import pack_moves  # noqa: E402
from reduction_counterfactual_schema import (  # noqa: E402
    FEATURE_SCHEMA_V2,
    SCHEMA,
    classify_pair,
    context_features_v2,
    rank_percentile,
    stable_partition,
)

BIN = ROOT / "engine" / "target" / "release" / "titanium.exe"
WEIGHTS = ROOT / "engine" / "src" / "acev13" / "net_weights.bin"

# ── phase classification ─────────────────────────────────────────────────────

def classify_phase(ply: int, moves: list[str]) -> str:
    """Heuristic phase tag based on ply and move composition."""
    n_walls = sum(1 for m in moves if m[-1] in ("h", "v"))
    if ply < 12:
        return "out_of_book"
    if ply >= 60:
        return "wall_endgame"
    if ply >= 40:
        return "late_mid"
    if ply >= 25 and n_walls >= 6:
        return "wall_heavy_mid"
    if ply >= 25:
        return "race_transition"
    return "complex_path"


PHASE_PLY_RANGES: dict[str, tuple[int, int]] = {
    "out_of_book":     (11, 20),
    "wall_heavy_mid":  (20, 40),
    "complex_path":    (20, 35),
    "race_transition": (30, 50),
    "late_mid":        (40, 60),
    "wall_endgame":    (55, 90),
}

# ── engine interaction ────────────────────────────────────────────────────────

def engine_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT / "engine", capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def run_probe(
    moves: list[str],
    depth: int,
    limit: int,
    target: int | None = None,
    min_event_depth: int = 0,
) -> tuple[list[dict], dict]:
    cmd = [str(BIN), "reduction-probe", "--depth", str(depth),
           "--limit", str(limit)]
    if target is not None:
        cmd.extend(["--target", str(target)])
    if min_event_depth > 0:
        cmd.extend(["--min-event-depth", str(min_event_depth)])
    cmd.extend(moves)
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                            timeout=600)
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout)[-2000:])
    rows = [json.loads(ln) for ln in result.stdout.splitlines() if ln.startswith("{")]
    events = [r for r in rows if r.get("schema") == "reduction-probe-event-v1"]
    roots = [r for r in rows if r.get("schema") == "reduction-probe-root-v1"]
    if not roots:
        raise RuntimeError("probe produced no root record")
    return events, roots[-1]


# ── candidate generation ─────────────────────────────────────────────────────

def phase_balanced_prefixes(
    games,
    *,
    total_positions: int,
    seed: int,
    phase_weights: dict[str, float] | None = None,
) -> list[tuple[list[str], str, str, str]]:
    """Return (moves, outcome, source, game_key) tuples balanced across phases."""
    rng = random.Random(seed)
    if phase_weights is None:
        phase_weights = {
            "out_of_book":     1.0,
            "wall_heavy_mid":  1.5,
            "complex_path":    1.5,
            "race_transition": 1.0,
            "late_mid":        1.0,
            "wall_endgame":    0.5,
        }
    by_phase: dict[str, list] = {p: [] for p in phase_weights}
    for moves, outcome, source in games:
        game_key = hashlib.sha256(pack_moves(moves)).hexdigest()[:20]
        for phase, (lo, hi) in PHASE_PLY_RANGES.items():
            # Sample 2–4 plies per phase per game
            candidates = list(range(max(lo, 11), min(hi + 1, len(moves))))
            rng.shuffle(candidates)
            for ply in candidates[:3]:
                by_phase[phase].append((moves[:ply], outcome, source, game_key))

    total_weight = sum(phase_weights.values())
    result: list[tuple[list[str], str, str, str]] = []
    for phase, items in by_phase.items():
        n = int(total_positions * phase_weights[phase] / total_weight)
        rng.shuffle(items)
        result.extend(items[:n])
    rng.shuffle(result)
    return result


# ── row construction ──────────────────────────────────────────────────────────

def build_row(
    baseline: dict,
    counterfactual: dict,
    cf_root: dict,
    baseline_root: dict,
    moves: list[str],
    game_key: str,
    outcome: str,
    source: str,
    phase_tag: str,
    source_tag: str,
    trunk_hash: str,
    binary_hash: str,
    commit: str,
    split_seed: int,
    args_depth: int,
    args_minimum_nodes_saved: int,
    args_minimum_savings_ratio: float,
) -> dict:
    labels = classify_pair(
        baseline,
        counterfactual,
        minimum_nodes_saved=args_minimum_nodes_saved,
        minimum_savings_ratio=args_minimum_savings_ratio,
    )
    move = str(baseline["move"])
    mi = int(baseline["move_index"])
    n_legal = int(baseline.get("total_legal_moves", 128))
    raw_hist = int(baseline.get("history_score", 0))
    has_v2 = "total_legal_moves" in baseline and "history_score" in baseline

    return {
        "schema": SCHEMA,
        "feature_schema": FEATURE_SCHEMA_V2 if has_v2 else "halfpw-hidden32-search-context5-v1",
        "moves_bin": base64.b64encode(pack_moves(moves)).decode("ascii"),
        "source_game_key": game_key,
        "source": source,
        "outcome": outcome,
        "position_ply": len(moves),
        "phase_tag": phase_tag,
        "source_tag": source_tag,
        "parent_hash": baseline["parent_hash"],
        "child_hash": baseline["child_hash"],
        "move": move,
        "depth": baseline["depth"],
        "search_ply": baseline["ply"],
        "alpha": baseline["alpha"],
        "beta": baseline["beta"],
        "node_type": "ROOT" if int(baseline["ply"]) == 0 else "INTERNAL",
        "move_index": mi,
        "base_reduction": baseline["base_reduction"],
        "move_class": "wall_h" if move.endswith("h") else (
            "wall_v" if move.endswith("v") else "pawn"),
        "total_legal_moves": baseline.get("total_legal_moves"),
        "history_score": baseline.get("history_score"),
        "rank_percentile": rank_percentile(mi, n_legal) if has_v2 else None,
        "hidden32": baseline["hidden"],
        "context5": [
            min(max((int(baseline["depth"]) - 1) / 30.0, 0.0), 1.0),
            min(mi / 128.0, 1.0),
            min(int(baseline["base_reduction"]) / 4.0, 1.0),
            1.0 if move.endswith("h") else 0.0,
            1.0 if move.endswith("v") else 0.0,
        ],
        "context7": context_features_v2(baseline) if has_v2 else None,
        "baseline": baseline,
        "counterfactual": counterfactual,
        "baseline_root": baseline_root,
        "counterfactual_root": cf_root,
        "split": stable_partition(game_key, split_seed),
        "population": source_tag,
        "engine_commit": commit,
        "engine_binary_sha256": binary_hash,
        "trunk_sha256": trunk_hash,
        "collection": {
            "schema_version": 3,
            "depth": args_depth,
            "fixed_tt_bits": 18,
            "minimum_nodes_saved": args_minimum_nodes_saved,
            "minimum_savings_ratio": args_minimum_savings_ratio,
            "split_seed": split_seed,
        },
        **labels,
    }


# ── main natural collection ───────────────────────────────────────────────────

def collect_natural(args) -> None:
    games = load_games_from_db(Path(args.data))
    prefixes = phase_balanced_prefixes(
        games,
        total_positions=args.natural_target * 3,  # over-request; stop at target
        seed=args.seed,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "natural.jsonl"

    commit = engine_commit()
    trunk_hash = hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()
    binary_hash = hashlib.sha256(BIN.read_bytes()).hexdigest()
    rng = random.Random(args.seed ^ 0xDEADB00F)

    print(f"Trunk SHA-256 frozen: {trunk_hash[:32]}...")
    print(f"Target: {args.natural_target} natural events")
    print(f"Depth={args.depth}, min_event_depth={args.min_event_depth}, "
          f"min_ply={args.min_ply}")

    written = 0
    pos_tried = 0

    with out_path.open("a", encoding="utf-8") as handle:
        for moves, outcome, source, game_key in prefixes:
            if written >= args.natural_target:
                break
            if len(moves) < args.min_ply:
                continue
            pos_tried += 1
            phase_tag = classify_phase(len(moves), moves)
            try:
                baseline_events, baseline_root = run_probe(
                    moves, args.depth, args.event_scan_limit,
                    min_event_depth=args.min_event_depth,
                )
            except Exception as exc:
                print(f"  skip ply={len(moves)}: {exc}", file=sys.stderr)
                continue
            if not baseline_events:
                continue

            choices = list(baseline_events)
            rng.shuffle(choices)
            for baseline in choices[:args.samples_per_position]:
                try:
                    cf_events, cf_root = run_probe(
                        moves, args.depth, 1, int(baseline["ordinal"]),
                        min_event_depth=args.min_event_depth,
                    )
                    counterfactual = cf_events[0] if cf_events else {}
                except Exception as exc:
                    counterfactual = {}
                    cf_root = {}
                try:
                    row = build_row(
                        baseline, counterfactual, cf_root, baseline_root,
                        moves, game_key, outcome, source,
                        phase_tag=phase_tag,
                        source_tag="natural",
                        trunk_hash=trunk_hash,
                        binary_hash=binary_hash,
                        commit=commit,
                        split_seed=args.seed,
                        args_depth=args.depth,
                        args_minimum_nodes_saved=args.minimum_nodes_saved,
                        args_minimum_savings_ratio=args.minimum_savings_ratio,
                    )
                except Exception as exc:
                    print(f"  build_row error: {exc}", file=sys.stderr)
                    continue
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
                handle.flush()
                written += 1
                status = row.get("sample_status", "?")
                act = int(row.get("activate_plus_one", False))
                saved = row.get("net_nodes_saved", 0)
                print(
                    f"{written:6d} {status:<7} act={act} "
                    f"phase={phase_tag:<14} "
                    f"move={row['move']:<6} d={row['depth']} "
                    f"i={row['move_index']:3d} saved={saved:+d}"
                )
                if written >= args.natural_target:
                    break

    print(f"\nNatural collection done: {written} rows -> {out_path}")
    _print_summary(out_path)


# ── hard-negative mining pass ─────────────────────────────────────────────────

def is_hard_negative_candidate(baseline: dict) -> bool:
    """Return True for probe events that are likely hard negatives."""
    br = int(baseline.get("base_reduction", 0))
    mi = int(baseline.get("move_index", 0))
    d = int(baseline.get("depth", 0))
    nodes = int(baseline.get("nodes", 1))
    # High-depth + early move: these historically look safe but can be critical
    early_deep = d >= 6 and mi <= 10
    # Expensive scouts: baseline search was already costly
    expensive = nodes >= 30
    # High base reduction: biggest reduction tier
    big_red = br >= 3
    return early_deep or expensive or big_red


def collect_hard_negatives(args) -> None:
    if not args.natural_file:
        raise SystemExit("--natural-file required for --hard-negative-pass")
    nat_path = Path(args.natural_file)
    nat_rows = [json.loads(ln) for ln in nat_path.read_text(encoding="utf-8").splitlines() if ln]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "hard_negatives.jsonl"

    commit = engine_commit()
    trunk_hash = hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()
    binary_hash = hashlib.sha256(BIN.read_bytes()).hexdigest()
    rng = random.Random(args.seed ^ 0xBADC0FFE)

    print(f"Hard-negative mining from {nat_path} ({len(nat_rows)} rows)")
    print(f"Target: {args.hard_negative_target} hard-negative events")

    # Identify candidates from the natural file
    # We re-run the same position with targeted probe to replay the baseline event
    # and then run the counterfactual with extended probing.
    unsafe_found = 0
    written = 0

    with out_path.open("a", encoding="utf-8") as handle:
        rng.shuffle(nat_rows)
        for row in nat_rows:
            if written >= args.hard_negative_target:
                break
            if row.get("source_tag") not in ("natural", None):
                continue  # only re-probe natural events

            moves_bin = row.get("moves_bin")
            if not moves_bin:
                continue
            try:
                from move_codec import unpack_moves
                moves = unpack_moves(base64.b64decode(moves_bin))
            except Exception:
                continue

            # Re-run baseline for the specific ordinal
            target_ordinal = row.get("baseline", {}).get("ordinal")
            if target_ordinal is None:
                continue
            game_key = row["source_game_key"]
            outcome = row.get("outcome", "?")
            source = row.get("source", "?")
            phase_tag = row.get("phase_tag", "unknown")

            try:
                bl_events, bl_root = run_probe(
                    moves, args.depth, 1, int(target_ordinal),
                    min_event_depth=args.min_event_depth,
                )
                if not bl_events:
                    continue
                baseline = bl_events[0]
            except Exception as exc:
                print(f"  hn-baseline skip: {exc}", file=sys.stderr)
                continue

            if not is_hard_negative_candidate(baseline):
                continue

            try:
                cf_events, cf_root = run_probe(
                    moves, args.depth, 1, int(baseline["ordinal"]),
                    min_event_depth=args.min_event_depth,
                )
                counterfactual = cf_events[0] if cf_events else {}
            except Exception as exc:
                counterfactual = {}
                cf_root = {}

            try:
                new_row = build_row(
                    baseline, counterfactual, cf_root, bl_root,
                    moves, game_key, outcome, source,
                    phase_tag=phase_tag,
                    source_tag="hard_negative",
                    trunk_hash=trunk_hash,
                    binary_hash=binary_hash,
                    commit=commit,
                    split_seed=args.seed,
                    args_depth=args.depth,
                    args_minimum_nodes_saved=args.minimum_nodes_saved,
                    args_minimum_savings_ratio=args.minimum_savings_ratio,
                )
            except Exception as exc:
                print(f"  hn-build_row error: {exc}", file=sys.stderr)
                continue

            is_unsafe = new_row.get("sample_status") == "UNSAFE"
            is_big_cf = (int(new_row.get("counterfactual_nodes", 0)) >
                         5 * max(1, int(new_row.get("baseline_nodes", 1))))
            if not (is_unsafe or is_big_cf):
                continue  # not interesting as hard negative

            handle.write(json.dumps(new_row, separators=(",", ":")) + "\n")
            handle.flush()
            written += 1
            if is_unsafe:
                unsafe_found += 1
            print(
                f"HN {written:4d} {new_row['sample_status']:<7} "
                f"unsafe_found={unsafe_found} "
                f"cf_nodes={new_row.get('counterfactual_nodes', '?')} "
                f"move={new_row['move']}"
            )

    print(f"\nHard-negative mining done: {written} total, {unsafe_found} UNSAFE -> {out_path}")


# ── summary helpers ───────────────────────────────────────────────────────────

def _print_summary(path: Path) -> None:
    rows = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln]
    n = len(rows)
    n_safe = sum(1 for r in rows if r.get("sample_status") == "SAFE")
    n_unsafe = sum(1 for r in rows if r.get("sample_status") == "UNSAFE")
    n_unk = sum(1 for r in rows if r.get("sample_status") == "UNKNOWN")
    n_pos = sum(1 for r in rows if r.get("activate_plus_one"))
    splits = {}
    for r in rows:
        s = r.get("split", "?")
        splits[s] = splits.get(s, 0) + 1
    phases = {}
    for r in rows:
        p = r.get("phase_tag", "?")
        phases[p] = phases.get(p, 0) + 1
    print(f"\n  Summary: n={n} SAFE={n_safe} UNSAFE={n_unsafe} UNKNOWN={n_unk} pos={n_pos}")
    print(f"  Splits: {splits}")
    print(f"  Phases: {phases}")
    n_nonzero_hist = sum(1 for r in rows if (r.get("history_score") or 0) != 0)
    print(f"  Non-zero history_score: {n_nonzero_hist}/{n} ({100*n_nonzero_hist/max(1,n):.1f}%)")


# ── trunk lock ────────────────────────────────────────────────────────────────

def verify_trunk_unchanged(trunk_hash: str) -> None:
    current = hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()
    if current != trunk_hash:
        raise RuntimeError(
            f"Trunk hash changed during collection!\n"
            f"  expected: {trunk_hash}\n"
            f"  current:  {current}"
        )


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DB_PATH))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--min-event-depth", type=int, default=6)
    parser.add_argument("--min-ply", type=int, default=11)
    parser.add_argument("--event-scan-limit", type=int, default=256)
    parser.add_argument("--samples-per-position", type=int, default=3)
    parser.add_argument("--natural-target", type=int, default=10000)
    parser.add_argument("--minimum-nodes-saved", type=int, default=8)
    parser.add_argument("--minimum-savings-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=777)

    # Hard-negative pass
    parser.add_argument("--hard-negative-pass", action="store_true",
                        help="Run the hard-negative mining pass instead of natural collection")
    parser.add_argument("--natural-file",
                        help="Path to natural.jsonl for hard-negative mining")
    parser.add_argument("--hard-negative-target", type=int, default=200,
                        help="Stop mining when this many hard-negative events are found")

    args = parser.parse_args()

    assert_engine_ready(write_if_missing=True, parity=False)
    if not BIN.exists():
        raise SystemExit(f"missing {BIN}; build release binary first")

    trunk_hash_at_start = hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()
    print(f"=== LMR Counterfactual Collector v3 ===")
    print(f"Trunk SHA-256 (frozen at start): {trunk_hash_at_start}")

    if args.hard_negative_pass:
        collect_hard_negatives(args)
    else:
        collect_natural(args)

    verify_trunk_unchanged(trunk_hash_at_start)
    print(f"\nTrunk hash verified unchanged: {trunk_hash_at_start[:32]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
