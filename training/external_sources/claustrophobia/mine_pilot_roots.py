#!/usr/bin/env python3
"""Mine ranked Claustrophobia decision roots from paired pilot games."""
from __future__ import annotations
import argparse, hashlib, json, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "training"))
from diversity.claustrophobia_rows import MAX_ROWS_PER_FORK_LINEAGE, MAX_ROWS_PER_SOURCE_GAME
from diversity.eval_denylist import is_evaluation_leakage

def flat_moves(actions):
    return [a.get("move") if isinstance(a, dict) else a for a in (actions or [])]

def full_prefix_canonical_key(prefix: list[str], side_to_move: int) -> str:
    """Dedup key over the FULL move prefix.

    Do not use reflection_canonical_position_key(moves_prefix=...) here: that
    helper truncates to the first two plies and collapses distinct midgame
    positions into a handful of opening keys.
    """
    payload = {
        "moves_prefix": " ".join(prefix),
        "side_to_move": int(side_to_move),
        "canonical_state_version": "full-prefix-v1",
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]

def wall_moves(moves):
    return [m for m in moves if isinstance(m, str) and len(m) == 3 and m[-1] in "hv"]

def candidate_priority(row):
    tags = set(row["tags"])
    return (1000 * bool({"candidate_succeeds_epoch2_fails", "epoch2_succeeds_candidate_fails"} & tags)
            + 800 * ("candidate_epoch2_divergence" in tags)
            + 500 * bool({"unusual_defensive_walls", "wall_race"} & tags)
            + 300 * ("claustrophobia_later_wins" in tags)
            + 100 * (8 <= row["ply"] <= 50) + 50 * row["is_wall_move"]
            - 10 * max(0, 4 - row["ply"]))

def select_ranked_roots(candidates):
    """Canonical-dedup and enforce caps while balancing side-to-move."""
    remaining = sorted(candidates, key=lambda r: (-candidate_priority(r), r["source_game_id"], r["ply"]))
    selected, seen = [], set()
    per_game, per_lineage, stm = Counter(), Counter(), Counter()
    n_dup_rejected = 0
    n_cap_rejected = 0
    while remaining:
        target = 0 if stm[0] <= stm[1] else 1
        eligible = [r for r in remaining if r["side_to_move"] == target] or remaining
        row = eligible[0]
        remaining.remove(row)
        if row["canonical_key"] in seen:
            n_dup_rejected += 1
            continue
        if per_game[row["source_game_id"]] >= MAX_ROWS_PER_SOURCE_GAME:
            n_cap_rejected += 1
            continue
        if per_lineage[row["fork_lineage_id"]] >= MAX_ROWS_PER_FORK_LINEAGE:
            n_cap_rejected += 1
            continue
        seen.add(row["canonical_key"])
        row["root_id"] = f"mine-root-{len(selected):04d}"
        selected.append(row)
        per_game[row["source_game_id"]] += 1
        per_lineage[row["fork_lineage_id"]] += 1
        stm[row["side_to_move"]] += 1
    return selected, {
        "n_dup_rejected": n_dup_rejected,
        "n_cap_rejected": n_cap_rejected,
        "stm": dict(stm),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-roots", type=int, default=0)
    args = ap.parse_args()
    rows = [json.loads(x) for x in args.results.read_text(encoding="utf-8").splitlines() if x.strip()]
    by_idx = {}
    for row in rows:
        by_idx.setdefault(row.get("game_index"), {})[row.get("style")] = row
    raw, leakage_rejected = [], 0
    for idx, pair in by_idx.items():
        epoch2, candidate = pair.get("epoch2"), pair.get("candidate")
        if not epoch2 or not candidate:
            continue
        games = (epoch2, candidate)
        moves_by_style = {g.get("style"): flat_moves(g.get("actions")) for g in games}
        div = next((i for i, (x, y) in enumerate(zip(moves_by_style["epoch2"], moves_by_style["candidate"])) if x != y), None)
        if div is None and len(moves_by_style["epoch2"]) != len(moves_by_style["candidate"]):
            div = min(len(moves_by_style["epoch2"]), len(moves_by_style["candidate"]))
        specs = []
        if div is not None:
            specs.append((epoch2, moves_by_style["epoch2"][:div], moves_by_style["epoch2"][div], moves_by_style["candidate"][div], "candidate_epoch2_divergence"))
        opening_len = max(len(g.get("opening_moves") or g.get("opening") or []) for g in games)
        for game in games:
            actions = game.get("actions") or []
            moves = flat_moves(actions)
            for ply, action in enumerate(actions):
                if ply >= opening_len and isinstance(action, dict) and action.get("side") == "claustrophobia":
                    specs.append((game, moves[:ply], action.get("move"), None, "claustrophobia_ply_candidate"))
        for source, prefix, own_action, other_action, kind in specs:
            gid = source.get("source_game_id") or f"{source.get('run_id', 'unknown')}:{source.get('style')}:{idx}"
            lineage = source.get("fork_lineage_id") or gid
            key = full_prefix_canonical_key(prefix, len(prefix) % 2)
            leaked, _ = is_evaluation_leakage(canonical_key=key, lineage_id=lineage)
            if leaked:
                leakage_rejected += 1
                continue
            walls = wall_moves(prefix)
            unusual = [m for m in walls if m[0] in "abih" or m[1] in "19"]
            tags = [kind]
            if epoch2.get("winner_side") != candidate.get("winner_side"):
                tags.append("paired_style_outcome_disagreement")
                # winner_side is from Titanium's perspective in the harness.
                if epoch2.get("winner_side") == "titanium" and candidate.get("winner_side") != "titanium":
                    tags.append("epoch2_succeeds_candidate_fails")
                if candidate.get("winner_side") == "titanium" and epoch2.get("winner_side") != "titanium":
                    tags.append("candidate_succeeds_epoch2_fails")
            if source.get("winner_side") == "claustrophobia":
                tags.append("claustrophobia_later_wins")
            if unusual:
                tags.append("unusual_defensive_walls")
            if len(walls) >= 4:
                tags.append("wall_race")
            is_wall = isinstance(own_action, str) and len(own_action) == 3 and own_action[-1] in "hv"
            if is_wall:
                tags.append("wall_move")
            provenance = {k: source.get(k) for k in ("run_id", "claustrophobia_release_tag",
                "claustrophobia_checkpoint_sha256", "repository_commit", "titanium_binary_sha256",
                "titanium_weights_sha256", "generation_config")}
            game_index = int(source.get("game_index", idx) or idx)
            session = ("pre_reset" if game_index < 37 else "post_reset") if source.get("style") == "epoch2" else "post_reset"
            raw.append({
                "source_game_id": gid, "game_index": source.get("game_index", idx), "style": source.get("style"),
                "source_kind": "claustrophobia_pilot_crossplay",
                "run_id": source.get("run_id"), "source_run_id": source.get("run_id"),
                "source_process_session": session, "process_session": session, "opening_id": source.get("opening_id"),
                "opening_seed_id": source.get("opening_id"), "winner_side": source.get("winner_side"),
                "epoch2_winner": epoch2.get("winner_side"), "candidate_winner": candidate.get("winner_side"),
                "titanium_first": source.get("titanium_first"), "plies": source.get("plies"), "seconds": source.get("seconds"),
                "fork_lineage_id": lineage, "ply": len(prefix), "prefix_moves": prefix,
                "candidate_action": other_action, "epoch2_action": own_action,
                "claustrophobia_action": own_action if kind == "claustrophobia_ply_candidate" else (other_action or own_action),
                "canonical_key": key, "semantic_hash": hashlib.sha256((" ".join(prefix)).encode()).hexdigest(),
                "side_to_move": len(prefix) % 2, "phase": "opening" if len(prefix) < 8 else ("middlegame" if len(prefix) <= 50 else "end"),
                "tension": {"wall_count": len(walls), "kind": "wall" if walls else "race"}, "wall_count": len(walls),
                "wall_stocks": source.get("wall_stocks"), "is_wall_move": is_wall, "tags": sorted(set(tags)),
                "unusual_defensive_walls": unusual[:16], "training_eligible": False, "evaluation_eligible": False,
                "relabeling_status": "PENDING", "exact_label_kind": "unavailable", "provenance": provenance,
                "provenance_complete": all(provenance.values()),
            })
    selected, selection = select_ranked_roots(raw)
    if args.max_roots:
        selected = selected[:args.max_roots]
    out = {"roots": selected, "n_roots": len(selected), "n_raw_before_dedup": len(raw),
           "n_dup_rejected": selection["n_dup_rejected"],
           "n_cap_rejected": selection.get("n_cap_rejected", 0),
           "n_leakage_rejected": leakage_rejected,
           "canonical_key_scheme": "full-prefix-v1",
           "by_style": dict(Counter(r.get("style") for r in selected)), "stm": selection["stm"],
           "phase": dict(Counter(r.get("phase") for r in selected)),
           "caps": {"per_game": MAX_ROWS_PER_SOURCE_GAME, "per_lineage": MAX_ROWS_PER_FORK_LINEAGE},
           "training_eligible": False, "opening_book": "off",
           "provenance_complete_definition": "all mandatory source fields non-null"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
