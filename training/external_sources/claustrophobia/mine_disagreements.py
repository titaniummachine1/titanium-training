#!/usr/bin/env python3
"""Mine evaluation-only Claustrophobia cross-play games for disagreement roots.

Does NOT relabel. Writes a quarantine candidate list for Titanium deep-search /
exact-oracle relabeling. Never marks rows training-eligible.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch2-results", type=Path, required=True)
    ap.add_argument("--candidate-results", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    def load(path: Path) -> list[dict]:
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows

    e2 = load(args.epoch2_results)
    cand = load(args.candidate_results)
    by_key = {}
    for r in e2:
        key = (r.get("game_idx"), tuple(r.get("opening_seed") or r.get("opening") or []), r.get("titanium_first"))
        by_key[key] = {"epoch2": r}
    for r in cand:
        key = (r.get("game_idx"), tuple(r.get("opening_seed") or r.get("opening") or []), r.get("titanium_first"))
        by_key.setdefault(key, {})["candidate"] = r

    roots = []
    stats = Counter()
    for key, pair in by_key.items():
        a = pair.get("epoch2")
        b = pair.get("candidate")
        if not a or not b:
            stats["unpaired"] += 1
            continue
        if a.get("termination") == "PROTOCOL_ERROR" or b.get("termination") == "PROTOCOL_ERROR":
            stats["protocol_skip"] += 1
            continue
        e2_win = a.get("winner_side") == "titanium"
        cand_win = b.get("winner_side") == "titanium"
        claustro_beat_both = a.get("winner_side") == "claustrophobia" and b.get("winner_side") == "claustrophobia"
        cand_ok_e2_fail = cand_win and not e2_win
        e2_ok_cand_fail = e2_win and not cand_win

        moves_a = a.get("moves") or []
        moves_b = b.get("moves") or []
        # First ply divergence as a crude disagreement root
        diverge_ply = None
        for i, (ma, mb) in enumerate(zip(moves_a, moves_b)):
            if ma != mb:
                diverge_ply = i
                break

        walls_a = [m for m in moves_a if len(m) == 3 and m[-1] in "hv"]
        unusual_walls = [m for m in walls_a if m[0] in "abih" or m[1] in "19"]

        if claustro_beat_both:
            stats["claustrophobia_wins_both"] += 1
        if cand_ok_e2_fail:
            stats["candidate_succeeds_epoch2_fails"] += 1
        if e2_ok_cand_fail:
            stats["epoch2_succeeds_candidate_fails"] += 1
        if diverge_ply is not None:
            stats["move_path_divergence"] += 1

        if not (claustro_beat_both or cand_ok_e2_fail or e2_ok_cand_fail or diverge_ply is not None):
            continue

        prefix = moves_a[: diverge_ply] if diverge_ply is not None else moves_a[: min(8, len(moves_a))]
        roots.append(
            {
                "evaluation_only": True,
                "training_eligible": False,
                "game_idx": key[0],
                "opening_seed": list(key[1]),
                "titanium_first": key[2],
                "epoch2_winner": a.get("winner_side"),
                "candidate_winner": b.get("winner_side"),
                "diverge_ply": diverge_ply,
                "prefix_moves": prefix,
                "epoch2_move_at_diverge": moves_a[diverge_ply] if diverge_ply is not None and diverge_ply < len(moves_a) else None,
                "candidate_move_at_diverge": moves_b[diverge_ply] if diverge_ply is not None and diverge_ply < len(moves_b) else None,
                "unusual_defensive_walls_epoch2": unusual_walls[:16],
                "tags": [
                    t
                    for t, cond in [
                        ("claustrophobia_win", claustro_beat_both),
                        ("candidate_beats_where_epoch2_fails", cand_ok_e2_fail),
                        ("epoch2_beats_where_candidate_fails", e2_ok_cand_fail),
                        ("path_divergence", diverge_ply is not None),
                    ]
                    if cond
                ],
                "relabeling_status": "pending_titanium_deep_search_or_exact_oracle",
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "purpose": "disagreement_roots_for_titanium_relabel",
        "do_not_train_until_relabeled": True,
        "n_roots": len(roots),
        "stats": dict(stats),
        "roots": roots,
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"n_roots": len(roots), "stats": dict(stats), "out": str(args.out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
