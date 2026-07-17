#!/usr/bin/env python3
"""Post-process clean_v1 Claustrophobia benches — integrity + diagnostic export.

Does not alter openings/harness. Evaluation-only. No training import.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

EXPECTED_MANIFEST_SHA = "2245260f11224e5fa901208c1b6561431a929a4d9202da3a9f2000597bcd7502"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def first_move_disagreement(a_moves: list[str], b_moves: list[str]) -> dict | None:
    for i, (ma, mb) in enumerate(zip(a_moves, b_moves)):
        if ma != mb:
            return {"ply": i, "epoch2_move": ma, "candidate_move": mb, "prefix": a_moves[:i]}
    if len(a_moves) != len(b_moves):
        return {
            "ply": min(len(a_moves), len(b_moves)),
            "epoch2_move": None,
            "candidate_move": None,
            "prefix": a_moves[: min(len(a_moves), len(b_moves))],
            "note": "length_mismatch_after_common_prefix",
        }
    return None


def collapse_window(moves: list[str], plies_before: int = 6) -> list[str]:
    if len(moves) <= 2:
        return list(moves)
    start = max(0, len(moves) - plies_before)
    return moves[start:]


def main() -> int:
    root = Path(__file__).resolve().parent / "eval_games" / "clean_v1"
    e2_dir = root / "epoch2_vs_claustrophobia"
    cand_dir = root / "candidate_vs_claustrophobia"
    out = root / "diagnostic_report.json"

    e2_sum = json.loads((e2_dir / "summary.json").read_text(encoding="utf-8"))
    c_sum = json.loads((cand_dir / "summary.json").read_text(encoding="utf-8"))
    e2_rows = load_jsonl(e2_dir / "results.jsonl")
    c_rows = load_jsonl(cand_dir / "results.jsonl")
    e2_open = json.loads((e2_dir / "openings_used.json").read_text(encoding="utf-8"))
    c_open = json.loads((cand_dir / "openings_used.json").read_text(encoding="utf-8"))
    e2_deny = json.loads((e2_dir / "DENYLIST.json").read_text(encoding="utf-8"))
    c_deny = json.loads((cand_dir / "DENYLIST.json").read_text(encoding="utf-8"))

    checks = {
        "epoch2_games_20": e2_sum.get("games") == 20 and len(e2_rows) == 20,
        "candidate_games_20": c_sum.get("games") == 20 and len(c_rows) == 20,
        "epoch2_protocol_errors_0": e2_sum.get("protocol_errors", 1) == 0,
        "candidate_protocol_errors_0": c_sum.get("protocol_errors", 1) == 0,
        "both_clean_flag": bool(e2_sum.get("clean")) and bool(c_sum.get("clean")),
        "matching_manifest_sha": (
            e2_sum.get("openings_manifest_sha256")
            == c_sum.get("openings_manifest_sha256")
            == EXPECTED_MANIFEST_SHA
            == e2_open.get("manifest_sha256")
            == c_open.get("manifest_sha256")
        ),
        "matching_seed": e2_sum.get("seed") == c_sum.get("seed") == 1337,
        "matching_sims": e2_sum.get("sims") == c_sum.get("sims") == 20,
        "matching_time_sec": e2_sum.get("time_sec") == c_sum.get("time_sec"),
        "matching_opening_ids": [o["opening_id"] for o in e2_open["openings"]]
        == [o["opening_id"] for o in c_open["openings"]],
        "eval_only_denylist": e2_deny.get("do_not_train_on") and c_deny.get("do_not_train_on"),
    }
    integrity_ok = all(checks.values())

    per_game = []
    for e, c in zip(e2_rows, c_rows):
        em = e.get("moves") or []
        cm = c.get("moves") or []
        disagree = first_move_disagreement(em, cm)
        per_game.append(
            {
                "game_idx": e.get("game_idx"),
                "opening_id": e.get("opening_id"),
                "titanium_first": e.get("titanium_first"),
                "epoch2_winner": e.get("winner_side"),
                "candidate_winner": c.get("winner_side"),
                "epoch2_plies": e.get("plies"),
                "candidate_plies": c.get("plies"),
                "epoch2_walls": e.get("walls_in_moves"),
                "candidate_walls": c.get("walls_in_moves"),
                "epoch2_seconds": e.get("seconds"),
                "candidate_seconds": c.get("seconds"),
                "first_move_disagreement": disagree,
                "epoch2_pre_collapse_window": collapse_window(em, 6),
                "candidate_pre_collapse_window": collapse_window(cm, 6),
                "claustrophobia_beat_epoch2": e.get("winner_side") == "claustrophobia",
                "claustrophobia_beat_candidate": c.get("winner_side") == "claustrophobia",
                "candidate_survived_longer": (c.get("plies") or 0) > (e.get("plies") or 0),
            }
        )

    diag = {
        "role": "diagnostic_only_not_promotion",
        "integrity_ok": integrity_ok,
        "checks": checks,
        "epoch2_summary": e2_sum,
        "candidate_summary": c_sum,
        "compare": {
            "epoch2_score_vs_claustrophobia": e2_sum.get("titanium_wins", 0) / max(1, e2_sum.get("games", 1)),
            "candidate_score_vs_claustrophobia": c_sum.get("titanium_wins", 0) / max(1, c_sum.get("games", 1)),
            "epoch2_avg_plies": e2_sum.get("avg_plies"),
            "candidate_avg_plies": c_sum.get("avg_plies"),
            "epoch2_total_walls": e2_sum.get("total_walls"),
            "candidate_total_walls": c_sum.get("total_walls"),
            "epoch2_color_split": e2_sum.get("color_split"),
            "candidate_color_split": c_sum.get("color_split"),
            "n_games_with_path_disagreement": sum(
                1 for g in per_game if g.get("first_move_disagreement")
            ),
            "n_candidate_survived_longer": sum(1 for g in per_game if g.get("candidate_survived_longer")),
        },
        "per_game": per_game,
        "weights": {
            "epoch2": e2_deny.get("titanium_weights"),
            "candidate": c_deny.get("titanium_weights"),
        },
        "note": (
            "20-game Claustrophobia screen is diagnostic only. "
            "Candidate stays quarantined due to frozen-anchor failure. "
            "Do not overwrite this clean_v1 20-sim benchmark with lower-sim runs."
        ),
    }
    out.write_text(json.dumps(diag, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"integrity_ok": integrity_ok, "out": str(out), "compare": diag["compare"]}, indent=2))
    return 0 if integrity_ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
