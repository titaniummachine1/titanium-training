#!/usr/bin/env python3
"""Adaptive Wilson-score gate for the broke-side binary match."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable


def wilson_lower(successes: float, n: int, z: float = 1.96) -> float:
    if n <= 0:
        return 0.0
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = p + z2 / (2.0 * n)
    margin = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n)
    return (center - margin) / denom


def wilson_upper(successes: float, n: int, z: float = 1.96) -> float:
    if n <= 0:
        return 1.0
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = p + z2 / (2.0 * n)
    margin = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n)
    return (center + margin) / denom


def next_even_target(current: int) -> int:
    target = max(current + 2, math.ceil(current * 1.5))
    return target if target % 2 == 0 else target + 1


def decide(
    a_wins: int, b_wins: int, draws: int, current_games: int, max_games: int = 450
) -> dict:
    n = a_wins + b_wins + draws
    score = (a_wins + 0.5 * draws) / n if n else 0.0
    lb = wilson_lower(a_wins + 0.5 * draws, n)
    ub = wilson_upper(a_wins + 0.5 * draws, n)
    elo = -400.0 * math.log10(1.0 / score - 1.0) if 0.0 < score < 1.0 else None
    if lb > 0.5:
        decision, reason = "KEEP", f"proven stronger: Wilson lower bound {lb:.4f} > 0.5000"
    elif lb >= 0.5:
        decision, reason = "KEEP", f"proven not worse: Wilson lower bound {lb:.4f} >= 0.5000"
    elif ub < 0.5:
        decision, reason = "REJECT", f"proven worse: Wilson upper bound {ub:.4f} < 0.5000"
    elif current_games >= max_games:
        decision = "KEEP" if score >= 0.5 else "REJECT"
        reason = (
            f"max_games={max_games} reached; inconclusive Wilson interval "
            f"[{lb:.4f}, {ub:.4f}], fallback score {score:.4f}"
        )
    else:
        decision, reason = "EXTEND", f"inconclusive Wilson interval [{lb:.4f}, {ub:.4f}]"
    next_games = (
        next_even_target(current_games) if decision == "EXTEND" else current_games
    )
    return {
        "decision": decision,
        "reason": reason,
        "score_a": score,
        "wilson_lb": lb,
        "wilson_ub": ub,
        "elo_diff": elo,
        "next_games": min(next_games, max_games) if decision == "EXTEND" else next_games,
    }


def merge_results_jsonl(paths: Iterable[str | Path]) -> tuple[int, int, int, int]:
    rows: dict[int, str] = {}
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            idx = int(row["game_idx"])
            winner = row.get("winner")
            rows[idx] = winner if winner in ("A", "B") else "draw"
    a = sum(w == "A" for w in rows.values())
    b = sum(w == "B" for w in rows.values())
    d = len(rows) - a - b
    return a, b, d, len(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--current-games", type=int, required=True)
    ap.add_argument("--max-games", type=int, default=450)
    args = ap.parse_args()
    a, b, d, n = merge_results_jsonl(args.results)
    result = decide(a, b, d, args.current_games, args.max_games)
    result.update({"a_wins": a, "b_wins": b, "draws": d, "completed_games": n})
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
