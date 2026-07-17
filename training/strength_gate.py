#!/usr/bin/env python3
"""Reproducible, warm-session strength gate for two Titanium weight files.

This is the promotion gate for a candidate net or a search change.  Unlike a
one-shot ``genmove`` loop, it starts one ``titanium session --engine ...``
process per side and keeps it for the entire game.  Each opening is played as
a colour-swapped pair, so a deterministic engine cannot win merely because it
always receives the first-player side of an opening.

Examples:
  python training/strength_gate.py --candidate training/runs/v16/net_weights_best.bin \
      --baseline training/runs/v16/net_weights_previous.bin --games 50 --time 1
  python training/strength_gate.py --candidate CANDIDATE --baseline CHAMPION \
      --openings training/data/opening_book/non_titanium_book_lines.txt

The JSONL log contains the complete experiment identity (engine/weight hashes,
opening and result) and is intentionally append-only.  A score is reported but
the tool never labels a small match a promotion; use its records in the larger
match report/SPRT gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

_TRAINING = Path(__file__).resolve().parent
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

from engine_session import EngineSession
from self_play_overnight import GameSessions, play_one_game
from titanium_training.paths import ENGINE_BIN, REPO_ROOT


# Fallback used only when the audited opening-DAG export is unavailable (for
# example, a minimal checkout used by a unit test).  A handful of deterministic
# openings is not enough evidence for a promotion match: repeat games would
# merely replay the same deterministic result.  The normal suite below loads
# 109 distinct 14-ply prefixes from non-Titanium games, so a 100-game gate has
# one unique opening per colour-swapped pair and a 200-game gate has only a
# small tail of repeats.
_FALLBACK_OPENINGS: tuple[tuple[str, ...], ...] = (
    ("e2", "e8", "e3", "e7", "e4", "e6", "a3h"),
    ("e2", "e8", "e3", "e7", "e4", "e6", "d3h", "c6h", "e6v"),
    ("e2", "e8", "e3", "e7", "e4", "e6", "d3h", "c6h", "d5v"),
    ("e2", "e8", "e3", "e7", "e4", "d4v"),
    ("e2", "e8", "e3", "e7", "e4", "e6", "a3h", "d4v", "c5h"),
    ("e2", "e8", "e3", "e7", "e4", "e6", "a3h", "h6h", "c3h"),
)

_OPENING_DAG_EXPORT = _TRAINING / "data" / "opening_book" / "non_titanium_10ply.json"


def _load_default_openings() -> tuple[tuple[str, ...], ...]:
    try:
        document = json.loads(_OPENING_DAG_EXPORT.read_text(encoding="utf-8"))
        rows = document["nodesByPly"]["14"]
        openings = tuple(
            sorted(
                {
                    tuple(str(move) for move in row["prefix"])
                    for row in rows
                    if len(row.get("prefix", ())) == 14
                }
            )
        )
        # A small/corrupt DAG would defeat the no-repeat property above.
        if len(openings) >= 100:
            return openings
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        pass
    return _FALLBACK_OPENINGS


DEFAULT_OPENINGS = _load_default_openings()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_openings(path: Path | None) -> tuple[tuple[str, ...], ...]:
    if path is None:
        return DEFAULT_OPENINGS
    lines: list[tuple[str, ...]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        # Also accept the repository's human-readable book summaries:
        # ``ply 7 ... line=e2 e8 ...``.  Other key/value metadata is ignored.
        if "line=" in text:
            text = text.partition("line=")[2].strip()
        elif "=" in text:
            continue
        moves = tuple(text.split())
        if moves:
            lines.append(moves)
    if not lines:
        raise ValueError(f"no plain opening lines found in {path}")
    return tuple(lines)


def candidate_score(outcome_p0: int, candidate_is_p0: bool) -> float:
    """Return candidate score (1/0.5/0) from P0-relative game outcome."""
    if outcome_p0 == 0:
        return 0.5
    candidate_won = (outcome_p0 > 0) == candidate_is_p0
    return 1.0 if candidate_won else 0.0


def score_summary(scores: Iterable[float]) -> dict[str, float | int | None]:
    values = list(scores)
    games = len(values)
    if not games:
        return {"games": 0, "score": 0.0, "rate": 0.0, "elo": None, "ci95_pct": None}
    score = sum(values)
    rate = score / games
    # Treat a draw as half a result for an intentionally conservative, simple
    # progress interval.  This is visibility only; promotion uses a separate
    # sequential statistical decision, not this normal approximation.
    se = math.sqrt(max(rate * (1.0 - rate), 0.0) / games)
    elo = None if rate <= 0.0 or rate >= 1.0 else 400.0 * math.log10(rate / (1.0 - rate))
    return {
        "games": games,
        "score": score,
        "rate": rate,
        "elo": elo,
        "ci95_pct": 1.96 * se * 100.0,
    }


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def play_gate_game(
    *,
    game_id: str,
    opening: tuple[str, ...],
    candidate: Path,
    baseline: Path,
    engine: str,
    candidate_is_p0: bool,
    time_sec: float,
    engine_threads: int,
) -> dict:
    weights_p0, weights_p1 = (candidate, baseline) if candidate_is_p0 else (baseline, candidate)
    sessions = GameSessions(
        p0=EngineSession(engine, weights_p0, threads=engine_threads),
        p1=EngineSession(engine, weights_p1, threads=engine_threads),
    )
    try:
        result = play_one_game(
            game_id,
            time_sec,
            weights_p0,
            weights_p1,
            mixed=True,
            current_is_p0=candidate_is_p0,
            opening=list(opening),
            game_seed=None,
            engine_p0=engine,
            engine_p1=engine,
            matchup_kind="strength_gate",
            sessions=sessions,
        )
    finally:
        sessions.close()
    result["candidate_is_p0"] = candidate_is_p0
    result["candidate_score"] = (
        None
        if result.get("aborted")
        else candidate_score(int(result["outcome_p0"]), candidate_is_p0)
    )
    return result


def main() -> int:
    from prep_guard import guard_real_work

    guard_real_work("candidate_gating", detail="strength_gate/Elo")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidate", required=True, type=Path, help="candidate net weights")
    parser.add_argument("--baseline", required=True, type=Path, help="frozen champion/baseline weights")
    parser.add_argument("--engine", default="titanium-v17", help="exact warm-session engine for both sides")
    parser.add_argument("--games", type=int, default=50, help="even number of colour-swapped games")
    parser.add_argument("--time", type=float, default=1.0, help="seconds of search per move")
    parser.add_argument("--engine-threads", type=int, default=1, help="threads per engine side")
    parser.add_argument("--openings", type=Path, help="plain text: one legal opening per line")
    parser.add_argument("--output", type=Path, help="JSONL experiment log")
    args = parser.parse_args()

    if args.games <= 0 or args.games % 2:
        parser.error("--games must be a positive even number (paired colour swap)")
    if args.time <= 0:
        parser.error("--time must be positive")
    if args.engine_threads <= 0:
        parser.error("--engine-threads must be positive")
    for label, path in (("candidate", args.candidate), ("baseline", args.baseline), ("engine", ENGINE_BIN)):
        if not path.is_file():
            parser.error(f"{label} file does not exist: {path}")

    candidate = args.candidate.resolve()
    baseline = args.baseline.resolve()
    openings = read_openings(args.openings)
    output = args.output or (REPO_ROOT / "training" / "data" / "tournament" / f"strength_gate_{utc_stamp()}.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    identity = {
        "schema": "titanium-strength-gate-v1",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "engine": args.engine,
        "engine_sha256": sha256_file(ENGINE_BIN),
        "candidate": str(candidate),
        "candidate_sha256": sha256_file(candidate),
        "baseline": str(baseline),
        "baseline_sha256": sha256_file(baseline),
        "time_sec": args.time,
        "engine_threads": args.engine_threads,
        "opening_count": len(openings),
    }
    print(f"strength gate: {args.games} games, {args.time:g}s/move, engine={args.engine}")
    print(f"candidate={candidate.name}  baseline={baseline.name}  openings={len(openings)}")
    print(f"log: {output}")

    scores: list[float] = []
    with output.open("x", encoding="utf-8") as log:
        log.write(json.dumps({"kind": "experiment", **identity}, separators=(",", ":")) + "\n")
        for pair in range(args.games // 2):
            opening = openings[pair % len(openings)]
            for flip in range(2):
                candidate_is_p0 = flip == 0
                game = play_gate_game(
                    game_id=f"strength-gate-{utc_stamp()}-{pair:04d}-{flip}",
                    opening=opening,
                    candidate=candidate,
                    baseline=baseline,
                    engine=args.engine,
                    candidate_is_p0=candidate_is_p0,
                    time_sec=args.time,
                    engine_threads=args.engine_threads,
                )
                row = {"kind": "game", "pair": pair, "opening": list(opening), **game}
                log.write(json.dumps(row, separators=(",", ":")) + "\n")
                log.flush()
                if game["candidate_score"] is not None:
                    scores.append(float(game["candidate_score"]))
                summary = score_summary(scores)
                elo = summary["elo"]
                elo_text = "n/a" if elo is None else f"{elo:+.0f}"
                print(
                    f"[{len(scores)}/{args.games}] candidate score {summary['score']:.1f}/"
                    f"{summary['games']} = {summary['rate'] * 100:.1f}%  Elo {elo_text}",
                    flush=True,
                )
        final = {"kind": "summary", **score_summary(scores)}
        log.write(json.dumps(final, separators=(",", ":")) + "\n")
    return 0 if len(scores) == args.games else 2


if __name__ == "__main__":
    raise SystemExit(main())
