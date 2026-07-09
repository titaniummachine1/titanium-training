"""Post-epoch validation for streaming NNUE training (no auto-deploy)."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

_TRAINING = Path(__file__).resolve().parent
_REPO = _TRAINING.parent
sys.path.insert(0, str(_TRAINING))

from streaming_checkpoint_chain import (
    FROZEN_WEIGHTS,
    load_chain,
    previous_accepted,
    resolve_accepted_weights,
    sha256_file,
)
from titanium_training.paths import ENGINE_BIN, REPO_ROOT
from titanium_training.validation.checkpoint_metadata import CheckpointArchitectureError
from titanium_training.validation.export_parity import verify_export_parity
from titanium_training.validation.opening_sanity import assert_opening_sanity


def is_validation_infrastructure_error(exc: BaseException) -> bool:
    """Distinguish validator bugs from candidate quality failures."""
    if isinstance(exc, CheckpointArchitectureError):
        return True
    msg = str(exc).lower()
    needles = (
        "size mismatch",
        "cannot infer checkpoint architecture",
        "refusing to default to h48",
        "missing model state",
    )
    return any(n in msg for n in needles)

RUN_DIR = _TRAINING / "runs" / "v16"
LOG_DIR = _TRAINING / "data" / "overnight_logs"

# Direct candidate-vs-parent match (see _match_candidate_vs_parent): the
# candidate's real just-trained weights play PRIOR_EPOCH_MIN_GAMES real games
# against the real immediately-previous-accepted weights, right now, as part
# of validation -- not an aggregate of self-play games logged during
# training (that approach measured whichever weights the pool's "current"
# slot happened to hold during the training window, which is always the
# OUTGOING already-accepted epoch, never the new candidate -- confirmed to
# have silently accepted epoch 9 on 254 games that were entirely
# epoch_8-vs-epoch_7 by weight hash).
PRIOR_EPOCH_MIN_GAMES = int(os.environ.get("STREAM_PRIOR_EPOCH_MIN_GAMES", "100"))
PRIOR_EPOCH_MIN_SCORE = float(os.environ.get("STREAM_PRIOR_EPOCH_MIN_SCORE", "0.45"))

# Second, independent gate: candidate vs its GRANDPARENT (parent's own
# parent, i.e. two accepted epochs back) -- catches a candidate that beats
# its immediate parent by drifting to exploit that one specific net's blind
# spots rather than actually getting stronger (classic self-play failure
# mode: each generation "cheats" its predecessor without real absolute
# progress). Real 2-generation progress should clearly beat a coin flip, so
# this bar is higher than the parent gate despite fewer games. Only run if
# the parent gate already passed -- no point spending real games proving
# non-drift on a candidate that's rejected either way.
GRANDPARENT_MIN_GAMES = int(os.environ.get("STREAM_GRANDPARENT_MIN_GAMES", "30"))
GRANDPARENT_MIN_SCORE = float(os.environ.get("STREAM_GRANDPARENT_MIN_SCORE", "0.50"))


def _last_accepted_at() -> str | None:
    try:
        chain = load_chain()
    except Exception:
        return None
    epochs = chain.get("epochs") or []
    if not epochs:
        return None
    return epochs[-1].get("accepted_at")


def _prior_epoch_selfplay_strength() -> dict[str, Any]:
    """Cheap status read of the last-computed prior-epoch strength gate result.

    Reads `latest_epoch_report.json` (written by `write_epoch_report` after
    every `run_epoch_validation` call) instead of re-running the 100-game
    direct match -- this exists for one-shot status tools (training_status.py)
    that need to display "did the gate pass" without spending minutes
    replaying games. For the live accept/reject decision, the coordinator
    calls `run_epoch_validation` -> `_match_candidate_vs_parent` directly.
    """
    report_path = LOG_DIR / "latest_epoch_report.json"
    if not report_path.is_file():
        return {
            "games": 0,
            "score": None,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "since": None,
            "passed": None,
            "skipped": True,
            "epoch": None,
        }
    try:
        doc = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "games": 0,
            "score": None,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "since": None,
            "passed": None,
            "skipped": True,
            "epoch": None,
        }
    match = (doc.get("validation") or {}).get("match_vs_previous") or {}
    return {
        "games": match.get("games", 0),
        "score": match.get("score"),
        "wins": match.get("wins", 0),
        "draws": match.get("draws", 0),
        "losses": match.get("losses", 0),
        "since": doc.get("recorded_at"),
        "passed": match.get("passed"),
        "skipped": match.get("skipped", False),
        "epoch": doc.get("epoch"),
    }


def _match_candidate_vs_parent(
    *,
    candidate_bin: Path,
    parent_bin: Path,
    games: int,
    time_sec: float = 1.0,
    max_ply: int = 128,
    concurrency: int = 4,
) -> dict[str, Any]:
    """Real, immediate match: the candidate's ACTUAL just-trained weights vs the
    ACTUAL parent it was trained from -- sides alternate, warm engine sessions
    (same mechanism as the pool/tournament), no lag.

    This replaced a bug (confirmed live, 2026-07-05): the self-play-log-based
    "match_vs_previous" read matchup_kind=="prior_epoch" rows since the last
    accept, but the self-play pool's "current" weights slot only refreshes
    AFTER a new epoch is accepted -- so every one of those games measured
    whatever was ALREADY accepted (the outgoing epoch) vs the one before
    that, never the new candidate about to be gated. Epoch 9 was accepted
    this way on 254 games that were entirely epoch_8-vs-epoch_7 by weight
    hash, not epoch_9-vs-epoch_8 -- and then lost a real, direct 300-game
    tournament against epoch_8 (33% vs 83%). This function plays the real
    comparison directly instead of trusting stale aggregate logs.
    """
    import random
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from engine_session import EngineSession
    from self_play_overnight import check_winner

    def play_one(game_idx: int) -> float:
        cand_is_p0 = game_idx % 2 == 0
        sess_p0 = EngineSession("titanium-v16", candidate_bin if cand_is_p0 else parent_bin)
        sess_p1 = EngineSession("titanium-v16", parent_bin if cand_is_p0 else candidate_bin)
        try:
            moves: list[str] = []
            for ply in range(max_ply):
                active = sess_p0 if ply % 2 == 0 else sess_p1
                if not active.sync(moves) or not active.alive():
                    break
                mv = active.go(time_sec)
                if not mv:
                    break
                moves.append(mv)
            winner = check_winner(moves)
        finally:
            sess_p0.close()
            sess_p1.close()
        if winner is None:
            return 0.5
        cand_won = (winner == 0) == cand_is_p0
        return 1.0 if cand_won else 0.0

    outcomes: list[float] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(play_one, i) for i in range(games)]
        for f in as_completed(futures):
            outcomes.append(f.result())

    n = len(outcomes)
    wins = sum(1 for s in outcomes if s == 1.0)
    losses = sum(1 for s in outcomes if s == 0.0)
    draws = n - wins - losses
    score = sum(outcomes) / n if n else None
    return {
        "games": n,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": round(score, 4) if score is not None else None,
        "candidate_sha256": sha256_file(candidate_bin),
        "parent_sha256": sha256_file(parent_bin),
        "min_games": PRIOR_EPOCH_MIN_GAMES,
        "min_score": PRIOR_EPOCH_MIN_SCORE,
    }


def _run_match(
    *,
    games: int,
    time_sec: float,
    engine_a: str,
    engine_b: str,
    weights_a: Path | None,
    weights_b: Path | None,
) -> dict[str, Any]:
    env = os.environ.copy()
    env.pop("TITANIUM_NET_WEIGHTS_PATH", None)
    if weights_a:
        env["TITANIUM_NET_WEIGHTS_PATH"] = str(weights_a.resolve())
    cmd = [
        str(ENGINE_BIN),
        "match",
        "--games",
        str(games),
        "--time",
        str(time_sec),
        "--openings",
        "book",
        "--a",
        engine_a,
        "--b",
        engine_b,
        "--no-early-stop",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT), env=env, timeout=3600)
    lines = (proc.stdout + proc.stderr).splitlines()
    summary = [ln for ln in lines if "wins" in ln.lower() or "STRENGTH" in ln or "score" in ln.lower()]
    return {
        "exit_code": proc.returncode,
        "weights_a_sha256": sha256_file(weights_a) if weights_a else None,
        "weights_b_sha256": sha256_file(weights_b) if weights_b else None,
        "summary": summary[-8:],
    }


def _search_bench(weights: Path | None) -> dict[str, Any]:
    bench = _REPO / "engine" / "target" / "release" / "search_bench.exe"
    if not bench.is_file():
        return {"skipped": True, "reason": "search_bench missing"}
    env = os.environ.copy()
    if weights:
        env["TITANIUM_NET_WEIGHTS_PATH"] = str(weights.resolve())
    else:
        env.pop("TITANIUM_NET_WEIGHTS_PATH", None)
    proc = subprocess.run(
        [str(bench), "time", "--sec", "2", "--runs", "3"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=env,
        timeout=120,
    )
    tail = proc.stdout.strip().splitlines()
    parsed = {}
    if tail:
        try:
            parsed = json.loads(tail[-1])
        except json.JSONDecodeError:
            parsed = {"raw": tail[-1]}
    return {
        "exit_code": proc.returncode,
        "threads": parsed.get("threads"),
        "engine_mode": parsed.get("engine_mode"),
        "median_nps": parsed.get("median_nps"),
        "median_depth": parsed.get("median_depth"),
        "move": parsed.get("move"),
    }


def run_epoch_validation(
    *,
    checkpoint: Path,
    candidate_bin: Path,
    previous_bin: Path | None,
    frozen_bin: Path = FROZEN_WEIGHTS,
    short_games: int = 20,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "candidate_sha256": sha256_file(candidate_bin),
        "checkpoint": str(checkpoint),
    }

    parity = verify_export_parity(checkpoint, candidate_bin)
    report["export_parity"] = {
        "passed": parity.passed,
        "max_cp": parity.max_parity_error,
    }

    try:
        assert_opening_sanity(candidate_bin)
        report["opening_sanity"] = {"passed": True}
    except Exception as exc:
        report["opening_sanity"] = {"passed": False, "error": str(exc)}

    report["parity_check"] = {
        "skipped": True,
        "blocking": False,
        "reason": "streaming NNUE training intentionally does not gate on Python/engine eval parity",
    }

    report["search_bench"] = _search_bench(candidate_bin)

    # Direct, immediate candidate-vs-parent match -- NOT the self-play-log
    # aggregate this used to be (see _match_candidate_vs_parent docstring for
    # the exact bug that made that measure the wrong pair of weights). If
    # there's no parent yet (bootstrap/epoch 1), there's nothing to gate
    # against -- not blocking.
    if previous_bin is None or not Path(previous_bin).is_file():
        report["match_vs_previous"] = {
            "skipped": True,
            "blocking": False,
            "reason": "no previous accepted weights to compare against (bootstrap epoch)",
        }
    else:
        match = _match_candidate_vs_parent(
            candidate_bin=candidate_bin,
            parent_bin=Path(previous_bin),
            games=PRIOR_EPOCH_MIN_GAMES,
        )
        score = match["score"]
        passed = score is not None and score >= PRIOR_EPOCH_MIN_SCORE
        report["match_vs_previous"] = {
            **match,
            "skipped": False,
            "blocking": True,
            "passed": passed,
            "reason": (
                f"direct match score {score:.3f} over {match['games']} real games "
                f"vs actual parent weights (need >= {PRIOR_EPOCH_MIN_SCORE})"
            ),
        }
    parent_ok = report["match_vs_previous"].get("passed", True)  # True when not enough data (non-blocking)

    grandparent = previous_accepted() if parent_ok else None
    if not parent_ok:
        report["match_vs_grandparent"] = {
            "skipped": True,
            "blocking": False,
            "reason": "parent gate already failed -- no point spending real games on drift check too",
        }
    elif grandparent is None:
        report["match_vs_grandparent"] = {
            "skipped": True,
            "blocking": False,
            "reason": "fewer than 2 accepted epochs in chain -- no grandparent to compare against yet",
        }
    else:
        grandparent_bin = resolve_accepted_weights(grandparent)
        gp_match = _match_candidate_vs_parent(
            candidate_bin=candidate_bin,
            parent_bin=grandparent_bin,
            games=GRANDPARENT_MIN_GAMES,
        )
        gp_score = gp_match["score"]
        gp_passed = gp_score is not None and gp_score >= GRANDPARENT_MIN_SCORE
        report["match_vs_grandparent"] = {
            **gp_match,
            "grandparent_epoch": grandparent.get("epoch"),
            "skipped": False,
            "blocking": True,
            "passed": gp_passed,
            "reason": (
                f"direct match score {gp_score:.3f} over {gp_match['games']} real games "
                f"vs epoch {grandparent.get('epoch')} (grandparent, need >= {GRANDPARENT_MIN_SCORE}) -- "
                "catches drift/exploit of the immediate parent that isn't real absolute progress"
            ),
        }
    grandparent_ok = report["match_vs_grandparent"].get("passed", True)  # True when skipped (non-blocking)

    report["match_vs_frozen"] = {
        "skipped": True,
        "blocking": False,
        "reason": "unfinished streaming weights are not compared against older weights",
    }
    report["control_vs_control"] = {
        "skipped": True,
        "blocking": False,
        "reason": "unfinished streaming weights are not match-tested during streaming acceptance",
    }

    report["passed"] = (
        report["export_parity"]["passed"]
        and report["opening_sanity"]["passed"]
        and parent_ok
        and grandparent_ok
    )
    if not parent_ok:
        report["reject_reason"] = "prior_epoch_selfplay_strength_gate"
    elif not grandparent_ok:
        report["reject_reason"] = "grandparent_drift_gate"
    return report
