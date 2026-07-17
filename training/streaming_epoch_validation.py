"""Post-epoch validation for streaming NNUE training (no auto-deploy)."""
from __future__ import annotations

import hashlib
import json
import math
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
from diversity.promotion_record import build_promotion_record
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
PRIOR_EPOCH_MIN_GAMES = int(os.environ.get("STREAM_PRIOR_EPOCH_MIN_GAMES", "200"))
PRIOR_EPOCH_MIN_SCORE = float(os.environ.get("STREAM_PRIOR_EPOCH_MIN_SCORE", "0.50"))

# Second, independent gate: candidate vs its GRANDPARENT (parent's own
# parent, i.e. two accepted epochs back) -- catches a candidate that beats
# its immediate parent by drifting to exploit that one specific net's blind
# spots rather than actually getting stronger (classic self-play failure
# mode: each generation "cheats" its predecessor without real absolute
# progress). Real 2-generation progress should clearly beat a coin flip, so
# this bar is higher than the parent gate despite fewer games. Only run if
# the parent gate already passed -- no point spending real games proving
# non-drift on a candidate that's rejected either way.
GRANDPARENT_MIN_GAMES = int(os.environ.get("STREAM_GRANDPARENT_MIN_GAMES", "100"))
GRANDPARENT_MIN_SCORE = float(os.environ.get("STREAM_GRANDPARENT_MIN_SCORE", "0.50"))

# A raw score over 50% is not evidence of an improvement.  Matches are played
# as colour-swapped opening pairs, so evaluate the direction of the result at
# the *pair* level and use a one-sided exact sign test.  This is deliberately
# conservative: a tiny gain needs more pairs, not a lucky 200-game sample.
PROMOTION_SIGN_TEST_ALPHA = float(os.environ.get("STREAM_PROMOTION_SIGN_TEST_ALPHA", "0.05"))
PROMOTION_MIN_DECISIVE_PAIRS = int(
    os.environ.get("STREAM_PROMOTION_MIN_DECISIVE_PAIRS", "20")
)


def _one_sided_sign_test_p_value(wins: int, decisive_pairs: int) -> float:
    """Exact P[X >= wins] for X~Binomial(decisive_pairs, 0.5).

    ``math.comb`` plus a direct division can overflow when a user requests a
    large confirmation match.  Log-sum-exp keeps the value stable while
    retaining an exact binomial model at the pair level.
    """
    if decisive_pairs <= 0 or wins <= 0:
        return 1.0
    if wins > decisive_pairs:
        raise ValueError("wins cannot exceed decisive_pairs")
    log_two = math.log(2.0)
    logs = [
        math.lgamma(decisive_pairs + 1)
        - math.lgamma(k + 1)
        - math.lgamma(decisive_pairs - k + 1)
        - decisive_pairs * log_two
        for k in range(wins, decisive_pairs + 1)
    ]
    peak = max(logs)
    return math.exp(peak) * sum(math.exp(term - peak) for term in logs)


def paired_promotion_evidence(
    pair_scores: list[float],
    *,
    alpha: float = PROMOTION_SIGN_TEST_ALPHA,
    min_decisive_pairs: int = PROMOTION_MIN_DECISIVE_PAIRS,
) -> dict[str, Any]:
    """Return fail-closed evidence for a colour-swapped match.

    A pair scores 1 when the candidate wins both colours, 0 when it loses
    both, and 0.5 when the two games split/draw.  Split pairs carry no sign
    evidence; treating the individual games as independent would overstate
    confidence because they share an opening and deterministic search setup.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("promotion alpha must be in (0, 1)")
    if min_decisive_pairs < 1:
        raise ValueError("minimum decisive pairs must be positive")
    wins = sum(score > 0.5 for score in pair_scores)
    losses = sum(score < 0.5 for score in pair_scores)
    ties = len(pair_scores) - wins - losses
    decisive = wins + losses
    p_value = _one_sided_sign_test_p_value(wins, decisive)
    mean_score = sum(pair_scores) / len(pair_scores) if pair_scores else None
    passed = (
        mean_score is not None
        and mean_score > 0.5
        and decisive >= min_decisive_pairs
        and p_value <= alpha
    )
    return {
        "pair_count": len(pair_scores),
        "pair_wins": wins,
        "pair_losses": losses,
        "pair_ties": ties,
        "decisive_pairs": decisive,
        "pair_score": round(mean_score, 4) if mean_score is not None else None,
        "sign_test_p_value": p_value,
        "sign_test_alpha": alpha,
        "min_decisive_pairs": min_decisive_pairs,
        "passed": passed,
    }


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
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from engine_session import EngineSession
    from self_play_overnight import check_winner
    from strength_gate import DEFAULT_OPENINGS

    if games <= 0 or games % 2:
        raise ValueError("candidate-vs-parent validation requires a positive, even game count")

    def play_one(pair_idx: int, flip: int) -> tuple[int, int, float | None, str | None]:
        # Both colours play the exact same opening.  This is essential for
        # deterministic engines: alternating colours alone still lets opening
        # distribution noise masquerade as a net improvement.
        cand_is_p0 = flip == 0
        opening = list(DEFAULT_OPENINGS[pair_idx % len(DEFAULT_OPENINGS)])
        if len(opening) >= max_ply:
            # A match that never asks either engine to move is not evidence.
            # Fail closed instead of scoring the supplied opening as a draw.
            return pair_idx, flip, None, "max_ply_not_beyond_opening"
        sess_p0 = EngineSession("titanium-v17", candidate_bin if cand_is_p0 else parent_bin)
        sess_p1 = EngineSession("titanium-v17", parent_bin if cand_is_p0 else candidate_bin)
        try:
            moves = opening
            for ply in range(len(moves), max_ply):
                active = sess_p0 if ply % 2 == 0 else sess_p1
                if not active.sync(moves) or not active.alive():
                    return pair_idx, flip, None, "session_sync_failed"
                mv = active.go(time_sec)
                if not mv:
                    return pair_idx, flip, None, "session_go_failed"
                moves.append(mv)
                # Never ask an engine to search after the game has ended.
                # Besides wasting the whole remaining match budget, a terminal
                # position often returns ``(none)`` and used to be recorded as
                # an infrastructure abort rather than its real result.
                winner = check_winner(moves)
                if winner is not None:
                    cand_won = (winner == 0) == cand_is_p0
                    return pair_idx, flip, (1.0 if cand_won else 0.0), None
        finally:
            sess_p0.close()
            sess_p1.close()
        # A match that reaches its declared ply cap without a winner is an
        # agreed draw.  It remains a completed pair observation, unlike any
        # session/sync failure above.
        return pair_idx, flip, 0.5, None

    outcomes: list[float] = []
    pair_outcomes: dict[int, dict[int, float]] = {}
    errors: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(play_one, pair_idx, flip)
            for pair_idx in range(games // 2)
            for flip in range(2)
        ]
        for f in as_completed(futures):
            pair_idx, flip, score, error = f.result()
            if score is None:
                errors[error or "unknown"] = errors.get(error or "unknown", 0) + 1
            else:
                outcomes.append(score)
                pair_outcomes.setdefault(pair_idx, {})[flip] = score

    n = len(outcomes)
    wins = sum(1 for s in outcomes if s == 1.0)
    losses = sum(1 for s in outcomes if s == 0.0)
    draws = n - wins - losses
    score = sum(outcomes) / n if n else None
    # Never promote using an incomplete pair.  The caller already rejects an
    # incomplete *match*, but retaining this explicit count makes the
    # statistical assumption auditable in the epoch report.
    pair_scores = [
        (outcomes_by_colour[0] + outcomes_by_colour[1]) / 2.0
        for _pair, outcomes_by_colour in sorted(pair_outcomes.items())
        if 0 in outcomes_by_colour and 1 in outcomes_by_colour
    ]
    evidence = paired_promotion_evidence(pair_scores)
    opening_fingerprint = hashlib.sha256(
        json.dumps(DEFAULT_OPENINGS, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "games": n,
        "requested_games": games,
        "aborted_games": games - n,
        "aborted_reasons": errors,
        "paired_openings": True,
        "opening_count": len(DEFAULT_OPENINGS),
        "opening_suite_sha256": opening_fingerprint,
        "completed_pairs": len(pair_scores),
        "promotion_evidence": evidence,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": round(score, 4) if score is not None else None,
        "candidate_sha256": sha256_file(candidate_bin),
        "parent_sha256": sha256_file(parent_bin),
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
    synthetic_only: bool = False,
) -> dict[str, Any]:
    if not synthetic_only:
        from prep_guard import guard_real_work

        guard_real_work("candidate_gating", detail="run_epoch_validation")
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
        score_text = "n/a" if score is None else f"{score:.3f}"
        evidence = match["promotion_evidence"]
        passed = (
            score is not None
            and match["games"] == PRIOR_EPOCH_MIN_GAMES
            and match["aborted_games"] == 0
            and score >= PRIOR_EPOCH_MIN_SCORE
            and match["completed_pairs"] == PRIOR_EPOCH_MIN_GAMES // 2
            and evidence["passed"]
        )
        report["match_vs_previous"] = {
            **match,
            "min_games": PRIOR_EPOCH_MIN_GAMES,
            "min_score": PRIOR_EPOCH_MIN_SCORE,
            "skipped": False,
            "blocking": True,
            "passed": passed,
            "reason": (
                f"direct paired match score {score_text} over {match['games']} real games "
                f"vs actual parent weights (need {PRIOR_EPOCH_MIN_GAMES} completed, "
                f"0 aborted, score >= {PRIOR_EPOCH_MIN_SCORE}, and paired sign-test "
                f"p <= {evidence['sign_test_alpha']} with at least "
                f"{evidence['min_decisive_pairs']} decisive pairs; observed "
                f"p={evidence['sign_test_p_value']:.6g} over "
                f"{evidence['decisive_pairs']} decisive pairs)"
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
        gp_score_text = "n/a" if gp_score is None else f"{gp_score:.3f}"
        gp_evidence = gp_match["promotion_evidence"]
        gp_passed = (
            gp_score is not None
            and gp_match["games"] == GRANDPARENT_MIN_GAMES
            and gp_match["aborted_games"] == 0
            and gp_score >= GRANDPARENT_MIN_SCORE
            and gp_match["completed_pairs"] == GRANDPARENT_MIN_GAMES // 2
            and gp_evidence["passed"]
        )
        report["match_vs_grandparent"] = {
            **gp_match,
            "min_games": GRANDPARENT_MIN_GAMES,
            "min_score": GRANDPARENT_MIN_SCORE,
            "grandparent_epoch": grandparent.get("epoch"),
            "skipped": False,
            "blocking": True,
            "passed": gp_passed,
            "reason": (
                f"direct paired match score {gp_score_text} over {gp_match['games']} real games "
                f"vs epoch {grandparent.get('epoch')} (grandparent, need {GRANDPARENT_MIN_GAMES} completed, "
                f"0 aborted, score >= {GRANDPARENT_MIN_SCORE}, and paired sign-test "
                f"p <= {gp_evidence['sign_test_alpha']} with at least "
                f"{gp_evidence['min_decisive_pairs']} decisive pairs; observed "
                f"p={gp_evidence['sign_test_p_value']:.6g} over "
                f"{gp_evidence['decisive_pairs']} decisive pairs) -- "
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
