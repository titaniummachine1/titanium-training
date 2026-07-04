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

from streaming_checkpoint_chain import FROZEN_WEIGHTS, load_chain, sha256_file
from titanium_training.paths import ENGINE_BIN, REPO_ROOT
from titanium_training.validation.export_parity import verify_export_parity
from titanium_training.validation.opening_sanity import assert_opening_sanity

RUN_DIR = _TRAINING / "runs" / "v16"
LOG_DIR = _TRAINING / "data" / "overnight_logs"
STRENGTH_MEASURE_PATH = LOG_DIR / "strength_games.tsv"

# Real strength signal instead of a synthetic bench match: at all times,
# STREAM_PRIOR_EPOCH_FRACTION (~30%) of self-play games are current weights vs
# the immediately previous accepted weights, same engine, same node/time
# budget. We read that already-collected data (matchup_kind == "prior_epoch")
# since the last accepted epoch and use its real win rate as the gate -- no
# separate bench match.
#
# The minimum-sample floor must scale with actual throughput, not sit at a
# number picked once and forgotten -- 600 positions was calibrated when only
# the local pool was running; with Oracle added, combined throughput makes
# 600 positions a well-under-one-minute sample, useless as a gate. Default to
# ~90% of what one epoch naturally produces at the prior-epoch fraction
# (STREAM_TRIGGER_THRESHOLD * STREAM_PRIOR_EPOCH_FRACTION), so the gate uses
# nearly all of an epoch's real signal instead of an early noisy slice, and
# keeps tracking throughput automatically as either knob changes.
_trigger_threshold = int(os.environ.get("STREAM_TRIGGER_THRESHOLD", "16384"))
_prior_epoch_fraction = float(os.environ.get("STREAM_PRIOR_EPOCH_FRACTION", "0.30"))
PRIOR_EPOCH_MIN_POSITIONS = int(
    os.environ.get(
        "STREAM_PRIOR_EPOCH_MIN_POSITIONS",
        str(int(_trigger_threshold * _prior_epoch_fraction * 0.9)),
    )
)
PRIOR_EPOCH_MIN_SCORE = float(os.environ.get("STREAM_PRIOR_EPOCH_MIN_SCORE", "0.45"))


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
    """Aggregate real self-play games (current vs immediately-previous weights) since
    the last accepted epoch. This is the actual strength signal, not a bench match."""
    since = _last_accepted_at()
    if not STRENGTH_MEASURE_PATH.is_file():
        return {"games": 0, "positions": 0, "since": since, "reason": "no strength_games.tsv yet"}

    wins = 0
    draws = 0
    losses = 0
    positions = 0
    with STRENGTH_MEASURE_PATH.open("r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        try:
            idx_recorded = header.index("recorded_at")
            idx_kind = header.index("matchup_kind")
            idx_won = header.index("current_won")
            idx_plies = header.index("plies")
        except ValueError:
            return {"games": 0, "positions": 0, "since": since, "reason": "strength_games.tsv missing columns"}
        for line in f:
            cells = line.rstrip("\n").split("\t")
            if len(cells) <= max(idx_recorded, idx_kind, idx_won, idx_plies):
                continue
            if cells[idx_kind] != "prior_epoch":
                continue
            if since and cells[idx_recorded] <= since:
                continue
            won = cells[idx_won]
            if won == "1":
                wins += 1
            elif won == "0":
                losses += 1
            else:
                draws += 1
            try:
                positions += int(cells[idx_plies])
            except ValueError:
                pass

    games = wins + draws + losses
    score = (wins + 0.5 * draws) / games if games else None
    return {
        "games": games,
        "positions": positions,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": round(score, 4) if score is not None else None,
        "since": since,
        "min_positions": PRIOR_EPOCH_MIN_POSITIONS,
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

    # Strength vs prior is measured from REAL self-play games, not a synthetic bench
    # match. STREAM_PRIOR_EPOCH_FRACTION (~30%) of pool games already pit current pool
    # weights vs the immediately previous accepted weights, same engine, same node/time
    # budget. We aggregate those games since the last accept and gate on their win rate.
    strength = _prior_epoch_selfplay_strength()
    games = strength.get("games", 0)
    if games < PRIOR_EPOCH_MIN_GAMES:
        report["match_vs_previous"] = {
            **strength,
            "skipped": True,
            "blocking": False,
            "reason": (
                f"only {games}/{PRIOR_EPOCH_MIN_GAMES} real prior-epoch self-play games "
                "collected since last accept; not enough signal yet, not blocking"
            ),
        }
    else:
        score = strength["score"]
        passed = score >= PRIOR_EPOCH_MIN_SCORE
        report["match_vs_previous"] = {
            **strength,
            "skipped": False,
            "blocking": True,
            "passed": passed,
            "reason": (
                f"real self-play score {score:.3f} over {games} prior-epoch games "
                f"(need >= {PRIOR_EPOCH_MIN_SCORE})"
            ),
        }
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

    strength_ok = report["match_vs_previous"].get("passed", True)  # True when not enough data (non-blocking)
    report["passed"] = (
        report["export_parity"]["passed"]
        and report["opening_sanity"]["passed"]
        and strength_ok
    )
    if not strength_ok:
        report["reject_reason"] = "prior_epoch_selfplay_strength_gate"
    return report
