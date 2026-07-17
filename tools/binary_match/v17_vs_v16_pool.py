#!/usr/bin/env python3
"""Run v17 vs v16 through the real training data-generation pipeline.

Uses ContinuousPool (local_game_pool) with a patched matchup selector so every
game is titanium-v17 vs titanium-v16, with the same exploration / temperature /
opening-exploration settings as normal training self-play.  Games are persisted
to games.db and labels.db exactly like generation data, and a strength log is
written alongside.  The pool stops automatically after --games successful games.

Usage:
  cd "c:\gitProjects\Quoridor best AI"
  set TITANIUM_ENGINE_BIN=engine\target\release\titanium.exe
  set RUSTFLAGS=-C target-cpu=native
  python tools/binary_match/v17_vs_v16_pool.py --threads 17 --games 200 --time 1.0

This does NOT touch the Oracle factory; it uses only local workers.  If you want
the 13 Oracle workers too, deploy a matching script on the Oracle host and
import the results via oracle_importer.py, or just run this locally with enough
threads to finish 200 games quickly.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
_TRAINING = _REPO / "training"
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

import local_game_pool
import continuous_pool
import generation_matchup

MATCHUP_V17_VS_V16 = "v17_vs_v16"


@dataclass(frozen=True)
class RunConfig:
    threads: int
    games: int
    time_sec: float
    nodes: int
    opening_exploration: bool
    explore_chance: float
    out_dir: Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_v17_vs_v16_matchup(
    rng: random.Random,
    *,
    current_weights: Path,
    previous_weights: Path | None,
) -> generation_matchup.GenerationMatchup:
    """Always return titanium-v17 vs titanium-v16, same weights, random seats."""
    v17_is_p0 = rng.random() < 0.5
    return generation_matchup.GenerationMatchup(
        kind=MATCHUP_V17_VS_V16,
        engine_p0="titanium-v17" if v17_is_p0 else "titanium-v16",
        engine_p1="titanium-v16" if v17_is_p0 else "titanium-v17",
        weights_p0=current_weights,
        weights_p1=current_weights,
        current_is_p0=v17_is_p0,
        opponent_engine=None,
        opening_exploration=True,
        metadata={"matchup": "v17_vs_v16"},
    )


class StoppingPool(continuous_pool.ContinuousPool):
    """ContinuousPool that stops after a target number of successfully persisted games."""

    def __init__(self, cfg: continuous_pool.PoolConfig, target_games: int, out_dir: Path):
        super().__init__(cfg)
        self.target_games = target_games
        self.out_dir = out_dir
        self._persisted_games = 0
        self._persist_lock = threading.Lock()
        self._stop_file = out_dir / "v17_vs_v16.stop"

    def _persist_game(self, r: dict) -> continuous_pool.PersistOutcome:
        outcome = super()._persist_game(r)
        if outcome.counted:
            with self._persist_lock:
                self._persisted_games += 1
                if self._persisted_games >= self.target_games:
                    self._stop.set()
        return outcome

    def _append_strength_measure(self, r: dict) -> None:
        # Log every v17_vs_v16 game as a strength measurement, not just prior-epoch games.
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / "strength.tsv"
        header = (
            "recorded_at\tgame_id\tmatchup_kind\tengine_p0\tengine_p1\t"
            "outcome_p0\tv17_is_p0\tv17_won\tplies\n"
        )
        v17_is_p0 = r.get("engine_p0") == "titanium-v17"
        outcome_p0 = r.get("outcome_p0", 0)
        v17_won = (outcome_p0 > 0) if v17_is_p0 else (outcome_p0 < 0)
        row = [
            utc_now(),
            r.get("game_id"),
            r.get("matchup_kind"),
            r.get("engine_p0"),
            r.get("engine_p1"),
            outcome_p0,
            "1" if v17_is_p0 else "0",
            "1" if v17_won else "0",
            r.get("plies"),
        ]
        needs_header = not path.exists()
        with path.open("a", encoding="utf-8", newline="") as f:
            if needs_header:
                f.write(header)
            f.write("\t".join(str(c) for c in row) + "\n")

    def _record_game(self, r: dict) -> int:
        n = super()._record_game(r)
        # Also update our own concise summary file.
        self._write_summary()
        return n

    def _write_summary(self) -> None:
        v17_wins = self._state.mixed_wins
        v16_wins = self._state.mixed_losses
        draws = self._state.mixed_draws
        n = v17_wins + v16_wins + draws
        score = (v17_wins + 0.5 * draws) / n if n else 0.0
        summary = {
            "running": not self._stop.is_set(),
            "target_games": self.target_games,
            "completed_games": n,
            "v17_wins": v17_wins,
            "v16_wins": v16_wins,
            "draws": draws,
            "score_v17": round(score, 4),
            "updated_at": utc_now(),
        }
        path = self.out_dir / "summary.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        tmp.replace(path)


def parse_args() -> RunConfig:
    ap = argparse.ArgumentParser(description="v17 vs v16 through the training pipeline")
    ap.add_argument("--threads", type=int, default=17)
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--time", type=float, default=1.0)
    ap.add_argument("--nodes", type=int, default=550000)
    ap.add_argument("--opening-exploration", action="store_true", default=True)
    ap.add_argument("--explore-chance", type=float, default=0.35)
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "runs" / "v17_vs_v16_pipeline")
    args = ap.parse_args()
    return RunConfig(
        threads=args.threads,
        games=args.games,
        time_sec=args.time,
        nodes=args.nodes,
        opening_exploration=args.opening_exploration,
        explore_chance=args.explore_chance,
        out_dir=args.out_dir,
    )


def build_argv(cfg: RunConfig) -> list[str]:
    return [
        "--threads", str(cfg.threads),
        "--time", str(cfg.time_sec),
        "--nodes", str(cfg.nodes),
        "--batch-games", str(max(cfg.games, 32)),
        "--no-initial-epoch",
        "--no-oracle",
        "--opening-exploration",
        "--explore-chance", str(cfg.explore_chance),
        "--explore-start-ply", "6",
        "--explore-max-loss-cp", "140",
        "--explore-candidate-count", "18",
        "--explore-top-n", "8",
        "--explore-temperature-cp", "45.0",
        "--explore-wall-bonus-cp", "12",
        "--explore-decay-after-hit", "0.55",
        "--explore-min-chance", "0.03",
        "--opening-temperature-initial", "1.0",
        "--opening-temperature-after-ply4", "1.0",
        "--opening-temperature-decay-per-ply", "0.08",
        "--opening-temperature-min-while-known", "0.45",
        "--opening-exploration-max-ply", "16",
        "--novel-prefix-temperature", "0.0",
        "--opening-high-freq-boost", "0.1",
        "--opening-prob-floor", "0.08",
    ]


def main() -> int:
    cfg = parse_args()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    # Patch the generation matchup to always return v17 vs v16.
    generation_matchup.choose_generation_matchup = _make_v17_vs_v16_matchup

    # Patch mixed-vs-prior flagging so v17_vs_v16 games are treated as "mixed"
    # and logged in the strength measure path.
    original_mixed = continuous_pool.ContinuousPool._record_game

    def _record_game_patched(self, r: dict) -> int:
        if r.get("matchup_kind") == MATCHUP_V17_VS_V16:
            r["mixed"] = True
            r["current_is_p0"] = r.get("engine_p0") == "titanium-v17"
            r["current_won"] = (r.get("outcome_p0", 0) > 0) if r["current_is_p0"] else (r.get("outcome_p0", 0) < 0)
            r["opponent_engine"] = "titanium-v16"
        return original_mixed(self, r)

    continuous_pool.ContinuousPool._record_game = _record_game_patched

    argv = build_argv(cfg)
    args = continuous_pool.parse_pool_args(argv)
    pool_cfg = continuous_pool.build_pool_config(args, no_oracle=True)
    pool = StoppingPool(pool_cfg, target_games=cfg.games, out_dir=cfg.out_dir)

    from pool_lock import PoolInstanceLock, release_pool_lock

    with PoolInstanceLock(lock_path=local_game_pool.LOCAL_GAME_POOL_LOCK_PATH) as lock_info:
        local_game_pool.PID_PATH.write_text(str(lock_info.pid), encoding="ascii")
        print(f"v17_vs_v16 pipeline started pid={lock_info.pid} threads={cfg.threads} target={cfg.games}")
        try:
            pool.run()
        finally:
            release_pool_lock(local_game_pool.LOCAL_GAME_POOL_LOCK_PATH)

    print(f"\nv17_vs_v16 pipeline finished. Summary: {cfg.out_dir / 'summary.json'}")
    print(f"Strength log: {cfg.out_dir / 'strength.tsv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
