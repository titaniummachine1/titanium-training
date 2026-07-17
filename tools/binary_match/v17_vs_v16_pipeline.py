#!/usr/bin/env python3
"""Self-contained v17 vs v16 match using the real training data pipeline.

Runs titanium-v17 vs titanium-v16 with the same exploration, temperature, and
opening-exploration as normal generation self-play.  Games are written directly
to games.db and labels.db via db_import, and a strength TSV is written to the
output directory.  The runner stops automatically after --games successful games.

Does NOT require pyarrow (avoids the teacher-parquet sync path).

Usage:
  cd "c:\gitProjects\Quoridor best AI"
  set TITANIUM_ENGINE_BIN=engine\target\release\titanium.exe
  set RUSTFLAGS=-C target-cpu=native
  python tools/binary_match/v17_vs_v16_pipeline.py --threads 17 --games 200 --time 1.0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import signal
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

from db_import import GAMES_DB_PATH, GAMES_SCHEMA, LABELS_DB_PATH, LABELS_SCHEMA, open_db, write_batch
from engine_session import EngineSession
from game_opening_gate import log_rejected_game, opening_sanity_ok
from generation_matchup import MATCHUP_PRIOR_EPOCH, GenerationMatchup, choose_generation_matchup
from opening_prefix_index import DEFAULT_INDEX_PATH, OpeningPrefixIndex
from self_play_overnight import (
    DEFAULT_CURRENT,
    ExplorationConfig,
    GameSessions,
    OpeningExplorationConfig,
    play_one_game,
)
from titanium_training.paths import ENGINE_BIN, REPO_ROOT

MATCHUP_V17_VS_V16 = "v17_vs_v16"
LOG_DIR = _TRAINING / "data" / "overnight_logs"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(msg: str) -> None:
    print(msg, flush=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with (LOG_DIR / "v17_vs_v16_pipeline.log").open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def _sha16(path: Path | None) -> str | None:
    if not path or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# Patch generation matchup to always play v17 vs v16.
_original_choose_generation_matchup = choose_generation_matchup


def _choose_v17_vs_v16(
    rng: random.Random,
    *,
    current_weights: Path,
    previous_weights: Path | None,
) -> GenerationMatchup:
    v17_is_p0 = rng.random() < 0.5
    return GenerationMatchup(
        kind=MATCHUP_V17_VS_V16,
        engine_p0="titanium-v17" if v17_is_p0 else "titanium-v16",
        engine_p1="titanium-v16" if v17_is_p0 else "titanium-v17",
        weights_p0=current_weights,
        weights_p1=current_weights,
        current_is_p0=v17_is_p0,
        opponent_engine="titanium-v16",
        opening_exploration=True,
        metadata={"matchup": "v17_vs_v16"},
    )


@dataclass(frozen=True)
class RunConfig:
    threads: int
    games: int
    time_sec: float
    nodes: int
    current_weights: Path
    opening_exploration: bool
    explore_chance: float
    out_dir: Path
    opening_index: Path | None
    resume: bool


def _load_resume_state(out_dir: Path) -> MatchState | None:
    summary_path = out_dir / "summary.json"
    if not summary_path.is_file():
        return None
    try:
        doc = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return MatchState(
        v17_wins=int(doc.get("v17_wins", 0)),
        v16_wins=int(doc.get("v16_wins", 0)),
        draws=int(doc.get("draws", 0)),
        completed=int(doc.get("completed_games", 0)),
        errors=int(doc.get("errors", 0)),
        persisted=int(doc.get("persisted_games", 0)),
    )


def parse_args() -> RunConfig:
    ap = argparse.ArgumentParser(description="v17 vs v16 through the generation pipeline (no pyarrow)")
    ap.add_argument("--threads", type=int, default=17)
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--time", type=float, default=1.0)
    ap.add_argument("--nodes", type=int, default=550000)
    ap.add_argument("--current-weights", type=Path, default=DEFAULT_CURRENT)
    ap.add_argument("--opening-exploration", action="store_true", default=True)
    ap.add_argument("--explore-chance", type=float, default=0.35)
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "runs" / "v17_vs_v16_pipeline")
    ap.add_argument("--opening-prefix-index", type=Path, default=DEFAULT_INDEX_PATH)
    ap.add_argument("--resume", action="store_true", help="Continue from existing summary.json in --out-dir")
    args = ap.parse_args()
    return RunConfig(
        threads=args.threads,
        games=args.games,
        time_sec=args.time,
        nodes=args.nodes,
        current_weights=args.current_weights,
        opening_exploration=args.opening_exploration,
        explore_chance=args.explore_chance,
        out_dir=args.out_dir,
        opening_index=args.opening_prefix_index,
        resume=args.resume,
    )


@dataclass
class MatchState:
    v17_wins: int = 0
    v16_wins: int = 0
    draws: int = 0
    completed: int = 0
    errors: int = 0
    persisted: int = 0


class MatchRunner:
    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self.stop = threading.Event()
        self.state = _load_resume_state(cfg.out_dir) if cfg.resume else MatchState()
        if self.state is None:
            self.state = MatchState()
        self.lock = threading.Lock()
        self.db_lock = threading.Lock()
        self.cfg.out_dir.mkdir(parents=True, exist_ok=True)
        self.strength_path = cfg.out_dir / "strength.tsv"
        self.summary_path = cfg.out_dir / "summary.json"
        self._prefix_index: OpeningPrefixIndex | None = None
        if cfg.opening_index and cfg.opening_index.is_file():
            try:
                self._prefix_index = OpeningPrefixIndex(cfg.opening_index)
            except Exception as exc:
                log(f"WARNING: could not load opening prefix index {cfg.opening_index}: {exc}")
        self._game_counter = 0

    def _write_summary(self) -> None:
        with self.lock:
            n = self.state.completed
            score = (self.state.v17_wins + 0.5 * self.state.draws) / n if n else 0.0
            summary = {
                "running": not self.stop.is_set(),
                "target_games": self.cfg.games,
                "completed_games": n,
                "persisted_games": self.state.persisted,
                "v17_wins": self.state.v17_wins,
                "v16_wins": self.state.v16_wins,
                "draws": self.state.draws,
                "score_v17": round(score, 4),
                "errors": self.state.errors,
                "updated_at": utc_now(),
            }
        tmp = self.summary_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        tmp.replace(self.summary_path)

    def _append_strength(self, r: dict) -> None:
        v17_is_p0 = r.get("engine_p0") == "titanium-v17"
        outcome_p0 = r.get("outcome_p0", 0)
        v17_won = (outcome_p0 > 0) if v17_is_p0 else (outcome_p0 < 0)
        row = [
            utc_now(),
            r.get("game_id"),
            r.get("matchup_kind"),
            r.get("engine_p0"),
            r.get("engine_p1"),
            str(outcome_p0),
            "1" if v17_is_p0 else "0",
            "1" if v17_won else "0",
            str(r.get("plies", 0)),
        ]
        header = (
            "recorded_at\tgame_id\tmatchup_kind\tengine_p0\tengine_p1\t"
            "outcome_p0\tv17_is_p0\tv17_won\tplies\n"
        )
        needs_header = not self.strength_path.exists()
        with self.strength_path.open("a", encoding="utf-8", newline="") as f:
            if needs_header:
                f.write(header)
            f.write("\t".join(row) + "\n")

    def _persist_game(self, r: dict) -> bool:
        """Write game to training DB if it passes sanity.  Strength is logged
        regardless of persistence, so a 200-game match stops at 200 completed
        games, not 200 teacher-quality games."""
        if r.get("aborted") or not r.get("moves"):
            return False
        if not opening_sanity_ok(r["moves"]):
            log_rejected_game(
                {
                    "game_id": r.get("game_id"),
                    "moves": r.get("moves"),
                    "matchup_kind": r.get("matchup_kind"),
                    "opponent_engine": r.get("opponent_engine"),
                    "engine_p0": r.get("engine_p0"),
                    "engine_p1": r.get("engine_p1"),
                    "weights_hash": r.get("weights_hash"),
                    "reason": "opening_sanity_failed",
                }
            )
            log(f"REJECT opening sanity: {r.get('game_id')}")
            return False

        with self.db_lock:
            games_db = open_db(GAMES_DB_PATH, GAMES_SCHEMA)
            labels_db = open_db(LABELS_DB_PATH, LABELS_SCHEMA)
            try:
                written, _positions, _labels = write_batch(
                    games_db,
                    labels_db,
                    [(r["game_id"], r["moves"], r["outcome_p0"], None, "pool_v17_vs_v16")],
                    chunk_size=512,
                    workers=1,
                )
            finally:
                games_db.close()
                labels_db.close()

        if written <= 0:
            log(f"REJECT eval-batch: {r.get('game_id')}")
            return False

        if self._prefix_index is not None:
            self._prefix_index.register_game(
                r["moves"],
                int(r["outcome_p0"]),
                source="pool_v17_vs_v16",
                max_ply=16,
            )

        with self.lock:
            self.state.persisted += 1
        return True

    def _record_outcome(self, r: dict) -> None:
        v17_is_p0 = r.get("engine_p0") == "titanium-v17"
        outcome_p0 = r.get("outcome_p0", 0)
        v17_won = (outcome_p0 > 0) if v17_is_p0 else (outcome_p0 < 0)
        v16_won = (outcome_p0 < 0) if v17_is_p0 else (outcome_p0 > 0)
        with self.lock:
            self.state.completed += 1
            if outcome_p0 == 0:
                self.state.draws += 1
            elif v17_won:
                self.state.v17_wins += 1
            elif v16_won:
                self.state.v16_wins += 1

    def _play_one(self, worker_id: int) -> dict | None:
        with self.lock:
            self._game_counter += 1
            gid_num = self._game_counter

        seed_now = int(time.time())
        rng = random.Random((seed_now << 16) ^ (worker_id << 8) ^ gid_num)
        matchup = _choose_v17_vs_v16(rng, current_weights=self.cfg.current_weights, previous_weights=None)

        game_seed = (seed_now << 16) ^ (worker_id << 8) ^ gid_num
        opening_enabled = self.cfg.opening_exploration and self._prefix_index is not None
        exploration = ExplorationConfig(
            start_ply=6,
            chance=0.0 if opening_enabled else self.cfg.explore_chance,
            max_loss_cp=140,
            candidate_count=18,
            top_n=8,
            temperature_cp=45.0,
            wall_bonus_cp=12,
            decay_after_hit=0.55,
            min_chance=0.03,
        )
        opening_exploration = OpeningExplorationConfig(
            enabled=opening_enabled,
            initial_temperature=1.0,
            after_ply4_temperature=1.0,
            decay_per_ply=0.08,
            min_while_known=0.45,
            max_exploration_ply=16,
            novel_prefix_temperature=0.0,
            high_freq_boost=0.1,
            max_loss_cp=140,
            candidate_count=18,
            top_n=8,
            wall_bonus_cp=12,
            prob_floor=0.08,
        )
        gid = f"v17v16_{seed_now}_{worker_id:02d}_{gid_num:06d}"
        sessions = GameSessions(
            p0=EngineSession(matchup.engine_p0, matchup.weights_p0),
            p1=EngineSession(matchup.engine_p1, matchup.weights_p1),
        )
        try:
            return play_one_game(
                gid,
                self.cfg.time_sec,
                matchup.weights_p0,
                matchup.weights_p1,
                mixed=True,
                current_is_p0=matchup.current_is_p0,
                exploration=exploration,
                opening_exploration=opening_exploration,
                prefix_index=self._prefix_index,
                rng=random.Random(game_seed),
                nodes=self.cfg.nodes if self.cfg.nodes > 0 else None,
                opening=[],
                game_seed=game_seed,
                weights_hash=_sha16(self.cfg.current_weights),
                engine_p0=matchup.engine_p0,
                engine_p1=matchup.engine_p1,
                matchup_kind=MATCHUP_V17_VS_V16,
                opponent_engine="titanium-v16",
                sessions=sessions,
            )
        finally:
            sessions.close()

    def worker_loop(self, worker_id: int) -> None:
        while not self.stop.is_set():
            try:
                r = self._play_one(worker_id)
                if not r:
                    continue
                self._record_outcome(r)
                self._append_strength(r)
                self._persist_game(r)
                self._write_summary()
                with self.lock:
                    snap = {
                        "completed": self.state.completed,
                        "v17": self.state.v17_wins,
                        "v16": self.state.v16_wins,
                        "draws": self.state.draws,
                    }
                    should_stop = self.state.completed >= self.cfg.games
                log(
                    f"game {r['game_id']}  {('v17' if ((r['outcome_p0'] > 0) == (r['engine_p0'] == 'titanium-v17')) else 'v16') + ' wins':8s}  "
                    f"({r['engine_p0']} p0, {r['plies']} plies)  "
                    f"v17:{snap['v17']}W {snap['draws']}D {snap['v16']}L  "
                    f"score={((snap['v17'] + 0.5 * snap['draws']) / snap['completed'] if snap['completed'] else 0.0):.3f}"
                )
                if should_stop:
                    self.stop.set()
                    break
            except Exception as exc:
                with self.lock:
                    self.state.errors += 1
                log(f"worker {worker_id} error: {exc}")

    def run(self) -> int:
        remaining = max(0, self.cfg.games - self.state.completed)
        log(
            f"v17_vs_v16 pipeline  engine={ENGINE_BIN}  threads={self.cfg.threads}  "
            f"target={self.cfg.games}  already={self.state.completed}  remaining={remaining}"
        )
        if remaining <= 0:
            log("Target already reached — nothing to do.")
            self._write_summary()
            return 0
        self._write_summary()
        threads = [
            threading.Thread(target=self.worker_loop, args=(wid,), name=f"v17v16-worker-{wid}")
            for wid in range(self.cfg.threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.stop.set()
        self._write_summary()
        with self.lock:
            n = self.state.completed
            score = (self.state.v17_wins + 0.5 * self.state.draws) / n if n else 0.0
        log(
            f"\nFINAL v17 vs v16: {self.state.v17_wins}W {self.state.draws}D {self.state.v16_wins}L / {n}  "
            f"score={score:.3f}  errors={self.state.errors}  persisted={self.state.persisted}"
        )
        return 0


def main() -> int:
    cfg = parse_args()

    def on_signal(_signum: int, _frame: Any) -> None:
        print("\nStopping v17_vs_v16 pipeline...", flush=True)
        sys.exit(0)

    signal.signal(signal.SIGINT, on_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, on_signal)

    runner = MatchRunner(cfg)
    return runner.run()


if __name__ == "__main__":
    raise SystemExit(main())
