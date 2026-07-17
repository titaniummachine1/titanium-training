#!/usr/bin/env python3
"""Generic engine-variant A/B through the generation pipeline.

Compare any two session engine flags (e.g. titanium-v17 vs titanium-v17-route-touch)
with exploration/temperature matching normal pool self-play.

Usage:
  python tools/binary_match/engine_variant_pipeline.py \\
    --engine-a titanium-v17-route-touch --engine-b titanium-v17 \\
    --threads 17 --games 50 --time 1.0 \\
    --out-dir tools/binary_match/runs/v17_route_touch_vs_v17
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
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
from generation_matchup import GenerationMatchup
from opening_prefix_index import DEFAULT_INDEX_PATH, OpeningPrefixIndex
from self_play_overnight import (
    DEFAULT_CURRENT,
    ExplorationConfig,
    GameSessions,
    OpeningExplorationConfig,
    play_one_game,
)
from titanium_training.paths import ENGINE_BIN, REPO_ROOT

LOG_DIR = _TRAINING / "data" / "overnight_logs"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(msg: str) -> None:
    print(msg, flush=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with (LOG_DIR / "engine_variant_pipeline.log").open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def wilson_lower_bound(wins: float, n: int, z: float = 1.96) -> float:
    if n <= 0:
        return 0.0
    p = wins / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return (centre - margin) / denom


def _sha16(path: Path | None) -> str | None:
    if not path or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


@dataclass(frozen=True)
class RunConfig:
    engine_a: str
    engine_b: str
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


@dataclass
class MatchState:
    a_wins: int = 0
    b_wins: int = 0
    draws: int = 0
    completed: int = 0
    errors: int = 0
    persisted: int = 0


def _load_resume_state(out_dir: Path) -> MatchState | None:
    summary_path = out_dir / "summary.json"
    if not summary_path.is_file():
        return None
    try:
        doc = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return MatchState(
        a_wins=int(doc.get("a_wins", 0)),
        b_wins=int(doc.get("b_wins", 0)),
        draws=int(doc.get("draws", 0)),
        completed=int(doc.get("completed_games", 0)),
        errors=int(doc.get("errors", 0)),
        persisted=int(doc.get("persisted_games", 0)),
    )


def parse_args() -> RunConfig:
    ap = argparse.ArgumentParser(description="Engine variant A/B via generation pipeline")
    ap.add_argument("--engine-a", required=True, help="challenger session engine flag")
    ap.add_argument("--engine-b", required=True, help="baseline session engine flag")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--time", type=float, default=1.0)
    ap.add_argument("--nodes", type=int, default=550000)
    ap.add_argument("--current-weights", type=Path, default=DEFAULT_CURRENT)
    ap.add_argument("--opening-exploration", action="store_true", default=True)
    ap.add_argument("--explore-chance", type=float, default=0.35)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--opening-prefix-index", type=Path, default=DEFAULT_INDEX_PATH)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    return RunConfig(
        engine_a=args.engine_a,
        engine_b=args.engine_b,
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


def _choose_matchup(
    rng: random.Random,
    *,
    engine_a: str,
    engine_b: str,
    current_weights: Path,
) -> GenerationMatchup:
    a_is_p0 = rng.random() < 0.5
    return GenerationMatchup(
        kind="engine_variant_ab",
        engine_p0=engine_a if a_is_p0 else engine_b,
        engine_p1=engine_b if a_is_p0 else engine_a,
        weights_p0=current_weights,
        weights_p1=current_weights,
        current_is_p0=a_is_p0,
        opponent_engine=engine_b,
        opening_exploration=True,
        metadata={"engine_a": engine_a, "engine_b": engine_b},
    )


class MatchRunner:
    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self.stop = threading.Event()
        self.state = _load_resume_state(cfg.out_dir) if cfg.resume else MatchState()
        if self.state is None:
            self.state = MatchState()
        self.lock = threading.Lock()
        self.db_lock = threading.Lock()
        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        self.strength_path = cfg.out_dir / "strength.tsv"
        self.summary_path = cfg.out_dir / "summary.json"
        self._prefix_index: OpeningPrefixIndex | None = None
        if cfg.opening_index and cfg.opening_index.is_file():
            try:
                self._prefix_index = OpeningPrefixIndex(cfg.opening_index)
            except Exception as exc:
                log(f"WARNING: opening index unavailable: {exc}")
        self._game_counter = 0

    def _write_summary(self) -> None:
        with self.lock:
            n = self.state.completed
            score_a = (self.state.a_wins + 0.5 * self.state.draws) / n if n else 0.0
            summary = {
                "running": not self.stop.is_set(),
                "engine_a": self.cfg.engine_a,
                "engine_b": self.cfg.engine_b,
                "target_games": self.cfg.games,
                "completed_games": n,
                "persisted_games": self.state.persisted,
                "a_wins": self.state.a_wins,
                "b_wins": self.state.b_wins,
                "draws": self.state.draws,
                "score_a": round(score_a, 4),
                "wilson_lb_a": round(wilson_lower_bound(self.state.a_wins + 0.5 * self.state.draws, n), 4),
                "errors": self.state.errors,
                "updated_at": utc_now(),
            }
        tmp = self.summary_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        tmp.replace(self.summary_path)

    def _record_outcome(self, r: dict[str, Any]) -> None:
        a_is_p0 = r.get("engine_p0") == self.cfg.engine_a
        outcome_p0 = int(r.get("outcome_p0", 0))
        a_won = (outcome_p0 > 0) if a_is_p0 else (outcome_p0 < 0)
        b_won = (outcome_p0 < 0) if a_is_p0 else (outcome_p0 > 0)
        with self.lock:
            self.state.completed += 1
            if outcome_p0 == 0:
                self.state.draws += 1
            elif a_won:
                self.state.a_wins += 1
            elif b_won:
                self.state.b_wins += 1

    def _worker(self, worker_id: int) -> None:
        rng = random.Random(time.time_ns() ^ worker_id)
        exploration = ExplorationConfig(chance=self.cfg.explore_chance)
        opening_cfg = OpeningExplorationConfig(enabled=self.cfg.opening_exploration)
        while not self.stop.is_set():
            with self.lock:
                if self.state.completed >= self.cfg.games:
                    break
            matchup = _choose_matchup(
                rng,
                engine_a=self.cfg.engine_a,
                engine_b=self.cfg.engine_b,
                current_weights=self.cfg.current_weights,
            )
            with self.lock:
                self._game_counter += 1
                game_id = f"evab_{int(time.time())}_{worker_id:02d}_{self._game_counter:06d}"
            sessions = GameSessions(
                p0=EngineSession(matchup.engine_p0, matchup.weights_p0),
                p1=EngineSession(matchup.engine_p1, matchup.weights_p1),
            )
            try:
                r = play_one_game(
                    game_id=game_id,
                    time_sec=self.cfg.time_sec,
                    w_p0=matchup.weights_p0,
                    w_p1=matchup.weights_p1,
                    mixed=False,
                    current_is_p0=matchup.current_is_p0,
                    nodes=self.cfg.nodes,
                    exploration=exploration,
                    opening_exploration=opening_cfg,
                    prefix_index=self._prefix_index,
                    rng=rng,
                    engine_p0=matchup.engine_p0,
                    engine_p1=matchup.engine_p1,
                    matchup_kind=matchup.kind,
                    opponent_engine=matchup.opponent_engine,
                    sessions=sessions,
                )
                r["weights_hash"] = _sha16(self.cfg.current_weights)
                self._record_outcome(r)
                self._write_summary()
                a_is_p0 = r.get("engine_p0") == self.cfg.engine_a
                outcome_p0 = int(r.get("outcome_p0", 0))
                a_won = (outcome_p0 > 0) if a_is_p0 else (outcome_p0 < 0)
                winner = self.cfg.engine_a if a_won else (self.cfg.engine_b if outcome_p0 != 0 else "draw")
                log(
                    f"game {self.state.completed:4d}  {winner:28s}  "
                    f"({r.get('engine_p0')} vs {r.get('engine_p1')}, {r.get('plies', 0)} plies)  "
                    f"score_a={self.state.a_wins + 0.5 * self.state.draws}/{self.state.completed}"
                )
            except Exception as exc:
                with self.lock:
                    self.state.errors += 1
                log(f"ERROR worker {worker_id}: {exc}")
                self._write_summary()

    def run(self) -> int:
        log(
            f"START {self.cfg.engine_a} vs {self.cfg.engine_b}  "
            f"games={self.cfg.games} threads={self.cfg.threads} time={self.cfg.time_sec}s"
        )
        threads = [
            threading.Thread(target=self._worker, args=(i,), daemon=True)
            for i in range(self.cfg.threads)
        ]
        for t in threads:
            t.start()

        def _handle_sig(*_args: object) -> None:
            log("STOP requested")
            self.stop.set()

        signal.signal(signal.SIGINT, _handle_sig)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, _handle_sig)

        for t in threads:
            t.join()
        self.stop.set()
        self._write_summary()
        n = self.state.completed
        score_a = (self.state.a_wins + 0.5 * self.state.draws) / n if n else 0.0
        lb = wilson_lower_bound(self.state.a_wins + 0.5 * self.state.draws, n)
        log(f"DONE  {self.cfg.engine_a}: {self.state.a_wins}W {self.state.draws}D {self.state.b_wins}L  score={score_a:.3f}  wilson_lb={lb:.3f}")
        return 0 if lb > 0.5 else 1


def main() -> int:
    if not ENGINE_BIN.is_file():
        log(f"ERROR: engine binary missing at {ENGINE_BIN}")
        return 2
    cfg = parse_args()
    return MatchRunner(cfg).run()


if __name__ == "__main__":
    raise SystemExit(main())
