#!/usr/bin/env python3
"""Continuous self-play pool — per-game persistence with periodic training.

Workflow:
  generate game -> persist games.db + labels.db + teacher parquet immediately
  -> repeat; every N successfully persisted games (default 1024):
  drain in-flight games -> consistency flush -> incremental cache append
  -> train 1 epoch -> promotion gate -> resume self-play

All worker threads normally play games. Use --reserve-worker0 when the host
needs worker id 0 kept idle for interactive use.

Run:
  python training/continuous_pool.py --threads 4 --batch-games 1024
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parent.parent
_TRAINING = _REPO / "training"
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

from db_import import GAMES_DB_PATH, GAMES_SCHEMA, LABELS_DB_PATH, LABELS_SCHEMA, open_db, write_batch
from pool_lock import PoolInstanceLock, TrainerRunLock, release_pool_lock, trainer_lock_held
from pool_state_io import load_json, save_json_atomic
from self_play_overnight import (
    DEFAULT_CURRENT,
    DEFAULT_FROZEN,
    DEFAULT_PREVIOUS,
    ExplorationConfig,
    GameSessions,
    OpeningExplorationConfig,
    play_one_game,
)
from engine_session import EngineSession
from opening_prefix_index import (
    DEFAULT_INDEX_PATH,
    OpeningMetricsSnapshot,
    OpeningPrefixIndex,
    update_metrics,
)
from game_opening_gate import log_rejected_game, opening_sanity_ok
from generation_matchup import MATCHUP_PRIOR_EPOCH, choose_generation_matchup
from streaming_checkpoint_chain import (
    candidate_weights_path,
    freeze_worker_game_weights,
    pool_weights_path,
    previous_opponent_weights_path,
    refresh_pool_weights_snapshot,
)
from sync_overnight_to_teacher import pool_teacher_dir, sync_single_game, sync_stragglers

CACHE_DIR = _TRAINING / "data" / "feature_cache"
RUN_DIR = _TRAINING / "runs" / "v16"
LOG_DIR = _TRAINING / "data" / "overnight_logs"
STATE_PATH = LOG_DIR / "continuous_pool_state.json"
PAUSE_EPOCHS_PATH = LOG_DIR / "pause_training_epochs.json"
STREAMING_READY_PATH = LOG_DIR / "streaming_training_ready.json"
EXPLORATION_META_PATH = LOG_DIR / "opening_exploration_games.jsonl"
STRENGTH_MEASURE_PATH = LOG_DIR / "strength_games.tsv"
OPENING_ENABLED_PATH = LOG_DIR / "opening_exploration_enabled.json"
FROZEN = _REPO / "engine" / "src" / "titanium" / "net_weights_frozen.bin"
ENGINE_WEIGHTS = _REPO / "engine" / "src" / "titanium" / "net_weights.bin"
BEST = RUN_DIR / "net_weights_best.bin"
PREVIOUS = RUN_DIR / "net_weights_previous.bin"

DEFAULT_EPOCH_GAMES = 1024


def log(msg: str) -> None:
    print(msg, flush=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with (LOG_DIR / "continuous_pool.log").open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def _weights_sha16(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _tsv_bool(value: object) -> str:
    if value is True:
        return "1"
    if value is False:
        return "0"
    return ""


def _tsv_cell(value: object) -> str:
    if value is None:
        return ""
    return str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ")


def run_cmd(cmd: list[str]) -> int:
    log(f"$ {' '.join(cmd)}")
    env = {**dict(__import__("os").environ), "PYTHONPATH": str(_TRAINING)}
    return subprocess.call(cmd, cwd=str(_REPO), env=env)


@dataclass
class PoolState:
    epoch: int = 0
    pool_games_total: int = 0
    games_since_epoch: int = 0
    positions_since_epoch: int = 0
    cache_generation: int = 0
    mixed_wins: int = 0
    mixed_losses: int = 0
    mixed_draws: int = 0


@dataclass
class PoolConfig:
    threads: int = 8
    reserve_worker0: bool = False
    time_sec: float = 2.0
    nodes: int = 0
    batch_games: int = DEFAULT_EPOCH_GAMES
    train_after_new_positions: int = 0
    use_position_trigger: bool = False
    recent_replay_fraction: float = 0.0
    recent_window_games: int = 128
    same_net_pct: float = 0.7
    current: Path = DEFAULT_CURRENT
    previous: Path = DEFAULT_PREVIOUS
    saturate_threshold: float = 0.45
    saturate_min_mixed: int = 32
    no_saturate: bool = False
    no_parity: bool = False
    saturate_grace_epochs: int = 0
    initial_epoch: bool = True
    force_epoch: bool = False
    db_streaming: bool = False
    oracle_url: str | None = None
    oracle_token: str | None = None
    oracle_poll_sec: float = 30.0
    explore_chance: float = 0.0
    explore_start_ply: int = 6
    explore_max_loss_cp: int = 140
    explore_candidate_count: int = 18
    explore_top_n: int = 8
    explore_temperature_cp: float = 45.0
    explore_wall_bonus_cp: int = 12
    explore_decay_after_hit: float = 0.55
    explore_min_chance: float = 0.03
    explore_worker0: bool = False
    opening_pct: float = 0.0
    opening_moves: tuple[str, ...] = ()
    opening_exploration: bool = False
    opening_temperature_initial: float = 1.0
    opening_temperature_after_ply4: float = 1.0
    opening_temperature_decay_per_ply: float = 0.08
    opening_temperature_min_while_known: float = 0.45
    opening_exploration_max_ply: int = 16
    novel_prefix_temperature: float = 0.0
    opening_high_freq_boost: float = 0.1
    opening_prob_floor: float = 0.08
    opening_prefix_index: Path = DEFAULT_INDEX_PATH
    rebuild_prefix_index: bool = False
    opening_explore_worker0: bool = False


@dataclass
class PersistOutcome:
    new_positions: int
    counted: bool
    cache_total: int = 0


class ContinuousPool:
    def __init__(self, cfg: PoolConfig):
        self.cfg = cfg
        self._stop = threading.Event()
        self._accept_games = threading.Event()
        self._accept_games.set()
        self._lock = threading.Lock()
        self._db_lock = threading.Lock()
        self._teacher_lock = threading.Lock()
        self._cache_lock = threading.Lock()
        self._epoch_lock = threading.Lock()
        self._inflight_lock = threading.Lock()
        self._inflight = 0
        self._state = self._load_state()
        self._game_counter = 0
        self._rng = random.Random(int(time.time()) % 1_000_000)
        self._persist_failures = 0
        self._max_persist_failures = 5
        self._oracle_importer = None
        self._prefix_index: OpeningPrefixIndex | None = None
        self._opening_metrics = OpeningMetricsSnapshot()
        self._metrics_log_every = 256
        self._skip_cache_append = False

    def _load_state(self) -> PoolState:
        d = load_json(STATE_PATH)
        if not d:
            return PoolState()
        pool_total = int(d.get("pool_games_total", d.get("total_games", 0)))
        return PoolState(
            epoch=int(d.get("epoch", 0)),
            pool_games_total=pool_total,
            games_since_epoch=int(d.get("games_since_epoch", 0)),
            positions_since_epoch=int(d.get("positions_since_epoch", 0)),
            cache_generation=int(d.get("cache_generation", 0)),
            mixed_wins=int(d.get("mixed_wins", 0)),
            mixed_losses=int(d.get("mixed_losses", 0)),
            mixed_draws=int(d.get("mixed_draws", 0)),
        )

    def _save_state(self) -> None:
        save_json_atomic(
            STATE_PATH,
            {
                "epoch": self._state.epoch,
                "pool_games_total": self._state.pool_games_total,
                "total_games": self._state.pool_games_total,
                "games_since_epoch": self._state.games_since_epoch,
                "positions_since_epoch": self._state.positions_since_epoch,
                "cache_generation": self._state.cache_generation,
                "mixed_wins": self._state.mixed_wins,
                "mixed_losses": self._state.mixed_losses,
                "mixed_draws": self._state.mixed_draws,
            },
        )

    def _begin_inflight(self) -> None:
        with self._inflight_lock:
            self._inflight += 1

    def _end_inflight(self) -> None:
        with self._inflight_lock:
            self._inflight -= 1

    def _drain_inflight(self, *, timeout_sec: float = 600.0, reserve: int = 0) -> bool:
        """Wait until inflight count drops to *reserve* or below.

        Pass reserve=1 when the calling thread is itself counted as inflight
        (e.g. the epoch-transition worker): that thread won't decrement until
        after this function returns, so the target is 1, not 0.
        """
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            with self._inflight_lock:
                if self._inflight <= reserve:
                    return True
            time.sleep(0.25)
        with self._inflight_lock:
            remaining = self._inflight
        log(f"WARN: drain timeout with {remaining - reserve} in-flight game(s)")
        return remaining <= reserve

    def _has_distinct_prior(self) -> bool:
        """True when a prior-epoch checkpoint exists and differs from current (not frozen fallback)."""
        cur = self.cfg.current
        prev = self.cfg.previous
        if not cur.is_file() or not prev.is_file():
            return False
        cur_sha = _weights_sha16(cur)
        prev_sha = _weights_sha16(prev)
        return bool(cur_sha and prev_sha and cur_sha != prev_sha)

    def _effective_same_net_pct(self) -> float:
        """70% same-net / 30% vs-prior when a distinct prior exists; else 100% latest."""
        if not self._has_distinct_prior():
            return 1.0
        return self.cfg.same_net_pct

    def _weights(self) -> tuple[Path, Path]:
        refresh_pool_weights_snapshot()
        cur = pool_weights_path()
        if not cur.is_file():
            cur = self.cfg.current if self.cfg.current.is_file() else candidate_weights_path()
        prev_path = previous_opponent_weights_path()
        prev = prev_path if prev_path and prev_path.is_file() else (
            self.cfg.previous if self.cfg.previous.is_file() else cur
        )
        return cur, prev

    def _wait_for_trainer_lock_clear(self, *, timeout_sec: float = 7200.0) -> None:
        deadline = time.monotonic() + timeout_sec
        while trainer_lock_held() and time.monotonic() < deadline:
            time.sleep(0.5)

    def _play_one(self, worker_id: int) -> dict | None:
        self._wait_for_trainer_lock_clear()
        refresh_pool_weights_snapshot()
        live_cur, live_prev = self._weights()
        if not live_cur.is_file():
            log(f"[w{worker_id}] missing current weights {live_cur}")
            time.sleep(5)
            return None
        # Freeze this game's weights ONCE, here, at game start. A game spawns one
        # titanium subprocess per move; if it kept reading the shared mutable
        # pool-active path, a checkpoint accept landing mid-game could swap a
        # side's weights partway through. These frozen copies only change between
        # games, never during one.
        cur, prev = freeze_worker_game_weights(worker_id, current=live_cur, previous=live_prev)

        with self._lock:
            self._game_counter += 1
            gid_num = self._game_counter

        seed_now = int(time.time())
        rng = random.Random((seed_now << 16) ^ (worker_id << 8) ^ gid_num)
        matchup = choose_generation_matchup(
            self._rng,
            current_weights=cur,
            previous_weights=prev if prev.resolve() != cur.resolve() else None,
        )
        mixed = matchup.kind == MATCHUP_PRIOR_EPOCH
        current_is_p0 = matchup.current_is_p0
        w_p0, w_p1 = matchup.weights_p0, matchup.weights_p1

        opening_moves = list(self.cfg.opening_moves) if (
            self.cfg.opening_moves and rng.random() < self.cfg.opening_pct
        ) else []
        gid = f"pool_{seed_now}_{worker_id:02d}_{gid_num:06d}"
        game_seed = (seed_now << 16) ^ (worker_id << 8) ^ gid_num
        explore_chance = self.cfg.explore_chance
        opening_enabled = matchup.opening_exploration and self.cfg.opening_exploration and OPENING_ENABLED_PATH.is_file()
        if worker_id == 0 and not self.cfg.explore_worker0:
            explore_chance = 0.0
        if worker_id == 0 and not self.cfg.opening_explore_worker0:
            opening_enabled = False
        exploration = ExplorationConfig(
            start_ply=self.cfg.explore_start_ply,
            chance=explore_chance if not opening_enabled else 0.0,
            max_loss_cp=self.cfg.explore_max_loss_cp,
            candidate_count=self.cfg.explore_candidate_count,
            top_n=self.cfg.explore_top_n,
            temperature_cp=self.cfg.explore_temperature_cp,
            wall_bonus_cp=self.cfg.explore_wall_bonus_cp,
            decay_after_hit=self.cfg.explore_decay_after_hit,
            min_chance=self.cfg.explore_min_chance,
        )
        opening_exploration = OpeningExplorationConfig(
            enabled=opening_enabled,
            initial_temperature=self.cfg.opening_temperature_initial,
            after_ply4_temperature=self.cfg.opening_temperature_after_ply4,
            decay_per_ply=self.cfg.opening_temperature_decay_per_ply,
            min_while_known=self.cfg.opening_temperature_min_while_known,
            max_exploration_ply=self.cfg.opening_exploration_max_ply,
            novel_prefix_temperature=self.cfg.novel_prefix_temperature,
            high_freq_boost=self.cfg.opening_high_freq_boost,
            max_loss_cp=self.cfg.explore_max_loss_cp,
            candidate_count=self.cfg.explore_candidate_count,
            top_n=self.cfg.explore_top_n,
            wall_bonus_cp=self.cfg.explore_wall_bonus_cp,
            prob_floor=self.cfg.opening_prob_floor,
        )
        rng = random.Random(game_seed)
        # One warm engine process per side for the WHOLE game (TT, dist-topology
        # LRU, killers, history all stay hot across every ply) instead of a fresh
        # cold process spawned every single move.
        sessions = GameSessions(
            p0=EngineSession(matchup.engine_p0, w_p0),
            p1=EngineSession(matchup.engine_p1, w_p1),
        )
        try:
            return play_one_game(
                gid,
                self.cfg.time_sec,
                w_p0,
                w_p1,
                mixed,
                current_is_p0,
                exploration=exploration,
                opening_exploration=opening_exploration,
                prefix_index=self._prefix_index,
                rng=rng,
                nodes=self.cfg.nodes if self.cfg.nodes > 0 else None,
                opening=opening_moves,
                game_seed=game_seed,
                weights_hash=_weights_sha16(cur),
                engine_p0=matchup.engine_p0,
                engine_p1=matchup.engine_p1,
                matchup_kind=matchup.kind,
                opponent_engine=matchup.opponent_engine,
                sessions=sessions,
            )
        finally:
            sessions.close()

    def _persist_game(self, r: dict) -> PersistOutcome:
        if r.get("aborted"):
            return PersistOutcome(0, False)
        if not r.get("moves"):
            return PersistOutcome(0, False)

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
            log(f"REJECT opening sanity: {r.get('game_id')} prefix={' '.join(r['moves'][:4])}")
            return PersistOutcome(0, False)

        from sync_overnight_to_teacher import load_synced_ids

        if r["game_id"] in load_synced_ids():
            return PersistOutcome(0, False)

        src = f"pool_{r.get('matchup_kind', 'generation')}"
        written_games = 0
        with self._db_lock:
            games_db = open_db(GAMES_DB_PATH, GAMES_SCHEMA)
            labels_db = open_db(LABELS_DB_PATH, LABELS_SCHEMA)
            try:
                written_games, written_positions, _written_labels = write_batch(
                    games_db,
                    labels_db,
                    [(r["game_id"], r["moves"], r["outcome_p0"], None, src)],
                    chunk_size=512,
                    workers=1,
                )
            finally:
                games_db.close()
                labels_db.close()
        if written_games <= 0:
            log_rejected_game(
                {
                    "game_id": r.get("game_id"),
                    "moves": r.get("moves"),
                    "matchup_kind": r.get("matchup_kind"),
                    "opponent_engine": r.get("opponent_engine"),
                    "engine_p0": r.get("engine_p0"),
                    "engine_p1": r.get("engine_p1"),
                    "weights_hash": r.get("weights_hash"),
                    "reason": "engine_eval_batch_rejected",
                }
            )
            log(f"REJECT engine eval-batch: {r.get('game_id')} plies={len(r.get('moves') or [])}")
            return PersistOutcome(0, False)

        with self._teacher_lock:
            sync_result = sync_single_game(
                r["game_id"],
                dataset_dir=pool_teacher_dir(),
                teacher_lock=None,
            )
        if sync_result.get("skipped") or not sync_result.get("counted", True):
            return PersistOutcome(0, False)

        cache_appended = 0
        cache_total = 0
        if not self._skip_cache_append:
            with self._cache_lock:
                from incremental_feature_cache import append_game_to_cache

                cache_stats = append_game_to_cache(CACHE_DIR, r["moves"], int(r["outcome_p0"]))
                if not cache_stats.get("ok"):
                    raise RuntimeError(f"cache append failed: {cache_stats.get('reason', cache_stats)}")
                cache_appended = int(cache_stats.get("appended", 0) or 0)
                cache_total = int(cache_stats.get("n_total", 0) or 0)

        parquet_new = int(sync_result.get("new_positions", 0) or 0)
        if parquet_new <= 0 and written_positions > 0:
            log(f"WARNING teacher sync added no positions after engine accepted {r.get('game_id')}")

        if self._prefix_index is not None:
            src_tag = "pool_generation_mixed" if r["mixed"] else "pool_generation_selfplay"
            self._prefix_index.register_game(
                r["moves"],
                int(r["outcome_p0"]),
                source=src_tag,
                max_ply=self.cfg.opening_exploration_max_ply,
            )
        if self.cfg.opening_exploration and not r.get("mixed"):
            src_tag = "pool_generation_selfplay"
            self._append_exploration_metadata(r, src_tag)

        return PersistOutcome(max(cache_appended, parquet_new), True, cache_total)

    def _append_exploration_metadata(self, r: dict, source: str) -> None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "game_id": r["game_id"],
            "source": source,
            "outcome_p0": r["outcome_p0"],
            "plies": r["plies"],
            "novel_prefix": r.get("novel_prefix"),
            "novel_exit_ply": r.get("novel_exit_ply"),
            "exited_novel_tree": r.get("exited_novel_tree"),
            "explored_moves": r.get("explored_moves"),
            "move_temperatures": r.get("move_temperatures"),
            "prefix_counts_at_explore": r.get("prefix_counts_at_explore"),
            "exploration_quality_rejects": r.get("exploration_quality_rejects"),
            "game_seed": r.get("game_seed"),
            "weights_hash": r.get("weights_hash"),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        with EXPLORATION_META_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, separators=(",", ":")) + "\n")

    def _append_strength_measure(self, r: dict) -> None:
        if not r.get("mixed"):
            return
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        header = (
            "recorded_at\tepoch\tgame_id\tmatchup_kind\topponent_engine\t"
            "current_is_p0\toutcome_p0\tcurrent_won\tengine_p0\tengine_p1\t"
            "weights_hash\tplies\n"
        )
        row = [
            datetime.now(timezone.utc).isoformat(),
            self._state.epoch,
            r.get("game_id"),
            r.get("matchup_kind"),
            r.get("opponent_engine"),
            _tsv_bool(r.get("current_is_p0")),
            r.get("outcome_p0"),
            _tsv_bool(r.get("current_won")),
            r.get("engine_p0"),
            r.get("engine_p1"),
            r.get("weights_hash"),
            r.get("plies"),
        ]
        needs_header = not STRENGTH_MEASURE_PATH.exists()
        with STRENGTH_MEASURE_PATH.open("a", encoding="utf-8", newline="") as f:
            if needs_header:
                f.write(header)
            f.write("\t".join(_tsv_cell(v) for v in row) + "\n")

    def _mixed_win_rate(self) -> float:
        d = self._state.mixed_wins + self._state.mixed_losses
        return self._state.mixed_wins / d if d else 0.5

    def _record_game(self, r: dict, outcome: PersistOutcome) -> int:
        if not outcome.counted:
            return self._state.games_since_epoch
        with self._lock:
            self._state.pool_games_total += 1
            if self._counts_toward_epoch(outcome.new_positions):
                self._state.games_since_epoch += 1
                self._state.positions_since_epoch += max(0, outcome.new_positions)
            if outcome.cache_total > 0:
                self._state.cache_generation = outcome.cache_total
            if r.get("mixed"):
                if r.get("current_won") is True:
                    self._state.mixed_wins += 1
                elif r.get("current_won") is False:
                    self._state.mixed_losses += 1
                else:
                    self._state.mixed_draws += 1
                self._append_strength_measure(r)
            n = self._state.games_since_epoch
            self._save_state()
            return n

    def _record_remote_import(self, item: dict[str, Any]) -> None:
        cache_appended = 0
        cache_total = 0
        game_id = str(item.get("game_id", ""))
        if game_id and not self._skip_cache_append:
            with self._cache_lock:
                from incremental_feature_cache import append_db_game_to_cache

                cache_stats = append_db_game_to_cache(CACHE_DIR, game_id)
                if cache_stats.get("ok"):
                    cache_appended = int(cache_stats.get("appended", 0) or 0)
                    cache_total = int(cache_stats.get("n_total", 0) or 0)
        with self._lock:
            self._state.pool_games_total += 1
            added = max(cache_appended, int(item.get("new_positions", 0) or 0))
            if self._counts_toward_epoch(added):
                self._state.games_since_epoch += 1
                self._state.positions_since_epoch += added
            if cache_total > 0:
                self._state.cache_generation = cache_total
            self._save_state()
            log(
                f"[oracle] imported {item.get('game_id')} "
                f"gen={item.get('generation_id')} matchup={item.get('matchup_type')} "
                f"cache+={cache_appended} cache={cache_total or self._state.cache_generation} "
                f"new_pos={item.get('new_positions', 0)} "
                f"epoch={self._epoch_progress_label()}"
            )

    def _epoch_progress_label(self) -> str:
        """Human-readable epoch progress (position trigger preferred over bogus game cap)."""
        with self._lock:
            pos = self._state.positions_since_epoch
            games = self._state.games_since_epoch
        if self.cfg.use_position_trigger and self.cfg.train_after_new_positions > 0:
            cap = self.cfg.train_after_new_positions
            return f"new_positions={pos}/{cap} (games_played={games})"
        return f"games={games}/{self.cfg.batch_games}"

    def _counts_toward_epoch(self, new_positions: int) -> bool:
        """Only fresh training rows should advance the epoch trigger."""
        return int(new_positions) > 0

    def _reset_epoch_counters(self) -> None:
        with self._lock:
            self._state.games_since_epoch = 0
            self._state.positions_since_epoch = 0
            self._state.mixed_wins = 0
            self._state.mixed_losses = 0
            self._state.mixed_draws = 0
            self._save_state()

    def _epoch_ready(self) -> bool:
        with self._lock:
            if (
                self.cfg.use_position_trigger
                and self.cfg.train_after_new_positions > 0
                and self._state.positions_since_epoch >= self.cfg.train_after_new_positions
            ):
                return True
            return self._state.games_since_epoch >= self.cfg.batch_games

    def _trigger_reason(self) -> str:
        with self._lock:
            if (
                self.cfg.use_position_trigger
                and self.cfg.train_after_new_positions > 0
                and self._state.positions_since_epoch >= self.cfg.train_after_new_positions
            ):
                return "position_threshold"
            return "game_count"

    def _streaming_training_ready(self) -> bool:
        if not STREAMING_READY_PATH.is_file():
            return False
        try:
            return bool(json.loads(STREAMING_READY_PATH.read_text(encoding="utf-8")).get("ready"))
        except Exception:
            return False

    def _try_epoch(self) -> None:
        pause_active = PAUSE_EPOCHS_PATH.is_file()
        streaming_ok = self._streaming_training_ready()
        if pause_active and not streaming_ok:
            try:
                pause = json.loads(PAUSE_EPOCHS_PATH.read_text(encoding="utf-8"))
            except Exception:
                pause = {"reason": "unknown"}
            if self._epoch_ready():
                with self._lock:
                    games_n = self._state.games_since_epoch
                    pos_n = self._state.positions_since_epoch
                log(
                    f"Epoch trigger reached during rebuild pause ({pos_n} positions, "
                    f"{games_n} games): {pause.get('reason', pause)} — clearing counters"
                )
                self._reset_epoch_counters()
            return
        if not self._epoch_lock.acquire(blocking=False):
            return
        try:
            if not self._epoch_ready():
                return
            with self._lock:
                games_n = self._state.games_since_epoch
                pos_n = self._state.positions_since_epoch
            reason = self._trigger_reason()
            log(
                f"=== epoch transition announced ({reason}): "
                f"{self._epoch_progress_label()} ==="
            )
            self._accept_games.clear()
            try:
                use_db = self.cfg.db_streaming or (pause_active and streaming_ok)
                if use_db:
                    if self.cfg.db_streaming:
                        log("Database-first streaming training (forced --db-streaming: teacher anchor + sane self-play, opening-gated)")
                    else:
                        log("Database-first streaming training (cache rebuild non-blocking)")
                saturated = self._run_epoch(_caller_inflight=True, use_db_streaming=use_db)
                if saturated:
                    log("SATURATED — current net weaker than previous; stopping pool.")
                    self._stop.set()
            finally:
                self._accept_games.set()
        finally:
            self._epoch_lock.release()

    def _consistency_flush(self) -> dict:
        with self._teacher_lock:
            stats = sync_stragglers(dataset_dir=pool_teacher_dir(), limit=512)
        log(f"Consistency flush: {stats}")
        return stats

    def _snapshot_previous_weights(self) -> None:
        from streaming_checkpoint_chain import atomic_copy2

        if BEST.is_file():
            atomic_copy2(BEST, PREVIOUS)
            log(f"Previous weights snapshot sha={_weights_sha16(PREVIOUS)}")

    def _run_epoch(self, *, rebuild_cache: bool = True, _caller_inflight: bool = False, use_db_streaming: bool = False) -> bool:
        # reserve=1 when the calling worker thread is itself counted as inflight
        # (the epoch runs inside worker-0's play loop, before its own _end_inflight)
        reserve = 1 if _caller_inflight else 0
        if not self._drain_inflight(reserve=reserve):
            log("Epoch aborted: in-flight games did not finish")
            return False

        with self._lock:
            pre_mixed = self._state.mixed_wins + self._state.mixed_losses
            pre_rate = self._mixed_win_rate()
            pre_wins = self._state.mixed_wins
            pre_losses = self._state.mixed_losses
            epoch = self._state.epoch + 1

        had_prior = self._has_distinct_prior()

        log(f"--- epoch {epoch} @ {datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')} ---")

        self._consistency_flush()
        self._snapshot_previous_weights()

        trainer = _TRAINING / "titanium_training" / "training" / "trainer.py"
        if use_db_streaming:
            from streaming_db_loader import db_counts

            counts = db_counts(LABELS_DB_PATH)
            log(
                f"Training on labels.db: eligible={counts.eligible_positions:,} "
                f"labeled={counts.labeled_positions:,} (no monolithic cache required)"
            )
            cmd = [
                sys.executable, str(trainer),
                "--labels-db", str(LABELS_DB_PATH),
                "--out-dir", str(RUN_DIR),
                "--epochs", "1",
                "--batch", "512",
                "--lr", "0.0005",
                "--checkpoint-steps", "999999",
                "--val-split", "0.05",
                "--patience", "0",
                "--cpu",
                "--log-every", "100",
                "--log-interval-sec", "10",
                "--stream-max-positions", "2048",
                "--stream-old-refresh-fraction", "0.05",
                "--stream-featurize-chunk", "4096",
            ]
        else:
            from build_feature_cache import check_fingerprint

            ok, reason = check_fingerprint(CACHE_DIR)
            if not ok:
                if self._streaming_training_ready():
                    log(f"CACHE INVALID ({reason}) — falling back to database streaming")
                    return self._run_epoch(
                        rebuild_cache=rebuild_cache,
                        _caller_inflight=_caller_inflight,
                        use_db_streaming=True,
                    )
                log(f"CACHE INVALID at epoch: {reason}")
                return False
            meta = json.loads((CACHE_DIR / "meta.json").read_text(encoding="utf-8"))
            n_cache = int(meta.get("n_total", 0))
            log(f"Training on live cache: {n_cache:,} positions (incremental per-game, no rebuild)")
            if n_cache == 0:
                if self._streaming_training_ready():
                    log("CACHE EMPTY — falling back to database streaming")
                    return self._run_epoch(
                        rebuild_cache=rebuild_cache,
                        _caller_inflight=_caller_inflight,
                        use_db_streaming=True,
                    )
                log("CACHE EMPTY — skip train this epoch, resuming self-play")
                self._reset_epoch_counters()
                return False

            cmd = [
                sys.executable, str(trainer),
                "--cache-dir", str(CACHE_DIR),
                "--out-dir", str(RUN_DIR),
                "--epochs", "1",
                "--batch", "512",
                "--lr", "0.0005",
                "--checkpoint-steps", "999999",
                "--val-split", "0.05",
                "--patience", "0",
                "--cpu",
                "--log-every", "100",
                "--log-interval-sec", "10",
            ]
        if self.cfg.recent_replay_fraction > 0:
            cmd.extend([
                "--recent-replay-fraction", str(self.cfg.recent_replay_fraction),
                "--recent-window-games", str(self.cfg.recent_window_games),
            ])
        if self.cfg.no_parity:
            cmd.append("--no-parity")
        ckpts = sorted(RUN_DIR.glob("ckpt_epoch*.pt"))
        if ckpts:
            cmd.extend(["--resume", "--ckpt", str(ckpts[-1])])

        # Every process that spawns trainer.py and touches RUN_DIR / BEST /
        # ENGINE_WEIGHTS must hold this lock — training_coordinator.py (or any
        # other trainer) does the same. Busy means "skip this cycle, resume
        # self-play, retry at the next trigger" — never block the pool.
        with TrainerRunLock() as _trainer_lock:
            if _trainer_lock is None:
                log("Training lock held by another process — skipping this epoch, resuming self-play")
                self._reset_epoch_counters()
                return False

            rc = run_cmd(cmd)
            if rc != 0:
                log(f"TRAIN FAILED rc={rc}")
                return False

            ckpts = sorted(RUN_DIR.glob("ckpt_epoch*.pt"))
            if ckpts:
                try:
                    from revert_checkpoint import export_checkpoint

                    export_checkpoint(ckpts[-1], deploy_engine=False)
                    log(f"Self-play weights refreshed from {ckpts[-1].name} (engine deploy still gated)")
                except Exception as exc:
                    log(f"WARN: could not export self-play weights: {exc}")

            from titanium_training.training.guards import maybe_deploy_after_train, post_train_check

            before_best = _weights_sha16(BEST)
            before_engine = _weights_sha16(ENGINE_WEIGHTS)
            export_t0 = time.perf_counter()
            post_train_check()
            deployed, deploy_msg = maybe_deploy_after_train(force=False)
            export_elapsed = time.perf_counter() - export_t0
            after_best = _weights_sha16(BEST)
            after_engine = _weights_sha16(ENGINE_WEIGHTS)
            log(
                f"Promotion gate {export_elapsed:.1f}s  "
                f"{'PROMOTED' if deployed else 'HELD'}: {deploy_msg}  "
                f"best {before_best} -> {after_best}  engine {before_engine} -> {after_engine}"
            )

            from position_usage import status as usage_status

            if CACHE_DIR.is_dir():
                log(f"Usage after epoch: {usage_status(CACHE_DIR)}")

        grace = self.cfg.saturate_grace_epochs
        if not had_prior:
            log("Strength monitor: skipped (no distinct prior-epoch weights)")
        elif grace > 0 and epoch <= grace:
            log(f"Strength monitor: skipped (grace epochs {epoch}/{grace})")
        else:
            log(
                f"Strength monitor (mixed in completed batch): "
                f"wins={pre_wins} losses={pre_losses} "
                f"rate={pre_rate:.3f} (need >={self.cfg.saturate_threshold} over {self.cfg.saturate_min_mixed})"
            )
        if (
            not self.cfg.no_saturate
            and had_prior
            and (grace <= 0 or epoch > grace)
            and pre_mixed >= self.cfg.saturate_min_mixed
            and pre_rate < self.cfg.saturate_threshold
        ):
            report = {
                "epoch": epoch,
                "mixed_games": pre_mixed,
                "current_win_rate": pre_rate,
                "pseudo_elo": round(-400 * math.log10(1 / pre_rate - 1), 1) if 0 < pre_rate < 1 else None,
            }
            (RUN_DIR / "SATURATED.txt").write_text(json.dumps(report, indent=2), encoding="utf-8")
            return True

        if use_db_streaming:
            from streaming_db_loader import db_counts as _dbc

            gen = _dbc(LABELS_DB_PATH).eligible_positions
        elif (CACHE_DIR / "meta.json").is_file():
            meta = json.loads((CACHE_DIR / "meta.json").read_text(encoding="utf-8"))
            gen = int(meta.get("n_total", 0))
        else:
            gen = 0
        with self._lock:
            self._state.epoch = epoch
            self._state.cache_generation = gen
        self._reset_epoch_counters()
        self._save_state()
        log(f"Epoch {epoch} complete; resuming self-play")
        return False

    def _worker(self, worker_id: int) -> None:
        log(f"[thread {worker_id}] started (play)")
        while not self._stop.is_set():
            if not self._accept_games.wait(timeout=1.0):
                continue
            if self._stop.is_set() or not self._accept_games.is_set():
                continue

            self._begin_inflight()
            try:
                try:
                    r = self._play_one(worker_id)
                except Exception as exc:
                    log(f"[w{worker_id}] play error: {exc}")
                    time.sleep(2)
                    continue
                if not r or not r.get("moves"):
                    continue

                try:
                    outcome = self._persist_game(r)
                    self._persist_failures = 0
                except Exception as exc:
                    self._persist_failures += 1
                    backoff = min(60.0, 2.0 ** (self._persist_failures - 1))
                    log(
                        f"[w{worker_id}] persist error ({self._persist_failures}/"
                        f"{self._max_persist_failures}): {exc} backoff={backoff:.0f}s"
                    )
                    if self._persist_failures >= self._max_persist_failures:
                        log("Stopping pool: repeated persistence failures")
                        self._stop.set()
                        break
                    time.sleep(backoff)
                    continue

                if not outcome.counted:
                    log(f"[w{worker_id}] game {r['game_id']} not counted (sync skipped/failed)")
                    continue

                n = self._record_game(r, outcome)
                if self.cfg.opening_exploration and not r.get("mixed"):
                    update_metrics(self._opening_metrics, r)
                    if self._state.pool_games_total % self._metrics_log_every == 0:
                        metrics_path = LOG_DIR / "opening_exploration_metrics.json"
                        payload = {
                            **self._opening_metrics.to_dict(),
                            "top_prefixes": (
                                self._prefix_index.frequency_distribution(
                                    max_ply=self.cfg.opening_exploration_max_ply,
                                )
                                if self._prefix_index
                                else []
                            ),
                        }
                        metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                log(
                    f"[w{worker_id}] game {self._state.pool_games_total} "
                    f"plies={r['plies']} mixed={r['mixed']} new_pos={outcome.new_positions} "
                    f"open={r.get('opening_plies', 0)} "
                    f"explore={r.get('explored_moves', 0)} "
                    f"novel={r.get('novel_prefix', False)} "
                    f"novel_ply={r.get('novel_exit_ply')} "
                    f"cache={outcome.cache_total or self._state.cache_generation} "
                    f"epoch={self._epoch_progress_label()}"
                )

                if self._epoch_ready():
                    self._try_epoch()
            finally:
                self._end_inflight()

        log(f"[thread {worker_id}] stopped")

    def run(self) -> int:
        from prep_guard import guard_real_work

        guard_real_work("corpus_generation", detail="ContinuousPool.run")
        if not BEST.is_file():
            log(f"ERROR: missing {BEST} — run with --from-frozen first")
            return 1

        refresh_pool_weights_snapshot()
        log(f"Pool shadow weights: {pool_weights_path().name}")

        pool_ds = pool_teacher_dir()
        self._skip_cache_append = True
        log(
            "Database-first runtime: feature cache append/bootstrap disabled; "
            "games and labels commit directly to canonical DBs."
        )

        if self.cfg.opening_exploration and not OPENING_ENABLED_PATH.is_file():
            log(
                "WARN: --opening-exploration set but prefix index not activated "
                f"(missing {OPENING_ENABLED_PATH.name}); exploration disabled until rebuild completes"
            )
        if self.cfg.opening_prefix_index.is_file() or self.cfg.rebuild_prefix_index:
            self._prefix_index = OpeningPrefixIndex(self.cfg.opening_prefix_index)
            if self.cfg.rebuild_prefix_index:
                log("Rebuilding opening prefix index from games.db ...")
                boot = self._prefix_index.build_from_games_db(max_ply=self.cfg.opening_exploration_max_ply)
                log(f"Prefix index rebuild: {boot}")
            else:
                boot = self._prefix_index.ensure_bootstrapped(
                    max_ply=self.cfg.opening_exploration_max_ply,
                    auto_build=False,
                )
                log(f"Opening prefix index: {boot} total_prefixes={self._prefix_index.total_prefixes():,}")
                if boot.get("action") == "empty" and boot.get("message"):
                    log(f"NOTE: {boot['message']}")

        if self.cfg.initial_epoch and self._state.epoch == 0:
            log("=== initial epoch: train on existing feature cache before self-play ===")
            self._accept_games.clear()
            try:
                if self._run_epoch(rebuild_cache=False):
                    log("SATURATED during initial epoch — stopping.")
                    return 2
            finally:
                self._accept_games.set()

        if self.cfg.force_epoch:
            log("=== forced epoch at startup ===")
            self._accept_games.clear()
            try:
                if self._run_epoch():
                    log("SATURATED during forced epoch — stopping.")
                    return 2
            finally:
                self._accept_games.set()

        if self.cfg.oracle_url and self.cfg.oracle_token:
            from oracle_laptop_client import OracleClientConfig, OracleImportThread

            cfg = OracleClientConfig(
                base_url=self.cfg.oracle_url,
                token=self.cfg.oracle_token,
                poll_sec=self.cfg.oracle_poll_sec,
                db_lock=self._db_lock,
                teacher_lock=self._teacher_lock,
            )
            self._oracle_importer = OracleImportThread(cfg, on_import=self._record_remote_import)
            self._oracle_importer.start()
            log(f"Oracle importer started url={self.cfg.oracle_url} poll={self.cfg.oracle_poll_sec}s")

        worker_ids = (
            range(1, self.cfg.threads + 1)
            if self.cfg.reserve_worker0
            else range(self.cfg.threads)
        )
        threads = []
        for i in worker_ids:
            t = threading.Thread(target=self._worker, args=(i,), daemon=True, name=f"pool-{i}")
            t.start()
            threads.append(t)

        pool_ds = pool_teacher_dir()
        mix_pct = self._effective_same_net_pct()
        mix_label = (
            f"{mix_pct:.0%} same-net / {1 - mix_pct:.0%} vs-prior"
            if mix_pct < 1.0
            else "100% same-net (no prior epoch)"
        )
        trigger = (
            f"{self.cfg.train_after_new_positions} new positions"
            if self.cfg.use_position_trigger and self.cfg.train_after_new_positions > 0
            else f"{self.cfg.batch_games} games"
        )
        log(
            f"Continuous pool: {self.cfg.threads} threads, "
            f"epoch_trigger={trigger}, time={self.cfg.time_sec}s, nodes={self.cfg.nodes}, "
            f"epoch={self._state.epoch}, {self._epoch_progress_label()}, "
            f"mix={mix_label}, explore_chance={self.cfg.explore_chance:.2f} "
            f"explore_start_ply={self.cfg.explore_start_ply} "
            f"opening_exploration={self.cfg.opening_exploration} "
            f"opening_pct={self.cfg.opening_pct:.2f} "
            f"explore_worker0={self.cfg.explore_worker0} "
            f"reserve_worker0={self.cfg.reserve_worker0}, pool_dataset={pool_ds}"
        )

        try:
            while not self._stop.is_set():
                time.sleep(1.0)
        except KeyboardInterrupt:
            log("Interrupt — stopping workers...")
            self._stop.set()
            self._accept_games.set()

        for t in threads:
            t.join(timeout=30)
        if self._oracle_importer:
            self._oracle_importer.stop_event.set()
        self._save_state()
        if self._prefix_index is not None:
            self._prefix_index.close()
        log("Pool stopped.")
        return 2 if (RUN_DIR / "SATURATED.txt").is_file() and self._stop.is_set() else 0


def ensure_previous_from_frozen() -> None:
    if not PREVIOUS.is_file() and FROZEN.is_file():
        shutil.copy2(FROZEN, PREVIOUS)
        log(f"Initialized previous weights from frozen -> {PREVIOUS.name}")


def parse_pool_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--reserve-worker0", action="store_true",
                    help="Start play workers at id 1 so local worker 0 stays idle")
    ap.add_argument("--time", type=float, default=2.0, help="Seconds per move")
    ap.add_argument("--nodes", type=int, default=0, help="Optional node cap per move (0=disabled)")
    ap.add_argument("--batch-games", type=int, default=DEFAULT_EPOCH_GAMES,
                    help="Successfully persisted games per training epoch")
    ap.add_argument("--train-after-new-positions", type=int, default=0,
                    help="Optional position trigger (0=disabled; use --position-trigger)")
    ap.add_argument("--db-streaming", action="store_true",
                    help="Force every epoch to train on the live labels.db stream "
                         "(teacher_dataset_good anchor + sane self-play, opening-gated, IO-bounded) "
                         "instead of the monolithic feature cache.")
    ap.add_argument("--position-trigger", action="store_true",
                    help="Also train when positions_since_epoch reaches threshold")
    ap.add_argument("--recent-replay-fraction", type=float, default=0.0)
    ap.add_argument("--recent-window-games", type=int, default=128)
    ap.add_argument("--same-net-pct", type=float, default=0.7)
    ap.add_argument("--from-frozen", action="store_true")
    ap.add_argument("--current", type=Path, default=DEFAULT_CURRENT)
    ap.add_argument("--previous", type=Path, default=DEFAULT_PREVIOUS)
    ap.add_argument("--saturate-threshold", type=float, default=0.45)
    ap.add_argument("--saturate-min-mixed", type=int, default=32)
    ap.add_argument("--no-saturate", action="store_true")
    ap.add_argument("--saturate-grace-epochs", type=int, default=0)
    ap.add_argument("--no-initial-epoch", action="store_true")
    ap.add_argument("--no-parity", action="store_true")
    ap.add_argument("--force-epoch", action="store_true",
                    help="Run one training epoch immediately at startup, then self-play")
    ap.add_argument("--oracle-url", default=None,
                    help="Optional Oracle game factory URL, normally http://127.0.0.1:8765")
    ap.add_argument("--oracle-token", default=None,
                    help="Bearer token for Oracle game factory")
    ap.add_argument("--oracle-poll-sec", type=float, default=30.0)
    ap.add_argument("--no-oracle", action="store_true",
                    help="Disable Oracle importer (use oracle_importer.py separately)")
    ap.add_argument("--explore-chance", type=float, default=0.0,
                    help="Chance to pick a close evaluated alternative after --explore-start-ply")
    ap.add_argument("--explore-start-ply", type=int, default=6)
    ap.add_argument("--explore-max-loss-cp", type=int, default=140)
    ap.add_argument("--explore-candidate-count", type=int, default=18)
    ap.add_argument("--explore-top-n", type=int, default=8)
    ap.add_argument("--explore-temperature-cp", type=float, default=45.0)
    ap.add_argument("--explore-wall-bonus-cp", type=int, default=12)
    ap.add_argument("--explore-decay-after-hit", type=float, default=0.55)
    ap.add_argument("--explore-min-chance", type=float, default=0.03)
    ap.add_argument("--explore-worker0", action="store_true",
                    help="Also explore on worker 0; default keeps worker 0 deterministic")
    ap.add_argument("--opening-pct", type=float, default=0.0,
                    help="Fraction of games that start from --opening-line")
    ap.add_argument("--opening-line", default="",
                    help="Move prefix used for a small opening curriculum (comma or space separated)")
    ap.add_argument("--opening-exploration", action="store_true",
                    help="Novelty-aware opening temperature (disabled on mixed promotion games)")
    ap.add_argument("--opening-temperature-initial", type=float, default=1.0)
    ap.add_argument("--opening-temperature-after-ply4", type=float, default=1.0)
    ap.add_argument("--opening-temperature-decay-per-ply", type=float, default=0.08)
    ap.add_argument("--opening-temperature-min-while-known", type=float, default=0.45)
    ap.add_argument("--opening-exploration-max-ply", type=int, default=16)
    ap.add_argument("--novel-prefix-temperature", type=float, default=0.0)
    ap.add_argument("--opening-high-freq-boost", type=float, default=0.1)
    ap.add_argument("--opening-prob-floor", type=float, default=0.08)
    ap.add_argument("--opening-prefix-index", type=Path, default=DEFAULT_INDEX_PATH)
    ap.add_argument("--rebuild-prefix-index", action="store_true",
                    help="Rebuild opening prefix index from games.db at startup")
    ap.add_argument("--opening-explore-worker0", action="store_true",
                    help="Allow opening exploration on worker 0 (default deterministic)")
    return ap.parse_args(argv)


def build_pool_config(args: argparse.Namespace, *, no_oracle: bool = False) -> PoolConfig:
    return PoolConfig(
        threads=max(1, args.threads),
        reserve_worker0=args.reserve_worker0,
        time_sec=args.time,
        nodes=max(0, args.nodes),
        batch_games=max(32, args.batch_games),
        train_after_new_positions=max(0, args.train_after_new_positions),
        use_position_trigger=args.position_trigger,
        recent_replay_fraction=max(0.0, min(1.0, args.recent_replay_fraction)),
        recent_window_games=max(1, args.recent_window_games),
        same_net_pct=args.same_net_pct,
        current=args.current,
        previous=args.previous,
        saturate_threshold=args.saturate_threshold,
        saturate_min_mixed=args.saturate_min_mixed,
        no_saturate=args.no_saturate,
        no_parity=args.no_parity,
        saturate_grace_epochs=max(0, args.saturate_grace_epochs),
        initial_epoch=not args.no_initial_epoch,
        force_epoch=args.force_epoch,
        db_streaming=args.db_streaming,
        oracle_url=None if (no_oracle or args.no_oracle) else args.oracle_url,
        oracle_token=None if (no_oracle or args.no_oracle) else args.oracle_token,
        oracle_poll_sec=max(1.0, args.oracle_poll_sec),
        explore_chance=max(0.0, min(1.0, args.explore_chance)),
        explore_start_ply=max(0, args.explore_start_ply),
        explore_max_loss_cp=max(0, args.explore_max_loss_cp),
        explore_candidate_count=max(1, args.explore_candidate_count),
        explore_top_n=max(1, args.explore_top_n),
        explore_temperature_cp=max(1.0, args.explore_temperature_cp),
        explore_wall_bonus_cp=args.explore_wall_bonus_cp,
        explore_decay_after_hit=max(0.0, min(1.0, args.explore_decay_after_hit)),
        explore_min_chance=max(0.0, min(1.0, args.explore_min_chance)),
        explore_worker0=args.explore_worker0,
        opening_pct=max(0.0, min(1.0, args.opening_pct)),
        opening_moves=tuple(m for m in args.opening_line.replace(",", " ").split() if m),
        opening_exploration=args.opening_exploration,
        opening_temperature_initial=max(0.0, args.opening_temperature_initial),
        opening_temperature_after_ply4=max(0.0, args.opening_temperature_after_ply4),
        opening_temperature_decay_per_ply=max(0.0, args.opening_temperature_decay_per_ply),
        opening_temperature_min_while_known=max(0.0, args.opening_temperature_min_while_known),
        opening_exploration_max_ply=max(4, args.opening_exploration_max_ply),
        novel_prefix_temperature=max(0.0, args.novel_prefix_temperature),
        opening_high_freq_boost=max(0.0, args.opening_high_freq_boost),
        opening_prob_floor=max(0.0, min(1.0, args.opening_prob_floor)),
        opening_prefix_index=args.opening_prefix_index,
        rebuild_prefix_index=args.rebuild_prefix_index,
        opening_explore_worker0=args.opening_explore_worker0,
    )


def main(argv: list[str] | None = None) -> int:
    from prep_guard import guard_real_work

    guard_real_work("corpus_generation", detail="continuous_pool")
    def _on_signal(signum, _frame):
        log(f"Signal {signum} — releasing pool lock and stopping...")
        release_pool_lock()
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _on_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _on_signal)

    args = parse_pool_args(argv)

    if args.from_frozen:
        rc = run_cmd([sys.executable, str(_TRAINING / "revert_to_frozen.py")])
        if rc != 0:
            return rc
        ensure_previous_from_frozen()

    cfg = build_pool_config(args)

    pool = ContinuousPool(cfg)
    with PoolInstanceLock() as lock_info:
        log(
            f"Pool lock acquired pid={lock_info.pid} lock_id={lock_info.pid}@"
            f"{lock_info.started_at} repo={lock_info.repo}"
        )
        try:
            return pool.run()
        finally:
            log(f"Pool lock released pid={lock_info.pid}")
            release_pool_lock()


if __name__ == "__main__":
    raise SystemExit(main())
