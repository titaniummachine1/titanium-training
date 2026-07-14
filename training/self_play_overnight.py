#!/usr/bin/env python3
"""Batch self-play: 8 parallel games, current vs current (70%) or current vs previous (30%).

Each worker runs one game at a time. When a `GameSessions` pair is supplied
(see continuous_pool.py), each side keeps one warm, persistent engine process
for the whole game (TT/dist-LRU/killers/history stay hot across plies) instead
of spawning a fresh cold process every ply. Without a session pair (tests,
one-off tools), falls back to the original one-subprocess-per-move path.
Labels: +1 win / -1 loss from side-to-move perspective (stored via db_import).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parent.parent
_TRAINING = _REPO / "training"
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

from titanium_training.paths import ENGINE_BIN, REPO_ROOT
from titanium_training.store.state import PositionState, apply_move
from db_import import GAMES_DB_PATH, LABELS_DB_PATH, GAMES_SCHEMA, LABELS_SCHEMA, open_db, write_batch
from engine_session import EngineSession


@dataclass
class GameSessions:
    """One warm engine process per side, reused for every ply of one game."""

    p0: EngineSession
    p1: EngineSession

    def for_side(self, is_p0: bool) -> EngineSession:
        return self.p0 if is_p0 else self.p1

    def close(self) -> None:
        for s in (self.p0, self.p1):
            try:
                s.close()
            except Exception:
                pass

MAX_PLIES = 128
REPETITION_DRAW_COUNT = 6
ENGINE_NAME = os.environ.get("TITANIUM_GENERATION_ENGINE", "titanium-v17").strip() or "titanium-v17"
DEFAULT_CURRENT = _TRAINING / "runs" / "v16" / "net_weights_best.bin"
DEFAULT_PREVIOUS = _TRAINING / "runs" / "v16" / "net_weights_previous.bin"
DEFAULT_FROZEN = _REPO / "engine" / "src" / "titanium" / "net_weights_frozen.bin"


@dataclass(frozen=True)
class ExplorationConfig:
    start_ply: int = 6
    chance: float = 0.0
    max_loss_cp: int = 140
    candidate_count: int = 18
    top_n: int = 8
    temperature_cp: float = 45.0
    wall_bonus_cp: int = 12
    decay_after_hit: float = 0.55
    min_chance: float = 0.03

    @property
    def enabled(self) -> bool:
        return self.chance > 0.0 and self.candidate_count > 0 and self.top_n > 1


OPENING_TEMPERATURE_DECAY = 0.95
OPENING_EXPLORATION_START_PLY = 5


def opening_temperature_for_move(
    ply_number: int,
    novelty_reached: bool,
    prefix_known: bool,
) -> tuple[float, bool]:
    """Shared opening temperature for local and Oracle current-vs-current games."""
    if ply_number < OPENING_EXPLORATION_START_PLY:
        return 0.0, novelty_reached
    if novelty_reached:
        return 0.0, novelty_reached
    if prefix_known:
        return OPENING_TEMPERATURE_DECAY ** (ply_number - OPENING_EXPLORATION_START_PLY), novelty_reached
    return 0.0, True


@dataclass(frozen=True)
class OpeningExplorationConfig:
    """Novelty-aware opening temperature schedule (disabled when mixed promotion games)."""

    enabled: bool = False
    initial_temperature: float = 1.0
    after_ply4_temperature: float = 1.0
    decay_per_ply: float = 0.08
    min_while_known: float = 0.45
    max_exploration_ply: int = 16
    novel_prefix_temperature: float = 0.0
    high_freq_boost: float = 0.1
    max_loss_cp: int = 140
    candidate_count: int = 18
    top_n: int = 8
    wall_bonus_cp: int = 12
    prob_floor: float = 0.08

    def temperature_for_move(
        self,
        ply: int,
        *,
        exited_novel_tree: bool,
        prefix_known: bool,
        prefix_count: int = 0,
    ) -> float:
        if not self.enabled:
            return 0.0
        if ply < OPENING_EXPLORATION_START_PLY or ply > self.max_exploration_ply:
            return 0.0
        if exited_novel_tree:
            return self.novel_prefix_temperature
        if not prefix_known:
            return 0.0
        if ply == OPENING_EXPLORATION_START_PLY:
            temp = self.initial_temperature
        else:
            decay_steps = max(0, ply - OPENING_EXPLORATION_START_PLY - 1)
            temp = self.after_ply4_temperature - self.decay_per_ply * decay_steps
            temp = max(self.min_while_known, temp)
        if prefix_count > 1 and self.high_freq_boost > 0.0:
            boost_steps = min(5, int(prefix_count).bit_length() - 1)
            temp += self.high_freq_boost * boost_steps
        return max(0.0, temp)


def check_winner(moves: list[str]) -> int | None:
    if not moves:
        return None
    last = moves[-1]
    if last[-1] in ("h", "v"):
        return None
    row = last[-1]
    mover = (len(moves) - 1) % 2
    if mover == 0 and row == "9":
        return 0
    if mover == 1 and row == "1":
        return 1
    return None


def _state_repetition_key(state: PositionState) -> bytes:
    return state.packed_state()


def _replay_opening_state(moves: list[str]) -> tuple[PositionState, dict[bytes, int]]:
    state = PositionState.initial()
    counts = {_state_repetition_key(state): 1}
    for mv in moves:
        state = apply_move(state, mv)
        key = _state_repetition_key(state)
        counts[key] = counts.get(key, 0) + 1
    return state, counts


def _engine_env(weights: Path | None) -> dict[str, str]:
    env = os.environ.copy()
    if weights and weights.is_file():
        env["TITANIUM_NET_WEIGHTS_PATH"] = str(weights.resolve())
    else:
        env.pop("TITANIUM_NET_WEIGHTS_PATH", None)
    return env


def engine_move(moves: list[str], time_sec: float, weights: Path | None) -> str | None:
    return engine_move_budget(moves, time_sec, weights, nodes=None)


def engine_move_budget(
    moves: list[str],
    time_sec: float,
    weights: Path | None,
    nodes: int | None = None,
    temperature: float = 0.0,
    engine: str | None = None,
) -> str | None:
    env = _engine_env(weights)
    if temperature > 0.0:
        env["TITANIUM_OPENING_TEMPERATURE"] = str(temperature)
    cmd = [str(ENGINE_BIN), "genmove", "--engine", engine or ENGINE_NAME]
    cmd += moves + ["--time", str(time_sec)]
    if nodes and nodes > 0:
        cmd += ["--nodes", str(nodes)]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, env=env, cwd=str(REPO_ROOT),
            timeout=max(time_sec * 6 + 15, 30),
        )
    except subprocess.TimeoutExpired:
        return None
    for line in reversed(proc.stdout.decode(errors="replace").splitlines()):
        line = line.strip()
        if line.startswith("bestmove "):
            tok = line.split()[1]
            if tok not in ("(none)",):
                return tok
    return None


def legal_moves(moves: list[str]) -> list[str]:
    cmd = [str(ENGINE_BIN), "moves", *moves]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            cwd=str(REPO_ROOT),
            timeout=20,
            text=True,
        )
    except subprocess.TimeoutExpired:
        return []
    if proc.returncode != 0:
        return []
    return [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]


def child_eval_scores(
    moves: list[str],
    candidates: list[str],
    weights: Path | None,
) -> dict[str, int]:
    if not candidates:
        return {}
    env = _engine_env(weights)
    payload = "\n".join(" ".join([*moves, mv]) for mv in candidates) + "\n"
    try:
        proc = subprocess.run(
            [str(ENGINE_BIN), "eval-batch", "--score-only"],
            input=payload,
            capture_output=True,
            env=env,
            cwd=str(REPO_ROOT),
            timeout=30,
            text=True,
        )
    except subprocess.TimeoutExpired:
        return {}
    if proc.returncode != 0:
        return {}
    scores: dict[str, int] = {}
    for mv, line in zip(candidates, proc.stdout.splitlines()):
        try:
            # eval is from the child side-to-move; negate to score the mover.
            scores[mv] = -int(json.loads(line)["eval"])
        except Exception:
            continue
    return scores


def choose_temperature_move(
    moves: list[str],
    best: str,
    weights: Path | None,
    temperature: float,
    *,
    max_loss_cp: int,
    candidate_count: int,
    top_n: int,
    wall_bonus_cp: int,
    prob_floor: float,
    rng: random.Random,
) -> tuple[str, bool, bool]:
    """Sample from scored candidates; never uniform over all legal moves.

    Returns (move, was_exploratory, rejected_by_quality_cutoff).
    """
    if temperature <= 0.0:
        return best, False, False

    legal = legal_moves(moves)
    if best not in legal:
        return best, False, False
    others = [mv for mv in legal if mv != best]
    rng.shuffle(others)
    candidates = [best, *others[: max(0, candidate_count - 1)]]
    scores = child_eval_scores(moves, candidates, weights)
    if best not in scores:
        return best, False, False

    ranked: list[tuple[str, int]] = []
    for mv, score in scores.items():
        adjusted = score + (wall_bonus_cp if mv.endswith(("h", "v")) else 0)
        ranked.append((mv, adjusted))
    ranked.sort(key=lambda x: x[1], reverse=True)

    best_score = ranked[0][1]
    allowed = [
        (mv, score)
        for mv, score in ranked[: max(1, top_n)]
        if best_score - score <= max_loss_cp
    ]
    if not allowed:
        return best, False, True
    if best not in {mv for mv, _score in allowed}:
        allowed.append((best, scores[best]))

    temp = max(1.0, temperature * 45.0)
    weights_f = [pow(2.718281828, (score - best_score) / temp) for _mv, score in allowed]
    total = sum(weights_f)
    if total <= 0:
        return best, False, False
    floor = max(0.0, prob_floor)
    if floor > 0:
        min_w = total * floor
        weights_f = [max(w, min_w) for w in weights_f]
        total = sum(weights_f)
    pick = rng.random() * total
    acc = 0.0
    for (mv, _score), w in zip(allowed, weights_f):
        acc += w
        if pick <= acc:
            return mv, mv != best, False
    chosen = allowed[-1][0]
    return chosen, chosen != best, False


def choose_exploration_move(
    moves: list[str],
    best: str,
    weights: Path | None,
    cfg: ExplorationConfig,
    rng: random.Random,
) -> str:
    ply = len(moves)
    if not cfg.enabled or ply < cfg.start_ply or rng.random() >= cfg.chance:
        return best

    chosen, _explored, _rejected = choose_temperature_move(
        moves,
        best,
        weights,
        cfg.temperature_cp / 45.0,
        max_loss_cp=cfg.max_loss_cp,
        candidate_count=cfg.candidate_count,
        top_n=cfg.top_n,
        wall_bonus_cp=cfg.wall_bonus_cp,
        prob_floor=0.0,
        rng=rng,
    )
    return chosen


def play_one_game(
    game_id: str,
    time_sec: float,
    w_p0: Path | None,
    w_p1: Path | None,
    mixed: bool,
    current_is_p0: bool,
    exploration: ExplorationConfig | None = None,
    opening_exploration: OpeningExplorationConfig | None = None,
    prefix_index: Any | None = None,
    rng: random.Random | None = None,
    nodes: int | None = None,
    opening: list[str] | None = None,
    game_seed: int | None = None,
    engine_hash: str | None = None,
    weights_hash: str | None = None,
    engine_p0: str | None = None,
    engine_p1: str | None = None,
    matchup_kind: str | None = None,
    opponent_engine: str | None = None,
    sessions: GameSessions | None = None,
) -> dict:
    moves: list[str] = list(opening or [])
    state, repetition_counts = _replay_opening_state(moves)
    explore_cfg = exploration or ExplorationConfig()
    opening_cfg = opening_exploration or OpeningExplorationConfig()
    explore_rng = rng or random.Random(game_seed)
    explored = 0
    quality_rejects = 0
    exited_novel_tree = False
    novel_exit_ply: int | None = None
    move_temperatures: list[float] = []
    prefix_counts_at_explore: list[int] = []
    use_opening = opening_cfg.enabled and prefix_index is not None

    eng_p0 = engine_p0 or ENGINE_NAME
    eng_p1 = engine_p1 or ENGINE_NAME

    for ply in range(len(moves), MAX_PLIES):
        is_p0 = (ply % 2 == 0)
        w = w_p0 if is_p0 else w_p1
        eng = eng_p0 if is_p0 else eng_p1
        ply_num = len(moves) + 1
        prefix_known = prefix_index.is_known(moves) if prefix_index is not None else True
        prefix_count = prefix_index.occurrence_count(moves) if prefix_index is not None else 0
        engine_temp = 0.0
        if use_opening:
            engine_temp = opening_cfg.temperature_for_move(
                ply_num,
                exited_novel_tree=exited_novel_tree,
                prefix_known=prefix_known,
                prefix_count=prefix_count,
            )
            if not prefix_known:
                exited_novel_tree = True
            if engine_temp > 0.0:
                explored += 1
                move_temperatures.append(engine_temp)
                prefix_counts_at_explore.append(prefix_count)

        if sessions is not None:
            session = sessions.for_side(is_p0)
            if not session.sync(moves) or not session.alive():
                mv = None
            else:
                mv = session.go(time_sec)
        else:
            mv = engine_move_budget(moves, time_sec, w, nodes=nodes, temperature=engine_temp, engine=eng)
        if not mv:
            break
        if use_opening and engine_temp > 0.0:
            chosen, was_exploratory, rejected = choose_temperature_move(
                moves,
                mv,
                w,
                engine_temp,
                max_loss_cp=opening_cfg.max_loss_cp,
                candidate_count=opening_cfg.candidate_count,
                top_n=opening_cfg.top_n,
                wall_bonus_cp=opening_cfg.wall_bonus_cp,
                prob_floor=opening_cfg.prob_floor,
                rng=explore_rng,
            )
            if rejected:
                quality_rejects += 1
            if was_exploratory:
                mv = chosen

        if explored > 0 and explore_cfg.enabled and not use_opening:
            cooled_chance = max(
                explore_cfg.min_chance,
                explore_cfg.chance * (explore_cfg.decay_after_hit ** explored),
            )
            move_cfg = ExplorationConfig(
                start_ply=explore_cfg.start_ply,
                chance=cooled_chance,
                max_loss_cp=explore_cfg.max_loss_cp,
                candidate_count=explore_cfg.candidate_count,
                top_n=explore_cfg.top_n,
                temperature_cp=explore_cfg.temperature_cp,
                wall_bonus_cp=explore_cfg.wall_bonus_cp,
                decay_after_hit=explore_cfg.decay_after_hit,
                min_chance=explore_cfg.min_chance,
            )
            chosen = choose_exploration_move(moves, mv, w, move_cfg, explore_rng)
            if chosen != mv:
                explored += 1
                mv = chosen
        elif explore_cfg.enabled:
            chosen = choose_exploration_move(moves, mv, w, explore_cfg, explore_rng)
            if chosen != mv:
                explored += 1
                mv = chosen

        moves.append(mv)
        if state is not None:
            try:
                state = apply_move(state, mv)
            except ValueError:
                state = None
                repetition_counts = {}

        if use_opening and not exited_novel_tree and prefix_index is not None:
            if not prefix_index.is_known(moves):
                exited_novel_tree = True
                novel_exit_ply = ply_num

        winner = check_winner(moves)
        if winner is not None:
            outcome_p0 = 1 if winner == 0 else -1
            return {
                "game_id": game_id,
                "moves": moves,
                "outcome_p0": outcome_p0,
                "mixed": mixed,
                "current_is_p0": current_is_p0,
                "current_won": (winner == 0) == current_is_p0 if mixed else None,
                "aborted": False,
                "draw_reason": None,
                "plies": len(moves),
                "explored_moves": explored,
                "opening_plies": len(opening or []),
                "novel_prefix": bool(novel_exit_ply),
                "novel_exit_ply": novel_exit_ply,
                "exited_novel_tree": exited_novel_tree,
                "move_temperatures": move_temperatures,
                "prefix_counts_at_explore": prefix_counts_at_explore,
                "exploration_quality_rejects": quality_rejects,
                "game_seed": game_seed,
                "engine_hash": engine_hash,
                "weights_hash": weights_hash,
                "matchup_kind": matchup_kind,
                "opponent_engine": opponent_engine,
                "engine_p0": eng_p0,
                "engine_p1": eng_p1,
            }
        if state is None:
            continue
        repeat_key = _state_repetition_key(state)
        repetition_counts[repeat_key] = repetition_counts.get(repeat_key, 0) + 1
        if repetition_counts[repeat_key] >= REPETITION_DRAW_COUNT:
            return {
                "game_id": game_id,
                "moves": moves,
                "outcome_p0": 0,
                "mixed": mixed,
                "current_is_p0": current_is_p0,
                "current_won": None,
                "aborted": False,
                "draw_reason": "repetition",
                "plies": len(moves),
                "explored_moves": explored,
                "opening_plies": len(opening or []),
                "novel_prefix": bool(novel_exit_ply),
                "novel_exit_ply": novel_exit_ply,
                "exited_novel_tree": exited_novel_tree,
                "move_temperatures": move_temperatures,
                "prefix_counts_at_explore": prefix_counts_at_explore,
                "exploration_quality_rejects": quality_rejects,
                "game_seed": game_seed,
                "engine_hash": engine_hash,
                "weights_hash": weights_hash,
            }
    if len(moves) >= MAX_PLIES:
        return {
            "game_id": game_id,
            "moves": moves,
            "outcome_p0": 0,
            "mixed": mixed,
            "current_is_p0": current_is_p0,
            "current_won": None,
            "aborted": False,
            "draw_reason": "max_plies",
            "plies": len(moves),
            "explored_moves": explored,
            "opening_plies": len(opening or []),
            "novel_prefix": bool(novel_exit_ply),
            "novel_exit_ply": novel_exit_ply,
            "exited_novel_tree": exited_novel_tree,
            "move_temperatures": move_temperatures,
            "prefix_counts_at_explore": prefix_counts_at_explore,
            "exploration_quality_rejects": quality_rejects,
            "game_seed": game_seed,
            "engine_hash": engine_hash,
            "weights_hash": weights_hash,
        }
    return {
        "game_id": game_id,
        "moves": moves,
        "outcome_p0": None,
        "mixed": mixed,
        "current_is_p0": current_is_p0,
        "current_won": None,
        "aborted": True,
        "abort_reason": "max_plies_or_no_move",
        "plies": len(moves),
        "explored_moves": explored,
        "opening_plies": len(opening or []),
        "novel_prefix": bool(novel_exit_ply),
        "novel_exit_ply": novel_exit_ply,
        "exited_novel_tree": exited_novel_tree,
        "move_temperatures": move_temperatures,
        "prefix_counts_at_explore": prefix_counts_at_explore,
        "exploration_quality_rejects": quality_rejects,
        "game_seed": game_seed,
        "engine_hash": engine_hash,
        "weights_hash": weights_hash,
    }


def _worker(args: tuple) -> dict:
    seed, game_idx, time_sec, current, previous, p_same_net = args
    rng = random.Random(seed + game_idx)
    mixed = rng.random() >= p_same_net  # p_same_net=0.7 -> 30% current vs previous
    if not mixed:
        w_p0 = w_p1 = current
        current_is_p0 = True
    else:
        current_is_p0 = rng.random() < 0.5
        if current_is_p0:
            w_p0, w_p1 = current, previous
        else:
            w_p0, w_p1 = previous, current
    gid = f"overnight_{int(time.time())}_{game_idx:05d}"
    return play_one_game(gid, time_sec, w_p0, w_p1, mixed, current_is_p0)


@dataclass
class SelfPlayStats:
    games: int = 0
    mixed_games: int = 0
    current_wins: int = 0
    current_losses: int = 0
    draws: int = 0

    @property
    def current_win_rate(self) -> float:
        d = self.current_wins + self.current_losses
        return self.current_wins / d if d else 0.5

    def saturated(self, min_games: int = 32, threshold: float = 0.45) -> bool:
        if self.mixed_games < min_games:
            return False
        return self.current_win_rate < threshold


def run_batch(
    *,
    n_games: int,
    threads: int,
    time_sec: float,
    current: Path,
    previous: Path,
    p_same_net: float = 0.7,
    write_db: bool = True,
    seed: int = 0,
    games_db_path: Path = GAMES_DB_PATH,
    labels_db_path: Path = LABELS_DB_PATH,
) -> tuple[SelfPlayStats, list[dict]]:
    if not ENGINE_BIN.is_file():
        raise FileNotFoundError(f"engine missing: {ENGINE_BIN}")
    if not current.is_file():
        raise FileNotFoundError(f"current weights missing: {current}")

    prev = previous if previous.is_file() else current
    tasks = [(seed, i, time_sec, current, prev, p_same_net) for i in range(n_games)]

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=threads) as pool:
        results = pool.map(_worker, tasks)

    stats = SelfPlayStats()
    db_batch = []
    for r in results:
        stats.games += 1
        if r.get("aborted"):
            continue
        if r["mixed"]:
            stats.mixed_games += 1
            if r["current_won"] is True:
                stats.current_wins += 1
            elif r["current_won"] is False:
                stats.current_losses += 1
            else:
                stats.draws += 1
        if r["moves"] and write_db:
            src = "overnight_mixed" if r["mixed"] else "overnight_selfplay"
            db_batch.append((r["game_id"], r["moves"], r["outcome_p0"], None, src))

    if write_db and db_batch:
        games_db = open_db(games_db_path, GAMES_SCHEMA)
        labels_db = open_db(labels_db_path, LABELS_SCHEMA)
        written_games, _written_positions, _written_labels = write_batch(
            games_db,
            labels_db,
            db_batch,
            chunk_size=64,
            workers=1,
        )
        games_db.close()
        labels_db.close()
        if written_games != len(db_batch):
            print(
                f"  engine eval-batch accepted {written_games}/{len(db_batch)} generated games",
                flush=True,
            )

    return stats, results


def run_batch_streaming(
    *,
    n_games: int,
    threads: int,
    time_sec: float,
    current: Path,
    previous: Path,
    p_same_net: float = 0.7,
    seed: int = 0,
    games_db_path: Path = GAMES_DB_PATH,
    labels_db_path: Path = LABELS_DB_PATH,
) -> tuple[SelfPlayStats, list[dict]]:
    """Like run_batch but commits each finished game to games.db immediately."""
    if not ENGINE_BIN.is_file():
        raise FileNotFoundError(f"engine missing: {ENGINE_BIN}")
    if not current.is_file():
        raise FileNotFoundError(f"current weights missing: {current}")

    prev = previous if previous.is_file() else (DEFAULT_FROZEN if DEFAULT_FROZEN.is_file() else current)
    tasks = [(seed, i, time_sec, current, prev, p_same_net) for i in range(n_games)]

    games_db = open_db(games_db_path, GAMES_SCHEMA)
    labels_db = open_db(labels_db_path, LABELS_SCHEMA)
    stats = SelfPlayStats()
    results: list[dict] = []

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=threads) as pool:
        for r in pool.imap_unordered(_worker, tasks):
            results.append(r)
            stats.games += 1
            if r.get("aborted"):
                print(
                    f"  game {stats.games}/{n_games} aborted "
                    f"plies={r['plies']} mixed={r['mixed']} reason={r.get('abort_reason')}",
                    flush=True,
                )
                continue
            if r["mixed"]:
                stats.mixed_games += 1
                if r["current_won"] is True:
                    stats.current_wins += 1
                elif r["current_won"] is False:
                    stats.current_losses += 1
                else:
                    stats.draws += 1
            if r["moves"]:
                src = "overnight_mixed" if r["mixed"] else "overnight_selfplay"
                written_games, _written_positions, _written_labels = write_batch(
                    games_db, labels_db,
                    [(r["game_id"], r["moves"], r["outcome_p0"], None, src)],
                    chunk_size=512,
                    workers=1,
                )
                if written_games > 0:
                    print(f"  game {stats.games}/{n_games}  plies={r['plies']}  mixed={r['mixed']}", flush=True)
                else:
                    print(
                        f"  game {stats.games}/{n_games} rejected by engine eval-batch "
                        f"plies={r['plies']} mixed={r['mixed']}",
                        flush=True,
                    )

    games_db.close()
    labels_db.close()
    return stats, results


def main() -> int:
    from prep_guard import guard_real_work

    guard_real_work("corpus_generation", detail="self_play_overnight.py")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=512)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--time", type=float, default=4.0)
    ap.add_argument("--current", type=Path, default=DEFAULT_CURRENT)
    ap.add_argument("--previous", type=Path, default=DEFAULT_PREVIOUS)
    ap.add_argument("--same-net-pct", type=float, default=0.7)
    ap.add_argument("--stream-db", action="store_true", help="Write each game to DB as it finishes")
    ap.add_argument("--no-db", action="store_true")
    ap.add_argument("--games-db", type=Path, default=GAMES_DB_PATH,
                    help="SQLite games DB to write (default: canonical games.db)")
    ap.add_argument("--labels-db", type=Path, default=LABELS_DB_PATH,
                    help="SQLite labels DB to write (default: canonical labels.db)")
    ap.add_argument("--out", type=Path, default=_TRAINING / "data" / "overnight_selfplay_last.json")
    args = ap.parse_args()

    if args.stream_db:
        stats, results = run_batch_streaming(
            n_games=args.games,
            threads=args.threads,
            time_sec=args.time,
            current=args.current,
            previous=args.previous,
            p_same_net=args.same_net_pct,
            games_db_path=args.games_db,
            labels_db_path=args.labels_db,
        )
    else:
        stats, results = run_batch(
            n_games=args.games,
            threads=args.threads,
            time_sec=args.time,
            current=args.current,
            previous=args.previous,
            p_same_net=args.same_net_pct,
            write_db=not args.no_db,
            games_db_path=args.games_db,
            labels_db_path=args.labels_db,
        )
    report = {
        "games": stats.games,
        "mixed_games": stats.mixed_games,
        "current_wins": stats.current_wins,
        "current_losses": stats.current_losses,
        "current_win_rate": stats.current_win_rate,
        "saturated": stats.saturated(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
