"""Single-game Linux worker helpers."""
from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_TRAINING = Path(__file__).resolve().parents[1]
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

from . import GAME_SCHEMA_VERSION, PROTOCOL_VERSION
from .matchup import MatchupChoice, choose_matchup, matchup_to_payload_fields
from .protocol import game_payload_checksum, sha256_file, utc_now, validate_game_payload

MAX_PLIES = 128
REPETITION_DRAW_COUNT = 6
ENGINE_NAME = "titanium-v17"
OPENING_TEMPERATURE_DECAY = 0.95
OPENING_EXPLORATION_START_PLY = 5
# Fallback window when opening_prefix_index isn't available (see
# _opening_prefix_index): without the DB-backed novelty tracking there is no
# "prefix_known" signal, so this applies the same decaying temperature for a
# fixed early window instead of the adaptive cutoff. Games between the same
# frozen weights are otherwise fully deterministic (temperature stays 0.0 for
# the whole game with no fallback) - confirmed live: 100+ Oracle games with
# zero exploration produced ZERO new eligible positions, since every game
# replayed an identical line.
FALLBACK_OPENING_EXPLORATION_MAX_PLY = 16
_PREFIX_INDEX = None


def opening_temperature_for_move(
    ply_number: int,
    novelty_reached: bool,
    prefix_known: bool,
) -> tuple[float, bool]:
    if ply_number < OPENING_EXPLORATION_START_PLY:
        return 0.0, novelty_reached
    if novelty_reached:
        return 0.0, novelty_reached
    if prefix_known:
        return OPENING_TEMPERATURE_DECAY ** (ply_number - OPENING_EXPLORATION_START_PLY), novelty_reached
    return 0.0, True


@dataclass(frozen=True)
class OraclePositionState:
    p0_cell: int = 4
    p1_cell: int = 76
    p0_walls: int = 10
    p1_walls: int = 10
    horizontal_walls: int = 0
    vertical_walls: int = 0
    side_to_move: int = 0

    def packed_state(self) -> bytes:
        head = bytes([1, self.p0_cell, self.p1_cell, self.p0_walls, self.p1_walls, self.side_to_move, 0, 0])
        return (
            head
            + int(self.horizontal_walls).to_bytes(8, "little", signed=False)
            + int(self.vertical_walls).to_bytes(8, "little", signed=False)
        )


def _cell_from_notation(move: str) -> int:
    return (int(move[1]) - 1) * 9 + (ord(move[0]) - ord("a"))


def _wall_slot_from_notation(move: str) -> int:
    return (int(move[1]) - 1) * 8 + (ord(move[0]) - ord("a"))


def _apply_move(state: OraclePositionState, move: str) -> OraclePositionState:
    next_side = 1 - state.side_to_move
    if move.endswith(("h", "v")):
        slot = _wall_slot_from_notation(move)
        bit = 1 << slot
        h_walls = state.horizontal_walls | bit if move.endswith("h") else state.horizontal_walls
        v_walls = state.vertical_walls | bit if move.endswith("v") else state.vertical_walls
        if state.side_to_move == 0:
            return OraclePositionState(
                state.p0_cell,
                state.p1_cell,
                state.p0_walls - 1,
                state.p1_walls,
                h_walls,
                v_walls,
                next_side,
            )
        return OraclePositionState(
            state.p0_cell,
            state.p1_cell,
            state.p0_walls,
            state.p1_walls - 1,
            h_walls,
            v_walls,
            next_side,
        )

    cell = _cell_from_notation(move)
    if state.side_to_move == 0:
        return OraclePositionState(
            cell,
            state.p1_cell,
            state.p0_walls,
            state.p1_walls,
            state.horizontal_walls,
            state.vertical_walls,
            next_side,
        )
    return OraclePositionState(
        state.p0_cell,
        cell,
        state.p0_walls,
        state.p1_walls,
        state.horizontal_walls,
        state.vertical_walls,
        next_side,
    )


def _opening_prefix_index(cfg: RuntimeConfig):
    global _PREFIX_INDEX
    if _PREFIX_INDEX is not None:
        return _PREFIX_INDEX
    try:
        from opening_prefix_index import OpeningPrefixIndex
    except ImportError:
        # Sibling module (training/opening_prefix_index.py + its own
        # db_import.py/titanium_training dependency) isn't deployed on every
        # host that runs this worker (e.g. Oracle only ships oracle_game_factory/).
        # Opening exploration is a training-diversity nicety, not required for
        # game generation - degrade to "no index" exactly like the missing-DB
        # case below, instead of crashing every single game on hosts without it.
        return None

    for candidate in (
        cfg.data_dir / "canonical" / "opening_prefix_index.db",
        Path("/opt/titanium-game-factory/canonical/opening_prefix_index.db"),
    ):
        if candidate.is_file():
            _PREFIX_INDEX = OpeningPrefixIndex(candidate)
            return _PREFIX_INDEX
    return None


@dataclass
class RuntimeConfig:
    engine_bin: Path
    data_dir: Path
    move_time: float = 5.0
    max_plies: int = MAX_PLIES
    node_budget: int = 200_000


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


def resolve_weight_path(data_dir: Path, weight_hash: str, generation_dir: Path) -> Path:
    """Load promoted weights from Oracle-local hash storage (no network)."""
    h = weight_hash.lower()
    for candidate in (
        data_dir / "weights" / f"{h}.bin",
        Path("/opt/titanium-game-factory/weights") / f"{h}.bin",
        generation_dir / "current.bin" if h == weight_hash else generation_dir / "prior.bin",
    ):
        if candidate.is_file() and sha256_file(candidate) == h:
            return candidate
    by_name = generation_dir / ("current.bin" if h else "prior.bin")
    if by_name.is_file():
        return by_name
    raise FileNotFoundError(f"weight hash not found locally: {h}")


def _parse_engine_info(stdout: str) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line.startswith("info json "):
            continue
        try:
            payload = json.loads(line[len("info json ") :])
        except json.JSONDecodeError:
            continue
        stats["nodes"] = payload.get("nodes")
        stats["depth"] = payload.get("searchDepth")
        stats["elapsed_ms"] = payload.get("elapsedMs")
        stats["stopped_by"] = payload.get("stoppedBy")
        stats["engine"] = payload.get("engine")
        if stats.get("nodes") and stats.get("elapsed_ms"):
            stats["nps"] = float(stats["nodes"]) / max(float(stats["elapsed_ms"]) / 1000.0, 1e-6)
        stats["timeout"] = stats.get("stopped_by") == "time"
        # Root move list from the SAME search call (no extra engine
        # invocation needed) - used for temperature-based opening move
        # selection instead of always taking the raw top move.
        root_moves = payload.get("rootMoves")
        if isinstance(root_moves, list):
            stats["root_moves"] = [
                (rm["move"], int(rm["score"]))
                for rm in root_moves
                if isinstance(rm, dict) and "move" in rm and "score" in rm
            ]
        break
    return stats


def choose_root_move_by_temperature(
    root_moves: list[tuple[str, int]],
    best: str,
    temperature: float,
    *,
    top_n: int,
    rng: random.Random,
) -> tuple[str, bool]:
    """Sample the played move from the search's own root-move scores
    (already computed by the one real search call - no extra engine
    invocations). Returns (move, was_exploratory).
    """
    if temperature <= 0.0 or not root_moves:
        return best, False
    ranked = sorted(root_moves, key=lambda mv_score: mv_score[1], reverse=True)[: max(1, top_n)]
    if not any(mv == best for mv, _score in ranked):
        ranked.append((best, next((s for mv, s in root_moves if mv == best), ranked[-1][1])))
    best_score = ranked[0][1]
    temp = max(1.0, temperature * 45.0)
    weights_f = [pow(2.718281828, (score - best_score) / temp) for _mv, score in ranked]
    total = sum(weights_f)
    if total <= 0:
        return best, False
    pick = rng.random() * total
    acc = 0.0
    for (mv, _score), w in zip(ranked, weights_f):
        acc += w
        if pick <= acc:
            return mv, mv != best
    chosen = ranked[-1][0]
    return chosen, chosen != best


def legal_moves(engine_bin: Path, moves: list[str]) -> list[str]:
    cmd = [str(engine_bin), "moves", *moves]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(engine_bin.resolve().parent.parent.parent if engine_bin.is_file() else Path.cwd()),
            timeout=20,
        )
    except subprocess.TimeoutExpired:
        return []
    if proc.returncode != 0:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def child_eval_scores(
    engine_bin: Path,
    moves: list[str],
    candidates: list[str],
    weights: Path | None,
) -> dict[str, int]:
    if not candidates:
        return {}
    env = os.environ.copy()
    if weights and weights.is_file():
        env["TITANIUM_NET_WEIGHTS_PATH"] = str(weights.resolve())
    payload = "\n".join(" ".join([*moves, mv]) for mv in candidates) + "\n"
    try:
        proc = subprocess.run(
            [str(engine_bin), "eval-batch", "--score-only"],
            input=payload,
            capture_output=True,
            text=True,
            cwd=str(engine_bin.resolve().parent.parent.parent if engine_bin.is_file() else Path.cwd()),
            timeout=30,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return {}
    if proc.returncode != 0:
        return {}
    scores: dict[str, int] = {}
    for mv, line in zip(candidates, proc.stdout.splitlines()):
        try:
            scores[mv] = -int(json.loads(line)["eval"])
        except Exception:
            continue
    return scores


def choose_eval_move_by_temperature(
    engine_bin: Path,
    moves: list[str],
    best: str,
    weights: Path | None,
    temperature: float,
    *,
    top_n: int,
    rng: random.Random,
    candidate_count: int = 18,
    max_loss_cp: int = 140,
    wall_bonus_cp: int = 12,
    prob_floor: float = 0.08,
) -> tuple[str, bool]:
    if temperature <= 0.0:
        return best, False
    legal = legal_moves(engine_bin, moves)
    if best not in legal:
        return best, False
    others = [mv for mv in legal if mv != best]
    rng.shuffle(others)
    candidates = [best, *others[: max(0, candidate_count - 1)]]
    scores = child_eval_scores(engine_bin, moves, candidates, weights)
    if best not in scores:
        return best, False

    ranked: list[tuple[str, int]] = []
    for mv, score in scores.items():
        ranked.append((mv, score + (wall_bonus_cp if mv.endswith(("h", "v")) else 0)))
    ranked.sort(key=lambda mv_score: mv_score[1], reverse=True)
    best_score = ranked[0][1]
    allowed = [
        (mv, score)
        for mv, score in ranked[: max(1, top_n)]
        if best_score - score <= max_loss_cp
    ]
    if not allowed:
        return best, False
    if best not in {mv for mv, _score in allowed}:
        allowed.append((best, scores[best]))

    temp = max(1.0, temperature * 45.0)
    weights_f = [pow(2.718281828, (score - best_score) / temp) for _mv, score in allowed]
    total = sum(weights_f)
    if total <= 0:
        return best, False
    if prob_floor > 0.0:
        min_w = total * prob_floor
        weights_f = [max(w, min_w) for w in weights_f]
        total = sum(weights_f)
    pick = rng.random() * total
    acc = 0.0
    for (mv, _score), w in zip(allowed, weights_f):
        acc += w
        if pick <= acc:
            return mv, mv != best
    chosen = allowed[-1][0]
    return chosen, chosen != best


def engine_move(
    engine_bin: Path,
    moves: list[str],
    time_sec: float,
    weights: Path | None,
    *,
    node_budget: int = 0,
    seed: int | None = None,
    temperature: float = 0.0,
    top_n: int = 8,
    weight_hash: str | None = None,
) -> tuple[str | None, dict[str, Any]]:
    # NOTE: TITANIUM_OPENING_TEMPERATURE / TITANIUM_SEARCH_SEED are set below
    # for provenance/debugging only - the engine binary does not read either
    # (verified: grep of engine/src finds zero references to both). Real
    # move selection under temperature happens in Python below, sampling
    # over the root-move scores the ONE real search call already produced
    # (no extra engine invocation) - see choose_root_move_by_temperature.
    env = os.environ.copy()
    if weights and weights.is_file():
        env["TITANIUM_NET_WEIGHTS_PATH"] = str(weights.resolve())
    if seed is not None:
        env["TITANIUM_SEARCH_SEED"] = str(seed)
    if temperature > 0:
        env["TITANIUM_OPENING_TEMPERATURE"] = str(temperature)
    cmd = [str(engine_bin), "genmove", "--engine", ENGINE_NAME, "--log", *moves]
    if node_budget > 0:
        cmd += ["--nodes", str(node_budget), "--time", str(time_sec)]
    else:
        cmd += ["--time", str(time_sec)]
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(engine_bin.resolve().parent.parent.parent if engine_bin.is_file() else Path.cwd()),
        timeout=max(time_sec * 3 + 20, 30),
        env=env,
    )
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-1000:] or f"engine exited {proc.returncode}")
    stats: dict[str, Any] = {"elapsed_sec": elapsed}
    stats.update(_parse_engine_info(proc.stdout))
    if weight_hash:
        stats["weight_hash"] = weight_hash
    stats["temperature"] = temperature
    if seed is not None:
        stats["seed"] = seed
    stats["engine_hash"] = os.environ.get("TITANIUM_ENGINE_HASH")
    best: str | None = None
    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if line.startswith("bestmove "):
            token = line.split()[1]
            best = None if token == "(none)" else token
            break
    if best is not None and temperature > 0:
        root_moves = stats.get("root_moves") or []
        rng = random.Random(seed if seed is not None else 0)
        if root_moves:
            chosen, was_exploratory = choose_root_move_by_temperature(
                root_moves, best, temperature, top_n=top_n, rng=rng
            )
        else:
            chosen, was_exploratory = choose_eval_move_by_temperature(
                engine_bin,
                moves,
                best,
                weights,
                temperature,
                top_n=top_n,
                rng=rng,
            )
        stats["exploratory"] = was_exploratory
        return chosen, stats
    stats["exploratory"] = False
    return best, stats


def play_matchup_game(
    *,
    cfg: RuntimeConfig,
    generation: dict[str, Any],
    matchup: MatchupChoice,
    game_id: str,
    worker_id: int,
    game_index: int,
) -> dict[str, Any]:
    gen_dir = Path(generation["path"])
    manifest = generation["manifest"]
    current_hash = manifest["current_deployed_hash"]
    prior_hash = manifest.get("prior_deployed_hash")
    engine_hash = manifest["engine_build_hash"]

    p0_weights = resolve_weight_path(cfg.data_dir, matchup.p0_hash, gen_dir)
    p1_weights = resolve_weight_path(cfg.data_dir, matchup.p1_hash, gen_dir)

    moves: list[str] = []
    move_stats: list[dict[str, Any]] = []
    state = OraclePositionState()
    repetitions = {state.packed_state(): 1}
    started_at = utc_now()
    termination = "max_plies"
    winner: int | None = None
    rng_seed = int(hashlib.sha256(game_id.encode()).hexdigest()[:8], 16)
    novelty_reached = False
    prefix_index = _opening_prefix_index(cfg) if matchup.opening_exploration else None
    use_opening = matchup.opening_exploration and prefix_index is not None
    # Matchup wants exploration but the DB-backed index isn't deployed here -
    # use the fixed-window fallback instead of silently playing 0-temperature
    # (fully deterministic) for the whole game.
    use_fallback_opening = matchup.opening_exploration and prefix_index is None

    for ply in range(cfg.max_plies):
        side = ply % 2
        weights = p0_weights if side == 0 else p1_weights
        weight_hash = matchup.p0_hash if side == 0 else matchup.p1_hash
        ply_num = len(moves) + 1
        engine_temp = 0.0
        if use_opening:
            prefix_known = prefix_index.is_known(moves)
            engine_temp, novelty_reached = opening_temperature_for_move(
                ply_num,
                novelty_reached,
                prefix_known,
            )
        elif use_fallback_opening and ply_num < FALLBACK_OPENING_EXPLORATION_MAX_PLY:
            engine_temp = OPENING_TEMPERATURE_DECAY ** (ply_num - OPENING_EXPLORATION_START_PLY) \
                if ply_num >= OPENING_EXPLORATION_START_PLY else 0.0
        mv, stats = engine_move(
            cfg.engine_bin,
            moves,
            cfg.move_time,
            weights,
            node_budget=cfg.node_budget,
            seed=rng_seed + ply,
            temperature=engine_temp,
            weight_hash=weight_hash,
        )
        stats["ply"] = ply
        move_stats.append(stats)
        if not mv:
            raise RuntimeError("engine produced no move")
        moves.append(mv)
        state = _apply_move(state, mv)
        if use_opening and not novelty_reached:
            if not prefix_index.is_known(moves):
                novelty_reached = True
        winner = check_winner(moves)
        if winner is not None:
            termination = "win"
            break
        packed = state.packed_state()
        repetitions[packed] = repetitions.get(packed, 0) + 1
        if repetitions[packed] >= REPETITION_DRAW_COUNT:
            termination = "repetition"
            break

    result = "DRAW" if winner is None else ("P0" if winner == 0 else "P1")
    payload: dict[str, Any] = {
        "game_id": game_id,
        "protocol_version": PROTOCOL_VERSION,
        "schema_version": GAME_SCHEMA_VERSION,
        "engine_build_hash": engine_hash,
        "current_weight_hash": current_hash,
        "prior_weight_hash": prior_hash,
        "generation_id": manifest["generation_id"],
        "worker_id": worker_id,
        "game_index": game_index,
        "seed": rng_seed,
        "moves": moves,
        "result": result,
        "termination_reason": termination,
        "draw_reason": termination if winner is None and termination in ("max_plies", "repetition") else None,
        "plies": len(moves),
        "time_control": {
            "move_time_sec": cfg.move_time,
            "node_budget": cfg.node_budget,
            "timeout_is_fallback": True,
        },
        "search": manifest.get("search_settings", {}),
        "started_at": started_at,
        "finished_at": utc_now(),
        "stats": {
            "avg_move_time_sec": sum(float(s.get("elapsed_sec", 0)) for s in move_stats)
            / max(len(move_stats), 1),
            "moves": move_stats,
        },
    }
    payload.update(matchup_to_payload_fields(matchup))
    payload["payload_checksum"] = game_payload_checksum(payload)
    errors = validate_game_payload(payload)
    if errors:
        raise ValueError("; ".join(errors))
    return payload


def play_scheduled_game(
    *,
    cfg: RuntimeConfig,
    generation: dict[str, Any],
    schedule: Any,
    worker_id: int,
) -> dict[str, Any]:
    """Backward-compatible wrapper for fixed schedules."""
    from .schedule import CURRENT_CURRENT, CURRENT_PRIOR_P0, PRIOR_CURRENT_P0

    manifest = generation["manifest"]
    current_hash = manifest["current_deployed_hash"]
    prior_hash = manifest.get("prior_deployed_hash")
    if schedule.matchup_type == CURRENT_CURRENT:
        matchup = choose_matchup(schedule.seed, current_hash, prior_hash)
    elif schedule.matchup_type == CURRENT_PRIOR_P0:
        matchup = MatchupChoice(
            p0_hash=current_hash,
            p1_hash=prior_hash or current_hash,
            kind="generation_mixed",
            opening_exploration=False,
            current_hash=current_hash,
            prior_hash=prior_hash,
        )
    elif schedule.matchup_type == PRIOR_CURRENT_P0:
        matchup = MatchupChoice(
            p0_hash=prior_hash or current_hash,
            p1_hash=current_hash,
            kind="generation_mixed",
            opening_exploration=False,
            current_hash=current_hash,
            prior_hash=prior_hash,
        )
    else:
        raise ValueError(f"unknown matchup_type {schedule.matchup_type}")
    game_id = (
        f"oracle-{manifest['generation_id']}-{worker_id:02d}-"
        f"{schedule.index:04d}-{uuid.uuid4().hex[:10]}"
    )
    return play_matchup_game(
        cfg=cfg,
        generation=generation,
        matchup=matchup,
        game_id=game_id,
        worker_id=worker_id,
        game_index=schedule.index,
    )


def preflight_engine(cfg: RuntimeConfig, generation: dict[str, Any]) -> dict[str, Any]:
    if not cfg.engine_bin.is_file():
        raise FileNotFoundError(cfg.engine_bin)
    gen_dir = Path(generation["path"])
    current = gen_dir / "current.bin"
    if sha256_file(current) != generation["manifest"]["current_deployed_hash"]:
        raise RuntimeError("current weight hash mismatch")
    mv, stats = engine_move(cfg.engine_bin, [], min(cfg.move_time, 0.1), current)
    if not mv:
        raise RuntimeError("engine returned no legal move from initial position")
    return {"ok": True, "initial_bestmove": mv, "stats": stats}
