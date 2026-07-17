#!/usr/bin/env python3
"""Parallel engine-vs-engine match (pool-style workers, sudden-death clocks).

Plays A vs B with warm ``titanium session`` processes, mirrored opening pairs,
and a per-side game clock (default 60s). Games run until a win (high ply cap
only as a safety net). Multiple worker threads each own one warm session pair
and pull game indices from a shared queue — same model as local_game_pool /
oracle supervisor, but for binary engine flags (e.g. titanium-v17 vs v16).

Shard across machines with --shard-count / --shard-offset / --shard-span so
4 local + 13 oracle workers can split 200 games without overlap.

Stop gracefully: create the stop file (or Ctrl+C). In-flight games finish;
no new games start; status.json records final totals.

Usage (local, 17 workers, 200 games):
  set TITANIUM_ENGINE_BIN=tools\\binary_match\\bin\\titanium_v17.exe
  python tools/binary_match/parallel_engine_match.py \\
      --engine-a titanium-v17 --engine-b titanium-v16 \\
      --games 200 --clock-sec 60 --workers 17

Shard 0 of 17 (local slot 0 only, 1 worker):
  python tools/binary_match/parallel_engine_match.py ... --shard-count 17 \\
      --shard-offset 0 --shard-span 1 --workers 1
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Queue
from typing import Any

_REPO = Path(os.environ.get("TITANIUM_GAME_FACTORY_ROOT", Path(__file__).resolve().parents[2]))
_TRAINING = _REPO / "training"
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

from engine_session import EngineSession  # noqa: E402


def check_winner(moves: list[str]) -> int | None:
    """Return the winning side after the latest pawn move, if any."""
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

DEFAULT_OUT = Path(__file__).resolve().parent / "runs" / "v17_vs_v16"
DEFAULT_STOP = Path(__file__).resolve().parent / "match_v17_vs_v16.stop"
VALID_TERMINATIONS = frozenset({"goal", "time", "ply_cap"})
MAX_INVALID_ATTEMPTS_PER_GAME = 3


@dataclass(frozen=True)
class MatchConfig:
    engine_a: str
    engine_b: str
    games: int
    clock_sec: float
    open_plies: int
    max_plies: int
    seed: int
    engine_threads: int
    shard_count: int
    shard_offset: int
    shard_span: int
    workers: int
    out_dir: Path
    stop_file: Path
    weights: Path | None
    weights_a: Path | None
    weights_b: Path | None
    engine_bin_a: Path | None
    engine_bin_b: Path | None
    resume_from: tuple[Path, ...]
    no_early_elimination: bool = False


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def wilson_lower(successes: float, n: int, z: float = 1.96) -> float:
    if n <= 0:
        return 0.0
    p = successes / n
    z2 = z * z
    denom = 1 + z2 / n
    center = p + z2 / (2 * n)
    margin = z * math.sqrt((p * (1.0 - p) + z2 / (4 * n)) / n)
    return (center - margin) / denom


def shard_owns_game(cfg: MatchConfig, game_idx: int) -> bool:
    slot = game_idx % cfg.shard_count
    return cfg.shard_offset <= slot < cfg.shard_offset + cfg.shard_span


def games_for_shard(cfg: MatchConfig) -> list[int]:
    return [i for i in range(cfg.games) if shard_owns_game(cfg, i)]


def early_elimination_enabled(cfg: MatchConfig) -> bool:
    """Only the process owning every game can infer the global match score."""
    return not cfg.no_early_elimination and len(games_for_shard(cfg)) == cfg.games


def _load_opening_dag() -> tuple[tuple[str, ...], ...]:
    """Load the audited non-Titanium opening DAG from the training data.

    The runner's old random-token generator produced illegal prefixes, causing
    engine session errors and fake 5-ply terminal wins.  Using the real DAG
    guarantees legal, varied openings.
    """
    configured = os.environ.get("TITANIUM_OPENING_BOOK")
    path = Path(configured) if configured else (
        _REPO / "training" / "data" / "opening_book" / "non_titanium_10ply.json"
    )
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        rows = doc["nodesByPly"].get("10", []) or doc["nodesByPly"].get("14", [])
        openings = sorted(
            {
                tuple(str(m) for m in row["prefix"])
                for row in rows
                if len(row.get("prefix", ())) >= 6
            }
        )
        if openings:
            return tuple(openings)
    except (OSError, KeyError, json.JSONDecodeError):
        pass
    # Fallback: deterministic human book lines if DAG is missing.
    return (
        ("e2", "e8", "e3", "e7", "e4", "e6"),
        ("e2", "e8", "e3", "e7", "e4", "d4v"),
        ("e2", "e8", "e3", "e7", "e4", "e6", "a3h", "d4v"),
        ("e2", "e8", "e3", "e7", "e4", "e6", "d3h", "c6h", "e6v"),
    )


_OPENING_DAG: tuple[tuple[str, ...], ...] = _load_opening_dag()


def opening_for_pair(cfg: MatchConfig, pair_idx: int, rng: random.Random) -> list[str]:
    """Return a deterministic, legal opening prefix for the mirrored pair."""
    pair_rng = random.Random((cfg.seed ^ (pair_idx * 0x9E3779B1)) & 0xFFFFFFFF)
    base = pair_rng.choice(_OPENING_DAG)
    # If open_plies is shorter than the book line, truncate deterministically.
    n = min(cfg.open_plies, len(base))
    return list(base[:n])


def play_clock_game(
    sess_a: EngineSession,
    sess_b: EngineSession,
    *,
    cfg: MatchConfig,
    opening: list[str],
    a_is_p0: bool,
) -> tuple[str, int, dict[str, float], str]:
    """Play one game with a hard per-side deadline.

    Position synchronization is intentionally free: sessions are warm and the
    website clock likewise starts only when the search begins.  The search
    result itself must arrive before the remaining game clock expires.
    """
    moves = list(opening)
    clock_ms = {"A": cfg.clock_sec * 1000.0, "B": cfg.clock_sec * 1000.0}

    for ply in range(len(moves), cfg.max_plies):
        winner = check_winner(moves)
        if winner is not None:
            if winner == 0:
                tag = "A" if a_is_p0 else "B"
            else:
                tag = "B" if a_is_p0 else "A"
            return tag, len(moves), {k: v / 1000.0 for k, v in clock_ms.items()}, "goal"

        is_p0_turn = (ply % 2) == 0
        side_a = is_p0_turn == a_is_p0
        sess = sess_a if side_a else sess_b
        tag = "A" if side_a else "B"

        if clock_ms[tag] <= 0.0:
            return (
                "B" if side_a else "A",
                len(moves),
                {k: v / 1000.0 for k, v in clock_ms.items()},
                "time",
            )
        if not sess.alive():
            return ("B" if side_a else "A"), len(moves), {k: v / 1000.0 for k, v in clock_ms.items()}, "engine_dead"
        if not sess.sync(moves):
            return ("B" if side_a else "A"), len(moves), {k: v / 1000.0 for k, v in clock_ms.items()}, "sync_failed"

        remaining_ms = clock_ms[tag]
        move_sec = max(0.001, remaining_ms / 1000.0 / 20.0)
        t0 = time.perf_counter()
        mv = sess.go(move_sec)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        clock_ms[tag] = max(0.0, remaining_ms - elapsed_ms)

        if elapsed_ms > remaining_ms:
            return (
                "B" if side_a else "A",
                len(moves),
                {k: v / 1000.0 for k, v in clock_ms.items()},
                "time",
            )
        if not mv:
            return ("B" if side_a else "A"), len(moves), {k: v / 1000.0 for k, v in clock_ms.items()}, "no_move"
        moves.append(mv)

    return "draw", len(moves), {k: v / 1000.0 for k, v in clock_ms.items()}, "ply_cap"


class MatchState:
    def __init__(self, cfg: MatchConfig, resumed_rows: dict[int, dict[str, Any]] | None = None):
        self.cfg = cfg
        self.early_elimination_enabled = early_elimination_enabled(cfg)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        resumed = list((resumed_rows or {}).values())
        self.a_wins = sum(row.get("winner") == "A" for row in resumed)
        self.b_wins = sum(row.get("winner") == "B" for row in resumed)
        self.draws = sum(row.get("winner") not in ("A", "B") for row in resumed)
        self.completed = len(resumed)
        self.resumed = len(resumed)
        self.errors = 0
        self.invalid_attempts: dict[int, int] = {}
        self.early_stop_reason: str | None = None
        self.started_at = utc_now()

    def record_invalid(self, game_idx: int) -> int:
        with self.lock:
            self.errors += 1
            attempts = self.invalid_attempts.get(game_idx, 0) + 1
            self.invalid_attempts[game_idx] = attempts
            if attempts >= MAX_INVALID_ATTEMPTS_PER_GAME:
                self.early_stop_reason = (
                    f"game {game_idx} failed {attempts} times without a valid termination"
                )
                self.stop.set()
            return attempts

    def claim_game(self) -> bool:
        """Claim a queued game unless a stop was already requested."""
        with self.lock:
            return not self.stop.is_set()

    def record(self, winner: str) -> str | None:
        with self.lock:
            if winner == "A":
                self.a_wins += 1
            elif winner == "B":
                self.b_wins += 1
            else:
                self.draws += 1
            self.completed += 1
            remaining = len(games_for_shard(self.cfg)) - self.completed
            max_possible_score = (
                self.a_wins + 0.5 * self.draws + remaining
            ) / (self.completed + remaining)
            if self.early_elimination_enabled and max_possible_score <= 0.5:
                self.early_stop_reason = (
                    "candidate A maximum possible final score "
                    f"{max_possible_score:.4f} <= 0.5000"
                )
                self.stop.set()
                return self.early_stop_reason
        return None

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            n = self.completed
            score = (self.a_wins + 0.5 * self.draws) / n if n else 0.0
            snapshot = {
                "running": not self.stop.is_set(),
                "target_games": self.cfg.games,
                "shard_games": len(games_for_shard(self.cfg)),
                "completed_games": n,
                "resumed_games": self.resumed,
                "a_wins": self.a_wins,
                "b_wins": self.b_wins,
                "draws": self.draws,
                "score_a": round(score, 4),
                "wilson_lb_a": round(wilson_lower(self.a_wins + 0.5 * self.draws, n), 4) if n else 0.0,
                "errors": self.errors,
                "engine_a": self.cfg.engine_a,
                "engine_b": self.cfg.engine_b,
                "clock_sec": self.cfg.clock_sec,
                "workers": self.cfg.workers,
                "shard": {
                    "count": self.cfg.shard_count,
                    "offset": self.cfg.shard_offset,
                    "span": self.cfg.shard_span,
                },
                "started_at": self.started_at,
                "updated_at": utc_now(),
            }
            if self.early_stop_reason is not None:
                snapshot["early_stop_reason"] = self.early_stop_reason
            return snapshot


def write_status(cfg: MatchConfig, state: MatchState) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.out_dir / "status.json"
    # Every worker may publish status concurrently. A shared `status.tmp`
    # races on Windows (`replace` can see another thread's open/moved file).
    tmp = path.with_name(f"status.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(state.snapshot(), indent=2), encoding="utf-8")
    for attempt in range(5):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.01 * (attempt + 1))


def append_jsonl(cfg: MatchConfig, row: dict[str, Any]) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.out_dir / f"results_shard_{cfg.shard_offset}_{cfg.shard_span}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def append_invalid_attempt(cfg: MatchConfig, row: dict[str, Any]) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.out_dir / f"invalid_attempts_shard_{cfg.shard_offset}_{cfg.shard_span}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def valid_result_row(row: dict[str, Any]) -> bool:
    return row.get("termination") in VALID_TERMINATIONS


def result_path(cfg: MatchConfig) -> Path:
    return cfg.out_dir / f"results_shard_{cfg.shard_offset}_{cfg.shard_span}.jsonl"


def load_resume_rows(cfg: MatchConfig) -> dict[int, dict[str, Any]]:
    """Load, de-duplicate, and repartition prior rows for this shard span.

    Resume inputs may contain results from a different shard layout. Stable
    global ``game_idx`` values let the established 0..16 layout retain those
    games without replaying or double-counting them.
    """
    rows: dict[int, dict[str, Any]] = {}
    sources = list(cfg.resume_from)
    own_path = result_path(cfg)
    if own_path.is_file() and own_path not in sources:
        sources.append(own_path)
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(f"resume results not found: {source}")
        for line_no, line in enumerate(source.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                game_idx = int(row["game_idx"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid resume row {source}:{line_no}: {exc}") from exc
            if not 0 <= game_idx < cfg.games or not shard_owns_game(cfg, game_idx):
                continue
            if not valid_result_row(row):
                continue
            previous = rows.get(game_idx)
            if previous is not None and previous != row:
                raise ValueError(f"conflicting resume rows for game_idx={game_idx}")
            rows[game_idx] = row

    if rows:
        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        own_path.write_text(
            "".join(json.dumps(rows[idx], separators=(",", ":")) + "\n" for idx in sorted(rows)),
            encoding="utf-8",
        )
    return rows


def worker_loop(
    worker_id: int,
    cfg: MatchConfig,
    state: MatchState,
    work: Queue[int],
    openings_cache: dict[int, list[str]],
    openings_lock: threading.Lock,
) -> None:
    weights_a = cfg.weights_a if cfg.weights_a is not None else cfg.weights
    weights_b = cfg.weights_b if cfg.weights_b is not None else cfg.weights
    sess_a = EngineSession(
        cfg.engine_a,
        weights_a,
        threads=cfg.engine_threads,
        engine_bin=cfg.engine_bin_a,
    )
    sess_b = EngineSession(
        cfg.engine_b,
        weights_b,
        threads=cfg.engine_threads,
        engine_bin=cfg.engine_bin_b,
    )
    try:
        while not state.stop.is_set():
            try:
                game_idx = work.get(timeout=0.5)
            except Empty:
                if state.stop.is_set():
                    break
                if cfg.stop_file.is_file():
                    state.stop.set()
                continue
            if not state.claim_game():
                work.task_done()
                break
            game_committed = False
            try:
                pair_idx = game_idx // 2
                a_is_p0 = (game_idx % 2) == 0
                with openings_lock:
                    if pair_idx not in openings_cache:
                        rng = random.Random(cfg.seed)
                        openings_cache[pair_idx] = opening_for_pair(cfg, pair_idx, rng)
                    opening = list(openings_cache[pair_idx])
                winner, plies, clocks, termination = play_clock_game(
                    sess_a, sess_b, cfg=cfg, opening=opening, a_is_p0=a_is_p0
                )
                if termination not in VALID_TERMINATIONS:
                    attempts = state.record_invalid(game_idx)
                    append_invalid_attempt(
                        cfg,
                        {
                            "game_idx": game_idx,
                            "pair_idx": pair_idx,
                            "a_is_p0": a_is_p0,
                            "reported_winner": winner,
                            "plies": plies,
                            "clocks": clocks,
                            "termination": termination,
                            "attempt": attempts,
                            "worker_id": worker_id,
                            "recorded_at": utc_now(),
                        },
                    )
                    print(
                        f"game {game_idx:3d} INVALID ({termination}, {plies} plies), "
                        f"attempt {attempts}/{MAX_INVALID_ATTEMPTS_PER_GAME}; restarting sessions",
                        flush=True,
                    )
                    sess_a.close()
                    sess_b.close()
                    if not state.stop.is_set():
                        sess_a = EngineSession(
                            cfg.engine_a,
                            weights_a,
                            threads=cfg.engine_threads,
                            engine_bin=cfg.engine_bin_a,
                        )
                        sess_b = EngineSession(
                            cfg.engine_b,
                            weights_b,
                            threads=cfg.engine_threads,
                            engine_bin=cfg.engine_bin_b,
                        )
                        work.put(game_idx)
                    write_status(cfg, state)
                    continue
                append_jsonl(
                    cfg,
                    {
                        "game_idx": game_idx,
                        "pair_idx": pair_idx,
                        "a_is_p0": a_is_p0,
                        "winner": winner,
                        "plies": plies,
                        "clocks": clocks,
                        "termination": termination,
                        "worker_id": worker_id,
                        "recorded_at": utc_now(),
                    },
                )
                early_stop_reason = state.record(winner)
                game_committed = True
                snap = state.snapshot()
                print(
                    f"game {game_idx:3d}  {winner} wins  "
                    f"(A as {'p0' if a_is_p0 else 'p1'}, {plies} plies)  "
                    f"shard A:{snap['a_wins']}W {snap['draws']}D {snap['b_wins']}L  "
                    f"score={snap['score_a']:.3f}",
                    flush=True,
                )
                if early_stop_reason is not None:
                    print(f"EARLY STOP: {early_stop_reason}", flush=True)
                write_status(cfg, state)
            except Exception as exc:
                if game_committed:
                    print(
                        f"worker {worker_id} game {game_idx} post-score warning: {exc}",
                        flush=True,
                    )
                    continue
                attempts = state.record_invalid(game_idx)
                append_invalid_attempt(
                    cfg,
                    {
                        "game_idx": game_idx,
                        "termination": "exception",
                        "error": repr(exc),
                        "attempt": attempts,
                        "worker_id": worker_id,
                        "recorded_at": utc_now(),
                    },
                )
                print(
                    f"worker {worker_id} game {game_idx} INVALID exception: {exc}; "
                    f"attempt {attempts}/{MAX_INVALID_ATTEMPTS_PER_GAME}; restarting sessions",
                    flush=True,
                )
                sess_a.close()
                sess_b.close()
                if not state.stop.is_set():
                    sess_a = EngineSession(
                        cfg.engine_a,
                        weights_a,
                        threads=cfg.engine_threads,
                        engine_bin=cfg.engine_bin_a,
                    )
                    sess_b = EngineSession(
                        cfg.engine_b,
                        weights_b,
                        threads=cfg.engine_threads,
                        engine_bin=cfg.engine_bin_b,
                    )
                    work.put(game_idx)
                write_status(cfg, state)
            finally:
                work.task_done()
    finally:
        sess_a.close()
        sess_b.close()


def parse_args() -> MatchConfig:
    ap = argparse.ArgumentParser(description="Parallel titanium engine A vs B match")
    ap.add_argument("--engine-a", default="titanium-v17")
    ap.add_argument("--engine-b", default="titanium-v16")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--clock-sec", type=float, default=60.0)
    ap.add_argument("--open-plies", type=int, default=4)
    ap.add_argument("--max-plies", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--engine-threads", type=int, default=1)
    ap.add_argument("--workers", type=int, default=17)
    ap.add_argument("--shard-count", type=int, default=17)
    ap.add_argument("--shard-offset", type=int, default=0)
    ap.add_argument("--shard-span", type=int, default=17)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--stop-file", type=Path, default=DEFAULT_STOP)
    ap.add_argument("--weights", type=Path, default=None)
    ap.add_argument("--weights-a", type=Path, default=None)
    ap.add_argument("--weights-b", type=Path, default=None)
    ap.add_argument("--engine-bin-a", type=Path, default=None)
    ap.add_argument("--engine-bin-b", type=Path, default=None)
    ap.add_argument("--opening-book", type=Path, default=None)
    ap.add_argument(
        "--no-early-elimination",
        action="store_true",
        help="Never stop on a local/shard score; required for adaptive sharded matches",
    )
    ap.add_argument(
        "--resume-from",
        type=Path,
        action="append",
        default=[],
        help="Existing JSONL results to retain, repartitioned by global game_idx",
    )
    args = ap.parse_args()
    if args.games <= 0 or args.games % 2:
        ap.error("--games must be a positive even number (mirrored pairs)")
    if args.shard_offset < 0 or args.shard_span <= 0:
        ap.error("invalid shard offset/span")
    if args.shard_offset + args.shard_span > args.shard_count:
        ap.error("shard offset+span exceeds shard-count")
    if args.opening_book is not None:
        os.environ["TITANIUM_OPENING_BOOK"] = str(args.opening_book)
        global _OPENING_DAG
        _OPENING_DAG = _load_opening_dag()
    return MatchConfig(
        engine_a=args.engine_a,
        engine_b=args.engine_b,
        games=args.games,
        clock_sec=args.clock_sec,
        open_plies=args.open_plies,
        max_plies=args.max_plies,
        seed=args.seed,
        engine_threads=args.engine_threads,
        shard_count=args.shard_count,
        shard_offset=args.shard_offset,
        shard_span=args.shard_span,
        workers=args.workers,
        out_dir=args.out_dir,
        stop_file=args.stop_file,
        weights=args.weights,
        weights_a=args.weights_a,
        weights_b=args.weights_b,
        engine_bin_a=args.engine_bin_a,
        engine_bin_b=args.engine_bin_b,
        resume_from=tuple(args.resume_from),
        no_early_elimination=args.no_early_elimination,
    )


def main() -> int:
    cfg = parse_args()
    if cfg.games % 2:
        print("error: games must be even", file=sys.stderr)
        return 2

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    if cfg.stop_file.is_file():
        cfg.stop_file.unlink()

    all_shard_games = games_for_shard(cfg)
    resumed_rows = load_resume_rows(cfg)
    shard_games = [game_idx for game_idx in all_shard_games if game_idx not in resumed_rows]
    state = MatchState(cfg, resumed_rows)
    work: Queue[int] = Queue()
    for game_idx in shard_games:
        work.put(game_idx)

    openings_cache: dict[int, list[str]] = {}
    openings_lock = threading.Lock()

    def on_signal(_signum: int, _frame: object) -> None:
        state.stop.set()

    signal.signal(signal.SIGINT, on_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, on_signal)

    engine_bin = os.environ.get("TITANIUM_ENGINE_BIN", "(default)")
    engine_bins = (
        f"A={cfg.engine_bin_a or engine_bin}, B={cfg.engine_bin_b or engine_bin}"
    )
    print(
        f"parallel_engine_match  A={cfg.engine_a}  B={cfg.engine_b}  "
        f"games={cfg.games}  shard={cfg.shard_offset}+{cfg.shard_span}/{cfg.shard_count}  "
        f"this_shard={len(all_shard_games)}  resumed={len(resumed_rows)}  "
        f"remaining={len(shard_games)}  workers={cfg.workers}  "
        f"clock={cfg.clock_sec}s/side/game  engines={engine_bins}",
        flush=True,
    )
    write_status(cfg, state)

    threads = [
        threading.Thread(
            target=worker_loop,
            args=(wid, cfg, state, work, openings_cache, openings_lock),
            name=f"match-worker-{wid}",
            daemon=True,
        )
        for wid in range(cfg.workers)
    ]
    for t in threads:
        t.start()

    try:
        while any(t.is_alive() for t in threads):
            # When every scheduled game has been recorded there is no more
            # queue work to wake the workers.  Request their normal timeout
            # exit instead of polling forever with a stale `running: true`.
            if state.snapshot()["completed_games"] >= len(all_shard_games):
                state.stop.set()
            if cfg.stop_file.is_file():
                print(f"stop file detected: {cfg.stop_file}", flush=True)
                state.stop.set()
            time.sleep(0.5)
    finally:
        state.stop.set()
        for t in threads:
            # A graceful stop promises that already-claimed games finish. Do
            # not publish a final status while those workers can still append
            # result rows behind it.
            while t.is_alive():
                t.join(timeout=0.5)
                write_status(cfg, state)

    final = state.snapshot()
    final["running"] = False
    final["finished_at"] = utc_now()
    (cfg.out_dir / "status.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(
        f"\nFINAL shard {cfg.shard_offset}+{cfg.shard_span}: "
        f"A {final['a_wins']}W {final['draws']}D {final['b_wins']}L / {final['completed_games']}  "
        f"score={final['score_a']:.3f}  wilson_lb={final['wilson_lb_a']:.3f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
