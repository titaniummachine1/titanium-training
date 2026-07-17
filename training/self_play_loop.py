#!/usr/bin/env python3
"""
Combined self-play loop: strength verification + training data generation.

Runs on ONE thread, alternating two game types:

  VERIFY game  — trained net (A) vs frozen baseline (B)
                 Tracks win rate to confirm training is making progress.
                 Uses TITANIUM_NET_WEIGHTS_PATH for A, baked-in weights for B.

  TRAIN game   — trained net vs trained net (clean self-play)
                 Both sides use the same current best weights.
                 Generates new (position, outcome_label) pairs for the training DB.

All games write positions + labels to:
  training/data/canonical/games.db   — game DAG
  training/data/canonical/labels.db  — positions + value labels

After each game, the script re-reads net_weights_best.bin so it automatically
picks up improvements as the trainer runs in parallel.

Per-move timing is logged for BOTH sides so you can detect if the trained net
is slower to evaluate (search speed regression on low-power hardware).

Usage:
  python training/self_play_loop.py                    # default 1s/move, run forever
  python training/self_play_loop.py --time 0.5         # light load (CPU is busy)
  python training/self_play_loop.py --time 2.0         # stronger play
  python training/self_play_loop.py --verify-ratio 1   # 1 verify per 1 train game
  python training/self_play_loop.py --verify-ratio 4   # 1 verify per 4 train games
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_TRAINING = _REPO / "training"
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

from titanium_training.paths import ENGINE_BIN, REPO_ROOT
from db_import import (
    GAMES_DB_PATH, LABELS_DB_PATH,
    GAMES_SCHEMA, LABELS_SCHEMA,
    open_db, write_batch, now_utc,
)

NET_BEST  = _TRAINING / "runs" / "value_oracle" / "net_weights_best.bin"
NET_BASE  : Path | None = None   # set by --baseline-weights; overrides baked-in for B side
LOG_PATH  = _TRAINING / "data" / "self_play.log"
MAX_PLIES = 200   # draw if neither player wins within this many moves


# ─────────────────────────────────────────────────────────────────────────────
# Single-move engine call
# ─────────────────────────────────────────────────────────────────────────────

def engine_move(moves: list[str], time_sec: float, use_trained: bool) -> tuple[str | None, float]:
    """
    Ask titanium for the best move given the move history.
    use_trained=True → loads net_weights_best.bin via env var
    use_trained=False → uses baked-in compiled weights (frozen baseline)
    Returns (move_alg, elapsed_ms) or (None, 0) on failure.
    """
    env = os.environ.copy()
    if use_trained:
        env["TITANIUM_NET_WEIGHTS_PATH"] = str(NET_BEST)
    elif NET_BASE is not None:
        env["TITANIUM_NET_WEIGHTS_PATH"] = str(NET_BASE)
    else:
        env.pop("TITANIUM_NET_WEIGHTS_PATH", None)

    cmd = [str(ENGINE_BIN), "genmove"] + moves + ["--time", str(time_sec)]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, env=env,
            cwd=str(REPO_ROOT), timeout=time_sec * 4 + 5,
        )
    except subprocess.TimeoutExpired:
        return None, 0.0

    elapsed_ms = (time.perf_counter() - t0) * 1000
    out = proc.stdout.decode(errors="replace").strip()
    # genmove outputs "bestmove <alg>" — extract just the move token
    for line in reversed(out.splitlines()):
        line = line.strip()
        if line.startswith("bestmove "):
            return line.split()[1], elapsed_ms
    return None, elapsed_ms


# ─────────────────────────────────────────────────────────────────────────────
# Game over detection
# ─────────────────────────────────────────────────────────────────────────────

def check_winner(moves: list[str]) -> int | None:
    """
    P0 (starts e1) wins by reaching row 9.  P1 (starts e9) wins by reaching row 1.
    Pawn moves have no 'h'/'v' suffix.  Returns 0, 1, or None (game continues).
    """
    if not moves:
        return None
    last = moves[-1]
    if last[-1] in ("h", "v"):
        return None   # wall move, game continues
    row = last[-1]
    mover = (len(moves) - 1) % 2   # 0=P0 just moved, 1=P1 just moved
    if mover == 0 and row == "9":
        return 0
    if mover == 1 and row == "1":
        return 1
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Play one game, return move list + timing info
# ─────────────────────────────────────────────────────────────────────────────

def play_game(
    time_sec: float,
    p0_trained: bool,   # True → P0 uses trained net, P1 uses frozen (or both trained)
    p1_trained: bool,
) -> tuple[list[str], int | None, float, float]:
    """
    Play a full game.  Returns (moves, winner, avg_ms_p0, avg_ms_p1).
    winner = 0, 1, or None (draw/max_plies).
    """
    moves: list[str] = []
    ms_p0: list[float] = []
    ms_p1: list[float] = []

    for ply in range(MAX_PLIES):
        p0_to_move = (ply % 2 == 0)
        use_trained = p0_trained if p0_to_move else p1_trained

        mv, ms = engine_move(moves, time_sec, use_trained)
        if mv is None:
            break

        moves.append(mv)
        (ms_p0 if p0_to_move else ms_p1).append(ms)

        winner = check_winner(moves)
        if winner is not None:
            avg0 = sum(ms_p0) / max(len(ms_p0), 1)
            avg1 = sum(ms_p1) / max(len(ms_p1), 1)
            return moves, winner, avg0, avg1

    avg0 = sum(ms_p0) / max(len(ms_p0), 1)
    avg1 = sum(ms_p1) / max(len(ms_p1), 1)
    return moves, None, avg0, avg1


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def run(time_sec: float, verify_ratio: int, max_games: int | None) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not ENGINE_BIN.exists():
        print(f"ERROR: engine not found at {ENGINE_BIN}"); sys.exit(1)
    if not NET_BEST.exists():
        print(f"ERROR: no trained weights at {NET_BEST}"); sys.exit(1)

    games_db  = open_db(GAMES_DB_PATH,  GAMES_SCHEMA)
    labels_db = open_db(LABELS_DB_PATH, LABELS_SCHEMA)

    print(f"self_play_loop: time={time_sec}s/move  verify_ratio=1/{verify_ratio}", flush=True)
    frozen_label = NET_BASE.name if NET_BASE else "baked-in"
    print(f"  trained={NET_BEST.name}  frozen={frozen_label}", flush=True)
    print(f"  log={LOG_PATH}", flush=True)

    verify_wins = verify_losses = verify_draws = 0
    train_games = 0
    net_stamp = NET_BEST.stat().st_mtime
    game_i = 0

    with open(LOG_PATH, "a") as log:
        while max_games is None or game_i < max_games:

            # Alternate verify/train by verify_ratio
            is_verify = (game_i % (verify_ratio + 1) == 0)
            # Alternate P0/P1 assignment each pair of games
            trained_is_p0 = (game_i % 2 == 0)

            if is_verify:
                p0_trained = trained_is_p0
                p1_trained = not trained_is_p0
                game_type = "verify"
                game_id = f"selfplay_verify_{game_i:06d}"
            else:
                p0_trained = True
                p1_trained = True
                game_type = "train"
                game_id = f"selfplay_train_{game_i:06d}"

            t0 = time.perf_counter()
            moves, winner, ms_p0, ms_p1 = play_game(time_sec, p0_trained, p1_trained)
            elapsed = time.perf_counter() - t0

            outcome_p0 = 1 if winner == 0 else (-1 if winner == 1 else 0)

            # Who is the "trained" side for verify games?
            if is_verify:
                trained_won = (winner == 0) == trained_is_p0
                if winner is None:
                    verify_draws += 1; rs = "draw "
                elif trained_won:
                    verify_wins += 1; rs = "A WIN"
                else:
                    verify_losses += 1; rs = "B WIN"
                total_v = verify_wins + verify_losses + verify_draws
                a_rate = (verify_wins + 0.5 * verify_draws) / max(total_v, 1)
                ms_trained = ms_p0 if trained_is_p0 else ms_p1
                ms_frozen  = ms_p1 if trained_is_p0 else ms_p0
                line = (
                    f"[verify] game {game_i+1:4d}  {rs}  "
                    f"A={verify_wins} B={verify_losses} D={verify_draws}  A-rate={a_rate:.3f}  "
                    f"ms/move T={ms_trained:.0f} F={ms_frozen:.0f}  {elapsed:.0f}s"
                )
            else:
                train_games += 1
                line = (
                    f"[train ] game {game_i+1:4d}  {'P0' if winner==0 else 'P1' if winner==1 else 'draw'} wins  "
                    f"{len(moves)} moves  {elapsed:.0f}s  ms/move={ms_p0:.0f}"
                )

            print(line, flush=True)
            log.write(line + "\n"); log.flush()

            # Write game + labels to DB
            if moves:
                source = f"selfplay_{game_type}"
                try:
                    write_batch(
                        games_db, labels_db,
                        [(game_id, moves, outcome_p0, None, source)],
                        chunk_size=512, workers=1,
                    )
                except Exception as e:
                    print(f"  [db write error: {e}]", flush=True)

            game_i += 1

            # Pick up new weights if training has improved
            if NET_BEST.exists():
                new_stamp = NET_BEST.stat().st_mtime
                if new_stamp != net_stamp:
                    net_stamp = new_stamp
                    msg = f"  [weights updated — trained engine now uses newer net]"
                    print(msg, flush=True)
                    log.write(msg + "\n"); log.flush()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--time",         type=float, default=1.0,
                    help="Seconds per move (default 1.0; use 0.5 when CPU is busy)")
    ap.add_argument("--verify-ratio", type=int,   default=3,
                    help="Train games per verify game (default 3 → 1 verify per 4 games)")
    ap.add_argument("--games",        type=int,   default=None,
                    help="Stop after N games (default: run forever)")
    ap.add_argument("--baseline-weights", type=str, default=None,
                    help="Path to frozen baseline weights file for B side (default: baked-in engine weights)")
    args = ap.parse_args()
    if args.baseline_weights:
        global NET_BASE
        NET_BASE = Path(args.baseline_weights).resolve()
        if not NET_BASE.exists():
            print(f"ERROR: baseline weights not found: {NET_BASE}"); return 1
    run(args.time, args.verify_ratio, args.games)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
