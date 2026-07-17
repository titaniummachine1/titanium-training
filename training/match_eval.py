#!/usr/bin/env python3
"""
Continuous match evaluator: trained net vs frozen baseline.

Runs games between:
  A (trained)  — uses TITANIUM_NET_WEIGHTS_PATH → net_weights_best.bin
                 Refreshed after every N games so training improvements are picked up.
  B (frozen)   — uses baked-in net_weights.bin (weights at last engine compile)

Reports per game: result + ms/move for each side.
A slowdown in ms/move means the trained net is more expensive to evaluate —
that would mean search speed is regressing, which matters on low-power hardware.

Usage:
  python training/match_eval.py                   # run forever
  python training/match_eval.py --games 20        # stop after 20 games
  python training/match_eval.py --time 1.0        # 1s per move
  python training/match_eval.py --time 0.5        # 0.5s (light, use when CPU is busy)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
ENGINE = _REPO / "engine" / "target" / "release" / "titanium.exe"
NET_BEST = _REPO / "training" / "runs" / "value_oracle" / "net_weights_best.bin"
LOG_PATH = _REPO / "training" / "data" / "match_eval.log"


def genmove(moves: list[str], time_sec: float, use_trained: bool) -> tuple[str | None, float]:
    """
    Ask the engine for the best move given the move history.
    Returns (move_alg, elapsed_ms).  Returns (None, 0) on failure.
    """
    cmd = [str(ENGINE), "genmove"] + moves + ["--time", str(time_sec)]
    env = os.environ.copy()
    if use_trained:
        env["TITANIUM_NET_WEIGHTS_PATH"] = str(NET_BEST)
    else:
        env.pop("TITANIUM_NET_WEIGHTS_PATH", None)   # use baked-in weights

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, env=env,
            cwd=str(_REPO), timeout=time_sec * 4 + 5,
        )
    except subprocess.TimeoutExpired:
        return None, 0.0
    elapsed_ms = (time.perf_counter() - t0) * 1000

    out = proc.stdout.decode(errors="replace").strip()
    # genmove returns the chosen move on the last non-empty line
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    move = lines[-1] if lines else None
    return move, elapsed_ms


def is_game_over(moves: list[str]) -> int | None:
    """
    Detect game over from the move sequence.
    Returns 0 (P0 wins), 1 (P1 wins), or None (game continues).

    P0 (starts e1) wins by moving a pawn to any cell in row 9.
    P1 (starts e9) wins by moving a pawn to any cell in row 1.
    A move is a pawn move if it has no 'h' or 'v' suffix.
    """
    if not moves:
        return None
    last = moves[-1]
    is_pawn = last[-1] not in ("h", "v")
    if not is_pawn:
        return None
    # last character of a pawn move is the row digit
    row = last[-1]
    ply = len(moves) - 1   # 0-indexed ply of the last move
    mover = ply % 2        # 0 = P0 moved, 1 = P1 moved
    if mover == 0 and row == "9":
        return 0   # P0 reached goal row
    if mover == 1 and row == "1":
        return 1   # P1 reached goal row
    return None


def play_game(
    time_sec: float,
    trained_is_p0: bool,
    max_plies: int = 200,
) -> tuple[int | None, float, float]:
    """
    Play one game.  Returns (winner, avg_ms_trained, avg_ms_frozen).
    winner = 0 (P0), 1 (P1), or None (max_plies reached).
    """
    moves: list[str] = []
    ms_trained: list[float] = []
    ms_frozen: list[float] = []

    for ply in range(max_plies):
        p0_to_move = (ply % 2 == 0)
        trained_to_move = (p0_to_move == trained_is_p0)

        mv, ms = genmove(moves, time_sec, use_trained=trained_to_move)
        if mv is None:
            return None, 0.0, 0.0   # engine error

        moves.append(mv)
        (ms_trained if trained_to_move else ms_frozen).append(ms)

        winner = is_game_over(moves)
        if winner is not None:
            avg_t = sum(ms_trained) / max(len(ms_trained), 1)
            avg_f = sum(ms_frozen) / max(len(ms_frozen), 1)
            return winner, avg_t, avg_f

    return None, 0.0, 0.0   # draw / timeout


def run_match(n_games: int | None, time_sec: float) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not ENGINE.exists():
        print(f"ERROR: engine not found at {ENGINE}"); sys.exit(1)
    if not NET_BEST.exists():
        print(f"ERROR: no trained weights at {NET_BEST}"); sys.exit(1)

    print(f"Match: A=trained({NET_BEST.name}) vs B=frozen(baked-in)  time={time_sec}s/move")
    print(f"Log: {LOG_PATH}")
    print()

    a_wins = b_wins = draws = 0
    game_i = 0
    net_stamp: float = NET_BEST.stat().st_mtime

    with open(LOG_PATH, "a") as log:
        while n_games is None or game_i < n_games:
            # Alternate colors: even games A=P0, odd games A=P1
            trained_is_p0 = (game_i % 2 == 0)

            t_game = time.perf_counter()
            winner, ms_t, ms_f = play_game(time_sec, trained_is_p0)
            elapsed_s = time.perf_counter() - t_game

            if winner is None:
                draws += 1
                result_str = "draw"
                a_score = 0.5
            else:
                trained_won = (winner == 0) == trained_is_p0
                if trained_won:
                    a_wins += 1
                    result_str = "A wins"
                else:
                    b_wins += 1
                    result_str = "B wins"
                a_score = 1.0 if trained_won else 0.0

            total = a_wins + b_wins + draws
            a_rate = (a_wins + 0.5 * draws) / max(total, 1)

            line = (
                f"game {game_i+1:4d}  {result_str:<8}"
                f"  A={a_wins} B={b_wins} D={draws}  A-rate={a_rate:.3f}"
                f"  ms/move: A={ms_t:.0f} B={ms_f:.0f}"
                f"  game={elapsed_s:.0f}s"
            )
            print(line, flush=True)
            log.write(line + "\n"); log.flush()

            game_i += 1

            # Re-read best weights if the file changed (picks up training improvements)
            new_stamp = NET_BEST.stat().st_mtime if NET_BEST.exists() else net_stamp
            if new_stamp != net_stamp:
                net_stamp = new_stamp
                msg = f"  [net_weights_best.bin updated — A now uses newer weights]"
                print(msg, flush=True)
                log.write(msg + "\n"); log.flush()

    total = a_wins + b_wins + draws
    a_rate = (a_wins + 0.5 * draws) / max(total, 1)
    print(f"\nFinal: A={a_wins} B={b_wins} D={draws}  A-rate={a_rate:.3f}")


def main() -> int:
    from prep_guard import guard_real_work

    guard_real_work("candidate_gating", detail="match_eval.py")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=None, help="Stop after N games (default: forever)")
    ap.add_argument("--time",  type=float, default=2.0, help="Seconds per move (default: 2.0)")
    args = ap.parse_args()
    run_match(args.games, args.time)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
