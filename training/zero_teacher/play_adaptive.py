#!/usr/bin/env python3
"""Play one validated Titanium vs zero-ink game and emit JSON events."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

from titanium_training.store.move_codec import algebraic_to_ace  # noqa: E402
from zero_teacher.client import (  # noqa: E402
    START_STATE,
    ZeroSettings,
    ZeroTeacherClient,
    ace_to_zero_move,
    apply_zero_move,
    zero_move_is_legal,
    zero_move_text,
)


class TitaniumSession:
    def __init__(self, binary: Path, engine: str):
        self.proc = subprocess.Popen(
            [str(binary), "session", "--engine", engine],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

    def _send(self, line: str) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def _read_until(self, prefix: str) -> str:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            line = line.strip()
            if line.startswith(prefix):
                return line
        raise RuntimeError("Titanium session exited unexpectedly")

    def best_move(self, moves: list[str], seconds: float) -> str:
        self._send(f"position {' '.join(moves)}" if moves else "reset")
        self._read_until("ready")
        self._send(f"go {seconds}")
        line = self._read_until("bestmove ")
        return line[len("bestmove "):].strip()

    def close(self) -> None:
        try:
            self._send("quit")
        except (BrokenPipeError, OSError):
            pass
        try:
            self.proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def emit(kind: str, **payload) -> None:
    print(json.dumps({"type": kind, **payload}, separators=(",", ":")), flush=True)


def play(args) -> dict:
    client = ZeroTeacherClient(base=args.base, model_id=args.model, timeout_sec=args.timeout)
    settings = ZeroSettings(visits=args.visits, threads=args.threads)
    state = dict(START_STATE)
    snapshot = client.position(state)
    if snapshot.get("winner") is not None:
        raise RuntimeError("zero-ink start position is terminal")
    session = TitaniumSession(Path(args.bin), args.engine)
    moves: list[str] = []
    winner = None
    try:
        for ply in range(args.max_ply):
            current = int(state["currentPlayer"])
            our_turn = (current == 0) == args.our_is_p1
            if our_turn:
                text = session.best_move(moves, args.our_time)
                if not text or text == "(none)":
                    raise RuntimeError("Titanium returned no move in a non-terminal position")
                move = ace_to_zero_move(algebraic_to_ace(text))
                if not zero_move_is_legal(snapshot, move):
                    raise RuntimeError(f"Titanium returned illegal move {text}")
                state = apply_zero_move(state, move)
                snapshot = client.position(state)
            else:
                response = client.bot_move(state, settings)
                move = response["move"]
                if not zero_move_is_legal(snapshot, move):
                    raise RuntimeError(f"zero-ink returned illegal move {move}")
                text = zero_move_text(move)
                state = response["stateAfter"]
                snapshot = client.position(state)
            moves.append(text)
            emit("ply", ply=len(moves), side="us" if our_turn else "zero")
            if snapshot.get("winner") is not None:
                winner = int(snapshot["winner"])
                break
    finally:
        session.close()

    if winner not in (0, 1):
        return {"complete": False, "moves": moves, "plies": len(moves)}
    our_win = (winner == 0) == args.our_is_p1
    return {
        "complete": True,
        "moves": moves,
        "plies": len(moves),
        "winner": winner + 1,
        "result": "W" if winner == 0 else "B",
        "our_win": our_win,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--visits", type=int, required=True)
    ap.add_argument("--our-is-p1", type=int, choices=(0, 1), required=True)
    ap.add_argument("--our-time", type=float, default=5.0)
    ap.add_argument("--max-ply", type=int, default=300)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--engine", default="titanium-v15")
    ap.add_argument("--bin", default=str(ROOT / "engine" / "target" / "release" / "titanium.exe"))
    ap.add_argument("--base", default="https://quoridor-zero.ink")
    ap.add_argument("--model", default="resume-188/model_000159")
    args = ap.parse_args()
    args.our_is_p1 = bool(args.our_is_p1)
    try:
        emit("result", **play(args))
        return 0
    except Exception as exc:
        emit("error", error=str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
