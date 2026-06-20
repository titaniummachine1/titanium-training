"""Persistent monotonic rollout curriculum for adaptive remote opponents."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE_PATH = ROOT / "training" / "data" / "opponent_curriculum.json"
EVENTS_PATH = ROOT / "training" / "data" / "opponent_curriculum_events.jsonl"

SCHEMA = "opponent-curriculum-v1"
START_VISITS = 1
VISIT_STEP = 20
MAX_VISITS = 1_000_000
WINDOW_GAMES = 16
TARGET_SCORE = 1.0 / (1.0 + 10.0 ** (20.0 / 400.0))
OPPONENTS = ("ka", "zero")
ZERO_FALLBACK_WINS = 4


def _default_opponent() -> dict:
    return {
        "visits": START_VISITS,
        "next_color": 0,
        "window": [],
        "windows_completed": 0,
        "games_completed": 0,
        "last_window_wins": None,
    }


def default_state() -> dict:
    return {
        "schema": SCHEMA,
        "opponents": {name: _default_opponent() for name in OPPONENTS},
    }


def load_state(path: Path = STATE_PATH) -> dict:
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default_state()
    if state.get("schema") != SCHEMA:
        raise RuntimeError(f"curriculum schema {state.get('schema')!r} != {SCHEMA!r}")
    opponents = state.setdefault("opponents", {})
    for name in OPPONENTS:
        base = _default_opponent()
        base.update(opponents.get(name) or {})
        base["visits"] = max(START_VISITS, min(MAX_VISITS, int(base["visits"])))
        base["next_color"] = int(base["next_color"]) & 1
        base["window"] = [bool(v) for v in base.get("window", [])][-WINDOW_GAMES:]
        opponents[name] = base
    return state


def save_state(state: dict, path: Path = STATE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def claim_game(opponent: str, path: Path = STATE_PATH) -> dict:
    if opponent not in OPPONENTS:
        raise ValueError(f"unknown adaptive opponent: {opponent}")
    state = load_state(path)
    entry = state["opponents"][opponent]
    our_is_p1 = entry["next_color"] == 0
    entry["next_color"] ^= 1
    save_state(state, path)
    return {
        "opponent": opponent,
        "opponent_visits": int(entry["visits"]),
        "our_is_p1": our_is_p1,
        "window_games": len(entry["window"]),
    }


def _append_event(event: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, separators=(",", ":")) + "\n")


def record_result(
    opponent: str,
    *,
    our_win: bool,
    game_id: str,
    visits: int,
    state_path: Path = STATE_PATH,
    events_path: Path = EVENTS_PATH,
) -> dict:
    """Record one completed game and possibly advance a 16-game window."""
    if opponent not in OPPONENTS:
        raise ValueError(f"unknown adaptive opponent: {opponent}")
    state = load_state(state_path)
    entry = state["opponents"][opponent]
    entry["window"].append(bool(our_win))
    entry["games_completed"] = int(entry.get("games_completed", 0)) + 1
    update = None
    if len(entry["window"]) >= WINDOW_GAMES:
        window = entry["window"][:WINDOW_GAMES]
        wins = sum(window)
        old_visits = int(entry["visits"])
        if wins / WINDOW_GAMES >= TARGET_SCORE:
            steps = max(1, math.ceil(wins - TARGET_SCORE * WINDOW_GAMES))
            entry["visits"] = min(MAX_VISITS, old_visits + VISIT_STEP * steps)
        entry["window"] = entry["window"][WINDOW_GAMES:]
        entry["windows_completed"] = int(entry.get("windows_completed", 0)) + 1
        entry["last_window_wins"] = wins
        update = {
            "schema": SCHEMA,
            "opponent": opponent,
            "game_id": game_id,
            "window_games": WINDOW_GAMES,
            "our_wins": wins,
            "target_score": TARGET_SCORE,
            "old_visits": old_visits,
            "new_visits": int(entry["visits"]),
            "sample_visits": int(visits),
        }
        _append_event(update, events_path)
    save_state(state, state_path)
    return {
        "visits": int(entry["visits"]),
        "window_games": len(entry["window"]),
        "update": update,
    }


def preferred_adaptive_opponent(state: dict | None = None) -> str:
    """Prefer zero; fall back to Ka only after a clearly losing zero window."""
    state = state or load_state()
    zero = state["opponents"]["zero"]
    if int(zero.get("windows_completed", 0)) == 0:
        return "zero"
    wins = zero.get("last_window_wins")
    return "ka" if wins is not None and int(wins) <= ZERO_FALLBACK_WINS else "zero"
