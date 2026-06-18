"""Persistent color/opening rotation for coordinator-claimed local games."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE_PATH = ROOT / "training" / "data" / "local_game_rotation.json"
SCHEMA = "local-game-rotation-v1"
_lock = threading.Lock()


def _load(path: Path) -> dict:
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"schema": SCHEMA, "pairings": {}}
    if state.get("schema") != SCHEMA:
        raise RuntimeError(f"local rotation schema {state.get('schema')!r} != {SCHEMA!r}")
    state.setdefault("pairings", {})
    return state


def _save(state: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def claim_local_game(pairing_key: str, path: Path = STATE_PATH) -> dict:
    with _lock:
        state = _load(path)
        entry = state["pairings"].setdefault(pairing_key, {"next_index": 0})
        game_index = max(0, int(entry.get("next_index", 0)))
        entry["next_index"] = game_index + 1
        _save(state, path)
    return {"game_index": game_index, "our_is_p1": game_index % 2 == 0}
