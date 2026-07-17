"""Shared helpers for the Claustrophobia label-quality audit."""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

MARGIN_THRESHOLD_CP = 50.0
SEED = 20260716


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def rows_from(payload: Any) -> list[dict]:
    if isinstance(payload, dict):
        value = payload.get("rows", payload.get("roots", payload))
    else:
        value = payload
    return list(value) if isinstance(value, list) else []


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def move_type(row: dict) -> str:
    action = str(row.get("claustrophobia_action") or "")
    return "wall" if len(action) >= 3 else "pawn"


def score_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        for key in ("cp", "centipawns", "value", "score"):
            result = score_value(value.get(key))
            if result is not None:
                return result
    return None


def result_score(info: dict | None) -> float | None:
    info = info or {}
    for key in ("rootScore", "score"):
        result = score_value(info.get(key))
        if result is not None:
            return result
    moves = info.get("rootMoves", info.get("root_moves", [])) or []
    if moves and isinstance(moves[0], dict):
        return score_value(moves[0].get("score"))
    return None


def top_moves(info: dict | None) -> list[dict]:
    info = info or {}
    moves = info.get("rootMoves", info.get("root_moves", [])) or []
    return [m for m in moves if isinstance(m, dict)]


def move_from_entry(entry: dict) -> str | None:
    for key in ("move", "bestmove", "uci", "action"):
        value = entry.get(key)
        if isinstance(value, str):
            return value
    return None


def margin_cp(info: dict | None) -> float | None:
    moves = top_moves(info)
    if len(moves) < 2:
        return None
    values = [score_value(m.get("score")) for m in moves[:2]]
    if any(v is None for v in values):
        return None
    return abs(float(values[0]) - float(values[1]))


def search_move(search: dict | None) -> str | None:
    return (search or {}).get("bestmove")


def classify_searches(searches: Iterable[dict], *, margin_threshold: float = MARGIN_THRESHOLD_CP) -> str:
    searches = list(searches)
    moves = [search_move(s) for s in searches]
    if len(moves) < 3 or not moves[0] or moves[0] != moves[1]:
        return "UNSTABLE"
    if moves[2] != moves[0]:
        return "FALSE_STABLE"
    deep_info = (searches[2] or {}).get("info")
    deep_top = top_moves(deep_info)
    if len(deep_top) >= 2:
        first = score_value(deep_top[0].get("score"))
        second = score_value(deep_top[1].get("score"))
        if first is not None and second is not None and first < second:
            return "LOW_CONFIDENCE"
    margin = margin_cp(deep_info)
    if margin is not None and margin < margin_threshold:
        return "LOW_CONFIDENCE"
    return "TRIPLE_STABLE"


def phase_bucket(row: dict) -> str:
    phase = str(row.get("phase") or "").lower()
    if phase in {"opening", "middlegame", "end"}:
        return phase
    ply = int(row.get("ply") or 0)
    return "opening" if ply < 12 else "end" if ply >= 50 else "middlegame"


def wall_bucket(row: dict) -> str:
    count = row.get("wall_count")
    if count is None:
        count = (row.get("tension") or {}).get("wall_count")
    try:
        count = int(count)
    except (TypeError, ValueError):
        return "unknown"
    return "0-3" if count <= 3 else "4-7" if count <= 7 else "8-11" if count <= 11 else "12+"


def score_margin_bucket(row: dict) -> str:
    scores = row.get("scores") or []
    values = [score_value(v) for v in scores[:2]]
    margin = abs(values[1] - values[0]) if len(values) == 2 and all(v is not None for v in values) else None
    if margin is None:
        moves = top_moves(row)
        values = [score_value(m.get("score")) for m in moves[:2]]
        margin = abs(values[1] - values[0]) if len(values) == 2 and all(v is not None for v in values) else None
    if margin is None:
        return "unknown"
    return "0-49" if margin < 50 else "50-149" if margin < 150 else "150-299" if margin < 300 else "300+"


def strata_key(row: dict, fields: Iterable[str]) -> tuple[str, ...]:
    values = []
    for field in fields:
        if field == "move_type":
            values.append(move_type(row))
        elif field == "phase":
            values.append(phase_bucket(row))
        elif field == "wall_count":
            values.append(wall_bucket(row))
        elif field == "score_margin":
            values.append(score_margin_bucket(row))
        elif field == "paired_fork":
            values.append("fork" if row.get("titanium_best") != row.get("claustrophobia_action") else "aligned")
        else:
            values.append(str(row.get(field, "unknown")))
    return tuple(values)


def stratified_sample(rows: list[dict], size: int, fields: Iterable[str], seed: int = SEED) -> tuple[list[dict], dict[str, int]]:
    import random
    rng = random.Random(seed)
    groups: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for row in rows:
        groups[strata_key(row, fields)].append(row)
    keys = sorted(groups)
    for key in keys:
        rng.shuffle(groups[key])
    selected: list[dict] = []
    # Round-robin keeps small strata represented, then fills deterministically.
    while len(selected) < min(size, len(rows)):
        advanced = False
        for key in keys:
            if groups[key] and len(selected) < size:
                selected.append(groups[key].pop())
                advanced = True
        if not advanced:
            break
    counts = Counter(strata_key(row, fields) for row in selected)
    return selected, {"|".join(key): count for key, count in sorted(counts.items())}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
