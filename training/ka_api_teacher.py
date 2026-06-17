#!/usr/bin/env python3
"""Deprecated Ka single-position teacher wrapper.

This path is intentionally disabled. The default NNUE pipeline trains from
completed games only: compact move list + final WDL outcome. Single-position
Ka/CNN labels were too easy to misuse as an eval teacher and are no longer
accepted by `train.py`.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = ROOT / "training" / "data" / "ka_teacher_cache.jsonl"


def load_cache(path: Path | None = None) -> dict[str, dict]:
    _ = path
    return {}


def cache_stats(cache: dict[str, dict] | None = None) -> dict:
    _ = cache
    return {"positions": 0, "disabled": True}


def teacher_cp_for_prefix(moves_prefix: list[str], cache: dict[str, dict] | None = None) -> float | None:
    _ = moves_prefix, cache
    return None


def main() -> int:
    print("Ka single-position teacher labeling is deprecated and disabled.")
    print("Use compact completed games in training/data/all_games.db and WDL outcomes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
