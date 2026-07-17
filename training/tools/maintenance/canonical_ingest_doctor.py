"""Doctor for canonical game-ingest paths.

Checks only the active ingest lane:
  website/plain-text wire -> game_store.db
  website/plain-text wire -> oracle spool
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from titanium_training.store.config import DATA_DIR, GAME_STORE_DB, LEGACY_GAME_DB, LEGACY_GAME_JSONL


@dataclass(frozen=True)
class CheckResult:
    level: str
    message: str


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return bool(row)


def run() -> tuple[list[CheckResult], int]:
    results: list[CheckResult] = []
    failures = 0

    if not GAME_STORE_DB.is_file():
        results.append(CheckResult("FAIL", f"canonical game store missing: {GAME_STORE_DB}"))
        failures += 1
    else:
        conn = sqlite3.connect(str(GAME_STORE_DB))
        try:
            required = ("positions", "edges", "games", "game_paths")
            missing = [name for name in required if not _table_exists(conn, name)]
            if missing:
                results.append(
                    CheckResult(
                        "FAIL",
                        f"canonical game store missing required tables: {', '.join(missing)}",
                    )
                )
                failures += 1
            else:
                counts = {
                    name: int(conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0])
                    for name in ("positions", "games", "game_paths")
                }
                results.append(
                    CheckResult(
                        "PASS",
                        "canonical game store ready "
                        f"(positions={counts['positions']}, games={counts['games']}, paths={counts['game_paths']})",
                    )
                )
        finally:
            conn.close()

    legacy_pending = DATA_DIR / "website_finished_games" / "pending.jsonl"
    if legacy_pending.exists():
        results.append(
            CheckResult(
                "FAIL",
                f"legacy website JSONL inbox still present: {legacy_pending}",
            )
        )
        failures += 1
    else:
        results.append(CheckResult("PASS", "no legacy website pending.jsonl inbox"))

    if LEGACY_GAME_JSONL.exists():
        results.append(CheckResult("WARN", f"legacy game JSONL still present: {LEGACY_GAME_JSONL}"))
    else:
        results.append(CheckResult("PASS", "legacy all_games.jsonl not present"))

    if LEGACY_GAME_DB.exists():
        results.append(CheckResult("WARN", f"legacy game DB still present: {LEGACY_GAME_DB}"))
    else:
        results.append(CheckResult("PASS", "legacy all_games.db not present"))

    oracle_readme = Path("training/oracle_game_factory/README.md")
    if oracle_readme.is_file():
        text = oracle_readme.read_text(encoding="utf-8", errors="replace")
        if "/submit/website-game" in text:
            results.append(CheckResult("PASS", "oracle README documents website submit endpoint"))
        else:
            results.append(CheckResult("WARN", "oracle README missing website submit endpoint docs"))

    return results, failures


def main() -> int:
    results, failures = run()
    print("Canonical ingest doctor")
    for item in results:
        print(f"[{item.level}] {item.message}")
    if failures:
        print(f"FAILED ({failures} critical issue(s))")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
