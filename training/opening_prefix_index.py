"""Canonical opening-prefix index for novelty-aware self-play exploration."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from db_import import GAMES_DB_PATH, open_db

DEFAULT_INDEX_PATH = Path(__file__).resolve().parent / "data" / "canonical" / "opening_prefix_index.db"

_FILE_MIRROR = str.maketrans("abcdefghi", "ihgfedcba")

INDEX_SCHEMA = """
CREATE TABLE IF NOT EXISTS opening_prefixes (
    prefix_hash       BLOB PRIMARY KEY,
    prefix_len        INTEGER NOT NULL,
    canonical_moves   TEXT NOT NULL,
    occurrence_count  INTEGER NOT NULL DEFAULT 0,
    p0_wins           INTEGER NOT NULL DEFAULT 0,
    p1_wins           INTEGER NOT NULL DEFAULT 0,
    draws             INTEGER NOT NULL DEFAULT 0,
    last_seen_at      TEXT NOT NULL,
    last_source       TEXT
);
CREATE INDEX IF NOT EXISTS idx_opening_prefix_len ON opening_prefixes(prefix_len);
CREATE INDEX IF NOT EXISTS idx_opening_prefix_count ON opening_prefixes(occurrence_count DESC);

CREATE TABLE IF NOT EXISTS index_metadata (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def mirror_move_alg(move: str) -> str:
    """Horizontal board mirror: a↔i … e fixed."""
    if len(move) == 2:
        return move[0].translate(_FILE_MIRROR) + move[1]
    if len(move) == 3 and move[2] in ("h", "v"):
        return move[0].translate(_FILE_MIRROR) + move[1:]
    return move


def canonical_move_prefix(moves: list[str]) -> tuple[str, ...]:
    """Smallest lexicographic move tuple under horizontal mirror symmetry."""
    if not moves:
        return ()
    mirrored = tuple(mirror_move_alg(m) for m in moves)
    original = tuple(moves)
    return min(original, mirrored)


def prefix_hash(moves: list[str]) -> bytes:
    canon = canonical_move_prefix(moves)
    payload = "|".join(canon).encode("utf-8")
    return hashlib.sha256(payload).digest()


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class PrefixRecord:
    prefix_hash: bytes
    prefix_len: int
    canonical_moves: tuple[str, ...]
    occurrence_count: int
    p0_wins: int
    p1_wins: int
    draws: int
    last_seen_at: str
    last_source: str | None


@dataclass
class OpeningMetricsSnapshot:
    games_total: int = 0
    games_with_exploration: int = 0
    novel_prefix_games: int = 0
    novel_exit_plies: list[int] = field(default_factory=list)
    exploratory_moves: int = 0
    quality_cutoff_rejects: int = 0
    wins_exploratory_openings: int = 0
    losses_exploratory_openings: int = 0
    wins_non_exploratory: int = 0
    losses_non_exploratory: int = 0

    def to_dict(self) -> dict:
        novel_plies = self.novel_exit_plies
        median_novel_ply = None
        if novel_plies:
            s = sorted(novel_plies)
            mid = len(s) // 2
            median_novel_ply = s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2
        explored = self.games_with_exploration
        return {
            "games_total": self.games_total,
            "pct_novel_prefix": round(100 * self.novel_prefix_games / max(1, self.games_total), 2),
            "median_novel_exit_ply": median_novel_ply,
            "unique_novel_per_1000": round(1000 * self.novel_prefix_games / max(1, self.games_total), 2),
            "exploratory_moves": self.exploratory_moves,
            "quality_cutoff_reject_rate": round(
                self.quality_cutoff_rejects / max(1, self.exploratory_moves + self.quality_cutoff_rejects),
                4,
            ),
            "win_rate_exploratory_openings": round(
                self.wins_exploratory_openings / max(1, self.wins_exploratory_openings + self.losses_exploratory_openings),
                3,
            ),
            "win_rate_non_exploratory": round(
                self.wins_non_exploratory / max(1, self.wins_non_exploratory + self.losses_non_exploratory),
                3,
            ),
            "games_with_exploration": explored,
        }


class OpeningPrefixIndex:
    """Thread-safe prefix index backed by SQLite (WAL)."""

    def __init__(self, path: Path = DEFAULT_INDEX_PATH):
        self.path = path
        self._lock = threading.RLock()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(path), isolation_level=None, timeout=120, check_same_thread=False
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(INDEX_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def lookup(self, moves: list[str]) -> PrefixRecord | None:
        if not moves:
            return None
        key = prefix_hash(moves)
        with self._lock:
            row = self._conn.execute(
                "SELECT prefix_hash, prefix_len, canonical_moves, occurrence_count, "
                "p0_wins, p1_wins, draws, last_seen_at, last_source "
                "FROM opening_prefixes WHERE prefix_hash=?",
                (key,),
            ).fetchone()
        if not row:
            return None
        return PrefixRecord(
            prefix_hash=bytes(row[0]),
            prefix_len=int(row[1]),
            canonical_moves=tuple(json.loads(row[2])),
            occurrence_count=int(row[3]),
            p0_wins=int(row[4]),
            p1_wins=int(row[5]),
            draws=int(row[6]),
            last_seen_at=str(row[7]),
            last_source=row[8],
        )

    def is_known(self, moves: list[str]) -> bool:
        if not moves:
            return True
        return self.lookup(moves) is not None

    def occurrence_count(self, moves: list[str]) -> int:
        rec = self.lookup(moves)
        return rec.occurrence_count if rec else 0

    def register_game(
        self,
        moves: list[str],
        outcome_p0: int,
        *,
        source: str,
        max_ply: int = 16,
    ) -> list[tuple[str, ...]]:
        """Register all prefixes up to max_ply; return prefixes that were new before this game."""
        if not moves:
            return []
        novel_created: list[tuple[str, ...]] = []
        now = _now_utc()
        with self._lock:
            for n in range(1, min(len(moves), max_ply) + 1):
                prefix = moves[:n]
                canon = canonical_move_prefix(prefix)
                key = prefix_hash(prefix)
                existed = self._conn.execute(
                    "SELECT 1 FROM opening_prefixes WHERE prefix_hash=?",
                    (key,),
                ).fetchone()
                if not existed:
                    novel_created.append(canon)
                p0_w = 1 if outcome_p0 > 0 else 0
                p1_w = 1 if outcome_p0 < 0 else 0
                drw = 1 if outcome_p0 == 0 else 0
                self._conn.execute(
                    """
                    INSERT INTO opening_prefixes(
                        prefix_hash, prefix_len, canonical_moves, occurrence_count,
                        p0_wins, p1_wins, draws, last_seen_at, last_source
                    ) VALUES (?, ?, ?, 1, ?, ?, ?, ?, ?)
                    ON CONFLICT(prefix_hash) DO UPDATE SET
                        occurrence_count = occurrence_count + 1,
                        p0_wins = p0_wins + excluded.p0_wins,
                        p1_wins = p1_wins + excluded.p1_wins,
                        draws = draws + excluded.draws,
                        last_seen_at = excluded.last_seen_at,
                        last_source = excluded.last_source
                    """,
                    (key, n, json.dumps(canon), p0_w, p1_w, drw, now, source),
                )
            self._conn.execute(
                "INSERT INTO index_metadata(key, value) VALUES('last_register_at', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (now,),
            )
        return novel_created

    def frequency_distribution(self, *, max_ply: int = 16, limit: int = 20) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT prefix_len, canonical_moves, occurrence_count "
                "FROM opening_prefixes WHERE prefix_len <= ? "
                "ORDER BY occurrence_count DESC LIMIT ?",
                (max_ply, limit),
            ).fetchall()
        return [
            {"prefix_len": int(r[0]), "moves": json.loads(r[1]), "count": int(r[2])}
            for r in rows
        ]

    def total_prefixes(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM opening_prefixes").fetchone()
        return int(row[0]) if row else 0

    def build_from_games_db(
        self,
        games_db: Path = GAMES_DB_PATH,
        *,
        max_ply: int = 16,
        batch_log: int = 10_000,
    ) -> dict:
        """Bootstrap index from canonical games.db (idempotent upsert)."""
        if not games_db.is_file():
            return {"ok": False, "reason": f"missing {games_db}"}

        gconn = sqlite3.connect(str(games_db))
        gconn.row_factory = sqlite3.Row
        total_games = int(gconn.execute("SELECT COUNT(*) FROM games").fetchone()[0])
        processed = 0
        skipped_no_moves = 0
        skipped_empty = 0
        for row in gconn.execute(
            "SELECT game_id, source, outcome_p0 FROM games ORDER BY imported_at, game_id"
        ):
            moves = [
                r[0]
                for r in gconn.execute(
                    "SELECT move_alg FROM game_moves WHERE game_id=? ORDER BY move_num",
                    (row["game_id"],),
                )
            ]
            if not moves:
                skipped_no_moves += 1
                continue
            if not any(m.strip() for m in moves):
                skipped_empty += 1
                continue
            self.register_game(
                moves,
                int(row["outcome_p0"]),
                source=str(row["source"]),
                max_ply=max_ply,
            )
            processed += 1
            if batch_log and processed % batch_log == 0:
                print(f"  prefix index: {processed:,}/{total_games:,} games", flush=True)
        gconn.close()
        with self._lock:
            self._conn.execute(
                "INSERT INTO index_metadata(key, value) VALUES('built_from_games_db', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (_now_utc(),),
            )
        return {
            "ok": True,
            "games_total_in_db": total_games,
            "games_indexed": processed,
            "games_skipped_no_moves": skipped_no_moves,
            "games_skipped_empty_moves": skipped_empty,
            "prefixes": self.total_prefixes(),
        }

    def ensure_bootstrapped(
        self,
        games_db: Path = GAMES_DB_PATH,
        *,
        max_ply: int = 16,
        min_prefixes: int = 100,
        auto_build: bool = False,
    ) -> dict:
        """Return status; full corpus build only when auto_build=True (see --rebuild-prefix-index)."""
        n = self.total_prefixes()
        if n >= min_prefixes:
            return {"ok": True, "action": "skip", "prefixes": n}
        if not auto_build:
            return {
                "ok": True,
                "action": "empty",
                "prefixes": n,
                "message": (
                    "Prefix index sparse — novelty checks use indexed games only. "
                    "Run once: python training/opening_prefix_index.py --rebuild"
                ),
            }
        return {"ok": True, "action": "bootstrap", **self.build_from_games_db(games_db, max_ply=max_ply)}


def update_metrics(snapshot: OpeningMetricsSnapshot, game_result: dict) -> None:
    """Accumulate rolling metrics from a completed pool game dict."""
    snapshot.games_total += 1
    explored = int(game_result.get("explored_moves", 0) or 0)
    if explored > 0:
        snapshot.games_with_exploration += 1
    snapshot.exploratory_moves += explored
    snapshot.quality_cutoff_rejects += int(game_result.get("exploration_quality_rejects", 0) or 0)
    if game_result.get("novel_prefix"):
        snapshot.novel_prefix_games += 1
        exit_ply = game_result.get("novel_exit_ply")
        if exit_ply is not None:
            snapshot.novel_exit_plies.append(int(exit_ply))
            if len(snapshot.novel_exit_plies) > 5000:
                snapshot.novel_exit_plies = snapshot.novel_exit_plies[-2500:]
    if game_result.get("mixed"):
        return
    won = game_result.get("current_won")
    if won is None:
        return
    if explored > 0:
        if won:
            snapshot.wins_exploratory_openings += 1
        else:
            snapshot.losses_exploratory_openings += 1
    else:
        if won:
            snapshot.wins_non_exploratory += 1
        else:
            snapshot.losses_non_exploratory += 1


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Build or inspect canonical opening prefix index")
    ap.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    ap.add_argument("--rebuild", action="store_true", help="Scan games.db and upsert all prefixes")
    ap.add_argument("--max-ply", type=int, default=16)
    ap.add_argument("--top", type=int, default=20, help="Show top N frequent prefixes")
    args = ap.parse_args()
    idx = OpeningPrefixIndex(args.index)
    try:
        if args.rebuild:
            stats = idx.build_from_games_db(max_ply=args.max_ply)
            print(json.dumps(stats, indent=2))
        else:
            print(json.dumps({"prefixes": idx.total_prefixes()}, indent=2))
            for row in idx.frequency_distribution(max_ply=args.max_ply, limit=args.top):
                print(row)
    finally:
        idx.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
