"""SQLite-backed position usage tracking for database-first training.

Replaces cache-index-based position_usage.npy for the streaming trainer path.
Positions are retired from normal sampling after MAX_TRAINING_VISITS touches;
protected replay positions remain eligible.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

MAX_TRAINING_VISITS = 5
MAX_RETIRED_FRAC = 0.10

POSITION_USAGE_SCHEMA = """
CREATE TABLE IF NOT EXISTS position_usage (
    pos_key             TEXT PRIMARY KEY,
    training_visits     INTEGER NOT NULL DEFAULT 0,
    retired             INTEGER NOT NULL DEFAULT 0,
    protected_replay    INTEGER NOT NULL DEFAULT 0,
    source              TEXT NOT NULL DEFAULT 'canonical_json',
    retirement_reason   TEXT,
    last_trained_at     TEXT,
    first_seen_at       TEXT NOT NULL,
    observation_count   INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_usage_retired ON position_usage(retired);
CREATE INDEX IF NOT EXISTS idx_usage_visits ON position_usage(training_visits);

CREATE TABLE IF NOT EXISTS training_trigger_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    pending_new_eligible INTEGER NOT NULL DEFAULT 0,
    claimed_total INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL
);
"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_schema(con: sqlite3.Connection) -> None:
    con.executescript(POSITION_USAGE_SCHEMA)
    cols = {row[1] for row in con.execute("PRAGMA table_info(position_usage)").fetchall()}
    if "source" not in cols:
        con.execute("ALTER TABLE position_usage ADD COLUMN source TEXT NOT NULL DEFAULT 'canonical_json'")
    if "retired_replay_count" not in cols:
        con.execute("ALTER TABLE position_usage ADD COLUMN retired_replay_count INTEGER NOT NULL DEFAULT 0")


def upsert_positions(
    con: sqlite3.Connection,
    pos_keys: list[str],
    *,
    observation_delta: int = 1,
    source: str = "canonical_json",
    protected_replay: bool = False,
) -> int:
    """Register newly seen positions after a canonical game commit.

    Returns the number of new usage rows created.
    """
    if not pos_keys:
        return 0
    ts = _utc_now()
    unique = list(dict.fromkeys(pos_keys))
    placeholders = ",".join("?" * len(unique))
    existing = {
        str(row[0])
        for row in con.execute(
            f"SELECT pos_key FROM position_usage WHERE pos_key IN ({placeholders})",
            unique,
        ).fetchall()
    }
    for key in unique:
        con.execute(
            """
            INSERT INTO position_usage(pos_key, first_seen_at, observation_count, source, protected_replay)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(pos_key) DO UPDATE SET
                observation_count = observation_count + excluded.observation_count,
                source = CASE
                    WHEN position_usage.source = 'canonical_json' THEN excluded.source
                    ELSE position_usage.source
                END,
                protected_replay = MAX(position_usage.protected_replay, excluded.protected_replay)
            """,
            (key, ts, max(1, observation_delta), source, 1 if protected_replay else 0),
        )
    return len([key for key in unique if key not in existing])


def increment_new_eligible(con: sqlite3.Connection, count: int) -> int:
    """Increment the pending training trigger counter after commit-worthy positions arrive."""
    if count <= 0:
        return pending_new_eligible(con)
    ts = _utc_now()
    con.execute(
        """
        INSERT INTO training_trigger_state(id, pending_new_eligible, claimed_total, updated_at)
        VALUES (1, ?, 0, ?)
        ON CONFLICT(id) DO UPDATE SET
            pending_new_eligible = pending_new_eligible + excluded.pending_new_eligible,
            updated_at = excluded.updated_at
        """,
        (int(count), ts),
    )
    return pending_new_eligible(con)


def pending_new_eligible(con: sqlite3.Connection) -> int:
    row = con.execute(
        "SELECT pending_new_eligible FROM training_trigger_state WHERE id = 1"
    ).fetchone()
    return int(row[0]) if row else 0


def claim_training_trigger(con: sqlite3.Connection, threshold: int = 2048) -> dict:
    """Claim one trigger if enough positions are pending; preserve overflow."""
    ensure_schema(con)
    con.execute("BEGIN IMMEDIATE")
    try:
        pending = pending_new_eligible(con)
        if pending < threshold:
            con.execute("COMMIT")
            return {"claimed": False, "claimed_count": 0, "remaining": pending}
        remaining = pending - threshold
        ts = _utc_now()
        con.execute(
            """
            UPDATE training_trigger_state
            SET pending_new_eligible = ?,
                claimed_total = claimed_total + ?,
                updated_at = ?
            WHERE id = 1
            """,
            (remaining, threshold, ts),
        )
        con.execute("COMMIT")
        return {"claimed": True, "claimed_count": threshold, "remaining": remaining}
    except Exception:
        con.execute("ROLLBACK")
        raise


def claim_all_pending(con: sqlite3.Connection) -> dict:
    """Claim the entire current backlog (used by the games-count trigger, which
    fires on completed games rather than a fixed position slice -- once it
    fires, the cycle should train on everything that accumulated, not an
    arbitrary fixed-size cut of it)."""
    ensure_schema(con)
    con.execute("BEGIN IMMEDIATE")
    try:
        pending = pending_new_eligible(con)
        ts = _utc_now()
        con.execute(
            """
            UPDATE training_trigger_state
            SET pending_new_eligible = 0,
                claimed_total = claimed_total + ?,
                updated_at = ?
            WHERE id = 1
            """,
            (pending, ts),
        )
        con.execute("COMMIT")
        return {"claimed": pending > 0, "claimed_count": pending, "remaining": 0}
    except Exception:
        con.execute("ROLLBACK")
        raise


def release_pending_claim(con: sqlite3.Connection, claimed_count: int) -> int:
    """Return a failed training claim to the pending counter (snapshot not consumed)."""
    count = max(0, int(claimed_count))
    if count <= 0:
        return pending_new_eligible(con)
    ts = _utc_now()
    con.execute("BEGIN IMMEDIATE")
    try:
        con.execute(
            """
            INSERT INTO training_trigger_state(id, pending_new_eligible, claimed_total, updated_at)
            VALUES (1, ?, 0, ?)
            ON CONFLICT(id) DO UPDATE SET
                pending_new_eligible = pending_new_eligible + excluded.pending_new_eligible,
                updated_at = excluded.updated_at
            """,
            (count, ts),
        )
        pending = pending_new_eligible(con)
        con.execute("COMMIT")
        return pending
    except Exception:
        con.execute("ROLLBACK")
        raise


def _resolve_pos_key(con: sqlite3.Connection, key: str) -> str | None:
    """Match a sampled key to its actual position_usage row.

    streaming_db_loader.sample_epoch_keys() normalizes every key it returns to
    a 'json:'-prefixed form (via _normalize_usage_key) so it round-trips
    through LabelsRepository.load_labeled_positions(), which expects that
    prefix. But position_usage.pos_key for canonical_json-sourced rows is
    stored bare (no prefix) -- confirmed live, 2026-07-05: bumping
    pos_key='json:<hash>' directly either misses entirely, or (worse) silently
    matches an unrelated row that happens to share the same hash under
    source='opening_sanity' (its own, separate usage-tracking row for the same
    underlying position) -- so the real bare-keyed canonical_json row the
    fresh-sample SELECT actually filtered on as training_visits=0 never gets
    bumped, and the same lowest-rowid "never trained" positions get resampled
    every cycle forever. The 'json:' prefix is purely an artifact for
    addressing the positions/labels tables (LabelsRepository) and was never a
    position_usage convention for canonical_json rows, so strip it
    unconditionally rather than trying the prefixed form first.
    """
    bare = key[5:] if key.startswith("json:") else key
    row = con.execute("SELECT 1 FROM position_usage WHERE pos_key = ?", (bare,)).fetchone()
    if row is not None:
        return bare
    row = con.execute("SELECT 1 FROM position_usage WHERE pos_key = ?", (key,)).fetchone()
    if row is not None:
        return key
    return None


def bump_training_visits(con: sqlite3.Connection, pos_keys: list[str]) -> dict:
    """Increment training_visits for sampled positions; auto-retire at threshold."""
    if not pos_keys:
        return {"touched": 0, "retired_total": 0, "active": 0}
    ensure_schema(con)
    ts = _utc_now()
    unique = list(dict.fromkeys(pos_keys))
    total = con.execute("SELECT COUNT(*) FROM position_usage").fetchone()[0]
    retired_now = con.execute(
        "SELECT COUNT(*) FROM position_usage WHERE retired = 1 AND protected_replay = 0"
    ).fetchone()[0]
    cap_skips = 0
    unresolved = 0
    for raw_key in unique:
        key = _resolve_pos_key(con, raw_key)
        if key is None:
            unresolved += 1
            continue
        row = con.execute(
            "SELECT training_visits, protected_replay, retired FROM position_usage WHERE pos_key = ?",
            (key,),
        ).fetchone()
        if row is None:
            continue
        visits, protected, _retired = int(row[0]), int(row[1]), int(row[2])
        if protected:
            con.execute(
                """
                UPDATE position_usage
                SET training_visits = training_visits + 1, last_trained_at = ?
                WHERE pos_key = ?
                """,
                (ts, key),
            )
            continue
        if visits >= MAX_TRAINING_VISITS:
            continue
        if (
            visits == MAX_TRAINING_VISITS - 1
            and total > 0
            and (retired_now + 1) / total > MAX_RETIRED_FRAC
        ):
            cap_skips += 1
            continue
        con.execute(
            """
            UPDATE position_usage
            SET training_visits = training_visits + 1,
                last_trained_at = ?,
                retired = CASE
                    WHEN protected_replay = 1 THEN 0
                    WHEN training_visits + 1 >= ? THEN 1
                    ELSE retired
                END,
                retirement_reason = CASE
                    WHEN protected_replay = 1 THEN retirement_reason
                    WHEN training_visits + 1 >= ? THEN 'max_visits'
                    ELSE retirement_reason
                END
            WHERE pos_key = ?
            """,
            (ts, MAX_TRAINING_VISITS, MAX_TRAINING_VISITS, key),
        )
        if visits == MAX_TRAINING_VISITS - 1:
            retired_now += 1
    retired = con.execute(
        "SELECT COUNT(*) FROM position_usage WHERE retired = 1 AND protected_replay = 0"
    ).fetchone()[0]
    active = con.execute(
        """
        SELECT COUNT(*) FROM position_usage
        WHERE retired = 0 OR protected_replay = 1
        """
    ).fetchone()[0]
    return {
        "touched": len(unique) - unresolved,
        "unresolved": unresolved,
        "retired_total": int(retired),
        "retired_frac": round(int(retired) / max(int(total), 1), 4),
        "retirement_cap_skips": cap_skips,
        "active": int(active),
        "max_retired_frac": MAX_RETIRED_FRAC,
    }


def bump_retired_replay(con: sqlite3.Connection, pos_keys: list[str]) -> int:
    if not pos_keys:
        return 0
    unique = list(dict.fromkeys(pos_keys))
    for key in unique:
        con.execute(
            "UPDATE position_usage SET retired_replay_count = retired_replay_count + 1 WHERE pos_key = ?",
            (key,),
        )
    return len(unique)


def commit_epoch_training_visits(con: sqlite3.Connection, pos_keys: list[str]) -> dict:
    """Atomic epoch-end commit of training visits (after successful checkpoint save)."""
    con.execute("BEGIN IMMEDIATE")
    try:
        stats = bump_training_visits(con, pos_keys)
        con.commit()
        return stats
    except Exception:
        con.rollback()
        raise


def count_eligible(con: sqlite3.Connection) -> int:
    ensure_schema(con)
    row = con.execute(
        """
        SELECT COUNT(*) FROM position_usage
        WHERE retired = 0 OR protected_replay = 1
        """
    ).fetchone()
    return int(row[0]) if row else 0


def status(con: sqlite3.Connection) -> dict:
    ensure_schema(con)
    total = con.execute("SELECT COUNT(*) FROM position_usage").fetchone()[0]
    retired = con.execute(
        "SELECT COUNT(*) FROM position_usage WHERE retired = 1 AND protected_replay = 0"
    ).fetchone()[0]
    active = con.execute(
        "SELECT COUNT(*) FROM position_usage WHERE retired = 0 OR protected_replay = 1"
    ).fetchone()[0]
    protected = con.execute(
        "SELECT COUNT(*) FROM position_usage WHERE protected_replay = 1"
    ).fetchone()[0]
    return {
        "total": int(total),
        "retired": int(retired),
        "active": int(active),
        "protected_replay": int(protected),
        "max_training_visits": MAX_TRAINING_VISITS,
    }


def open_labels_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path), timeout=120)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    ensure_schema(con)
    return con


def backfill_from_labels(con: sqlite3.Connection) -> int:
    """Populate position_usage rows for labeled positions not yet tracked."""
    ensure_schema(con)
    rows = con.execute(
        """
        SELECT 'json:' || p.pos_key, COALESCE(SUM(l.n_samples), 1)
        FROM positions p
        JOIN labels l ON l.pos_key = p.pos_key
        LEFT JOIN position_usage u ON u.pos_key = 'json:' || p.pos_key
        WHERE u.pos_key IS NULL
        GROUP BY p.pos_key
        """
    ).fetchall()
    if not rows:
        return 0
    ts = _utc_now()
    con.executemany(
        """
        INSERT INTO position_usage(pos_key, first_seen_at, observation_count)
        VALUES (?, ?, ?)
        ON CONFLICT(pos_key) DO NOTHING
        """,
        [(str(pos_key), ts, int(obs)) for pos_key, obs in rows],
    )
    return len(rows)
