#!/usr/bin/env python3
"""
Quoridor training data importer — two-database pipeline.

DATABASES (both in training/data/canonical/):
─────────────────────────────────────────────
  games.db    — every game collected as a DAG of positions + moves + outcomes.
                Games are stored incrementally: each batch of N games is committed
                immediately, so crashes don't lose work.  Re-running auto-skips
                already-imported game_ids (safe to resume).

  labels.db   — unique positions with per-source value labels for NNUE training.
                Each row is keyed by (pos_key, source).  Training resolves one
                target per position via label_resolution (tiered priority), never
                AVG(value_stm) across sources.

POSITION IDENTITY
  pos_key = first 16 hex chars of SHA-256 of the canonical eval-batch JSON
  (board state fields sorted, "eval" excluded).  Same board state from any
  move sequence → same JSON → same pos_key.

SOURCES
  wallz     — 139k human games from wallz.gg  (hard outcome labels)
  zeroink   — AlphaZero self-play collected from zero.ink  (soft NN value + outcome)

USAGE
  python training/db_import.py --status
  python training/db_import.py --wallz
  python training/db_import.py --zeroink
  python training/db_import.py --wallz --workers 3 --batch 500
  python training/db_import.py --wallz --limit 5000   # quick test
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

_TRAINING = Path(__file__).resolve().parent
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

from titanium_training.paths import ENGINE_BIN, REPO_ROOT

CANONICAL_DIR  = _TRAINING / "data" / "canonical"
GAMES_DB_PATH  = CANONICAL_DIR / "games.db"
LABELS_DB_PATH = CANONICAL_DIR / "labels.db"
WALLZ_GZ       = _TRAINING / "data" / "aditional imports" / "replays.jsonl.gz"
ZEROINK_DIR    = _TRAINING / "data" / "zeroink_games"

COL = "abcdefghi"   # column letters a-i


def aggregate_outcome_label(prior_values: list[float], new_value: float) -> float:
    """Running mean for duplicate ``pos_key`` rows (matches labels INSERT ON CONFLICT).

    When the same board key appears in games with different terminal outcomes,
    ``value_stm`` is the intentional fractional expected result — not a sign bug.
  """
    if not prior_values:
        return float(new_value)
    return (sum(float(v) for v in prior_values) + float(new_value)) / (
        len(prior_values) + 1
    )


# ─────────────────────────────────────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────────────────────────────────────

GAMES_SCHEMA = """
-- games.db: every imported game as a DAG of positions connected by moves.
-- To replay game G:
--   SELECT move_alg FROM game_moves WHERE game_id=G ORDER BY move_num

CREATE TABLE IF NOT EXISTS games (
    game_id     TEXT PRIMARY KEY,
    source      TEXT    NOT NULL,   -- 'wallz' | 'zeroink'
    outcome_p0  INTEGER NOT NULL,   -- +1=P0 wins (starts e1, goal row 9), -1=P1 wins
    move_count  INTEGER NOT NULL,
    imported_at TEXT    NOT NULL    -- ISO-8601 UTC
);

CREATE TABLE IF NOT EXISTS positions (
    pos_key       TEXT PRIMARY KEY,
    -- position_data: JSON from 'titanium eval-batch'.
    -- Contains board state fields (pawn cells, wall bitmaps, side_to_move, etc.).
    -- To get the 547-element feature vector: record_to_fv(json.loads(position_data), target)
    -- Same board state from any path → same JSON → same pos_key (content-deduplicated).
    position_data BLOB    NOT NULL,
    side_to_move  INTEGER NOT NULL  -- 0=P0 to move, 1=P1 to move
);

CREATE TABLE IF NOT EXISTS game_moves (
    game_id  TEXT    NOT NULL REFERENCES games(game_id),
    move_num INTEGER NOT NULL,   -- 0-indexed ply
    pos_key  TEXT    NOT NULL REFERENCES positions(pos_key),  -- position BEFORE this move
    move_alg TEXT    NOT NULL,   -- engine algebraic: 'e2' | 'a3h' | 'd5v'
    PRIMARY KEY (game_id, move_num)
);

CREATE TABLE IF NOT EXISTS game_line_hashes (
    line_hash     TEXT PRIMARY KEY,
    game_id       TEXT    NOT NULL REFERENCES games(game_id),
    move_count    INTEGER NOT NULL,
    first_seen_at TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_gm_pos ON game_moves(pos_key);
CREATE INDEX IF NOT EXISTS idx_glh_game ON game_line_hashes(game_id);

-- Hard invariant (2026-07-08): a `positions` row with no `game_moves` row
-- referencing it is an orphan with no way to ever recover which game/outcome
-- it belongs to -- confirmed 4.29M such orphans already exist from a past,
-- since-changed writer. This trigger makes a bare position insert physically
-- impossible going forward: write_batch() MUST insert game_moves before
-- positions for the same pos_key, or this fires and aborts the statement.
CREATE TRIGGER IF NOT EXISTS trg_positions_require_move
AFTER INSERT ON positions
WHEN NOT EXISTS (SELECT 1 FROM game_moves WHERE pos_key = NEW.pos_key)
BEGIN
    SELECT RAISE(ABORT, 'positions row inserted with no matching game_moves row -- insert game_moves before positions');
END;
"""

LABELS_SCHEMA = """
-- labels.db: positions + per-source value labels for NNUE training.
-- Multiple sources per position are kept as separate rows; training resolves
-- one target via label_resolution.py (tiered priority), never cross-source AVG.
-- value_stm is always from the current-player's perspective:
--   +1.0 = current player wins,  -1.0 = current player loses

CREATE TABLE IF NOT EXISTS positions (
    pos_key       TEXT PRIMARY KEY,
    position_data BLOB    NOT NULL,
    side_to_move  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS labels (
    pos_key   TEXT    NOT NULL REFERENCES positions(pos_key),
    -- source identifies dataset and label type:
    --   'wallz_outcome'   — hard win/loss from 139k human games
    --   'zeroink_nn'      — AlphaZero value estimate (soft, more informative)
    --   'zeroink_outcome' — hard win/loss from zero.ink self-play
    source    TEXT    NOT NULL,
    value_stm REAL    NOT NULL,   -- -1..+1, current-player perspective
    n_samples INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (pos_key, source)
);

CREATE INDEX IF NOT EXISTS idx_lbl_pos ON labels(pos_key);

-- Hard invariant (2026-07-08): mirrors trg_positions_require_move in
-- GAMES_SCHEMA -- a `positions` row with no `labels` row is unusable for
-- training and, per confirmed forensics, unrecoverable (no game history to
-- derive a label from after the fact). write_batch() MUST insert labels
-- before positions for the same pos_key, or this fires and aborts.
CREATE TRIGGER IF NOT EXISTS trg_positions_require_label
AFTER INSERT ON positions
WHEN NOT EXISTS (SELECT 1 FROM labels WHERE pos_key = NEW.pos_key)
BEGIN
    SELECT RAISE(ABORT, 'positions row inserted with no matching labels row -- insert labels before positions');
END;
"""


# ─────────────────────────────────────────────────────────────────────────────
# DB helpers
# ─────────────────────────────────────────────────────────────────────────────

def open_db(path: Path, schema: str) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, isolation_level=None, timeout=120)  # wait for stale WAL locks
    con.execute("PRAGMA journal_mode=WAL")   # allows concurrent readers + serialized writers
    con.execute("PRAGMA synchronous=NORMAL") # safe with WAL, faster than FULL
    con.executescript(schema)
    return con


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_pos_key(rec: dict) -> str:
    """Stable 16-char hex key from board state fields (path-independent)."""
    canonical = json.dumps(
        {k: rec[k] for k in sorted(rec) if k != "eval"},
        separators=(",", ":"), sort_keys=True,
    ).encode()
    return hashlib.sha256(canonical).hexdigest()[:16]


def make_pos_data(rec: dict) -> bytes:
    """Compact JSON bytes stored as position_data."""
    return json.dumps(
        {k: rec[k] for k in sorted(rec) if k != "eval"},
        separators=(",", ":"), sort_keys=True,
    ).encode()


def make_game_line_hash(moves: list[str] | tuple[str, ...]) -> str:
    h = hashlib.sha256()
    for move in moves:
        data = move.encode("ascii")
        h.update(len(data).to_bytes(2, "little"))
        h.update(data)
    return h.hexdigest()


def ensure_game_line_hashes(games_db: sqlite3.Connection) -> None:
    games_db.executescript(
        """
        CREATE TABLE IF NOT EXISTS game_line_hashes (
            line_hash     TEXT PRIMARY KEY,
            game_id       TEXT    NOT NULL REFERENCES games(game_id),
            move_count    INTEGER NOT NULL,
            first_seen_at TEXT    NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_glh_game ON game_line_hashes(game_id);
        """
    )
    existing = games_db.execute("SELECT COUNT(*) FROM game_line_hashes").fetchone()[0]
    if existing:
        return
    rows = games_db.execute(
        """
        SELECT g.game_id, g.move_count, g.imported_at, gm.move_alg
        FROM games g
        JOIN game_moves gm ON gm.game_id = g.game_id
        ORDER BY g.game_id, gm.move_num
        """
    )
    current_game = None
    current_imported_at = None
    moves: list[str] = []

    def flush() -> None:
        if current_game is None or not moves:
            return
        games_db.execute(
            "INSERT OR IGNORE INTO game_line_hashes VALUES (?,?,?,?)",
            (
                make_game_line_hash(moves),
                current_game,
                len(moves),
                current_imported_at or now_utc(),
            ),
        )

    for game_id, _move_count, imported_at, move_alg in rows:
        if current_game is not None and game_id != current_game:
            flush()
            moves = []
        current_game = game_id
        current_imported_at = imported_at
        moves.append(move_alg)
    flush()


# ─────────────────────────────────────────────────────────────────────────────
# Move notation converters
# ─────────────────────────────────────────────────────────────────────────────

def wallz_move_to_alg(move: dict) -> str | None:
    t = move.get("type")
    if t == "pawn":
        x, y = int(move["to"]["x"]), int(move["to"]["y"])
        return f"{COL[x]}{y + 1}" if 0 <= x <= 8 and 0 <= y <= 8 else None
    if t == "wall":
        w = move.get("wall", {})
        x, y, o = int(w.get("x", -1)), int(w.get("y", -1)), w.get("o", "")
        return f"{COL[x]}{y + 1}{o}" if 0 <= x <= 7 and 0 <= y <= 7 and o in ("h", "v") else None
    return None


def zeroink_move_to_alg(mc: dict) -> str | None:
    kind = mc.get("kind")
    if kind == "pawn":
        t = mc.get("target", -1)
        if t < 0: return None
        col, row = t % 9, t // 9
        return f"{COL[col]}{row + 1}"
    if kind == "wall":
        x, y = mc.get("x", -1), mc.get("y", -1)
        o = mc.get("orientation", "")
        o_char = "h" if o.startswith("h") else ("v" if o.startswith("v") else "")
        return f"{COL[x]}{y + 1}{o_char}" if 0 <= x <= 7 and 0 <= y <= 7 and o_char else None
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Engine eval-batch
# ─────────────────────────────────────────────────────────────────────────────

def eval_batch_chunk(lines: list[str]) -> list[dict | None]:
    """One call to `titanium eval-batch`: N move-sequence lines → N eval JSON dicts."""
    if not lines:
        return []
    try:
        proc = subprocess.run(
            [str(ENGINE_BIN), "eval-batch"],
            input=("\n".join(lines) + "\n").encode(),
            capture_output=True,
            cwd=str(REPO_ROOT),
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        return [None] * len(lines)
    if proc.returncode != 0:
        return [None] * len(lines)
    out_lines = [l for l in proc.stdout.decode(errors="replace").splitlines() if l.strip()]
    if len(out_lines) != len(lines):
        return [None] * len(lines)
    results = []
    for ln in out_lines:
        try:
            results.append(json.loads(ln))
        except Exception:
            results.append(None)
    return results


def eval_prefixes_parallel(
    prefixes: list[str],
    chunk_size: int,
    workers: int,
) -> dict[str, dict]:
    """Evaluate unique move-sequence prefixes in parallel → {prefix: eval_dict}."""
    if not prefixes:
        return {}
    chunks = [(i, prefixes[i : i + chunk_size]) for i in range(0, len(prefixes), chunk_size)]
    results: list[dict | None] = [None] * len(prefixes)

    def _run(item):
        start, chunk = item
        return start, eval_batch_chunk(chunk)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for fut in as_completed({pool.submit(_run, c): c for c in chunks}):
            start, recs = fut.result()
            for j, rec in enumerate(recs):
                results[start + j] = rec

    return {p: results[i] for i, p in enumerate(prefixes) if results[i] is not None}


# ─────────────────────────────────────────────────────────────────────────────
# Batch writer — processes one batch of games, commits, returns stats
# ─────────────────────────────────────────────────────────────────────────────

def write_batch(
    games_db: sqlite3.Connection,
    labels_db: sqlite3.Connection,
    batch: list[tuple],  # (game_id, moves, outcome_p0, nn_values|None, source)
    chunk_size: int,
    workers: int,
) -> tuple[int, int, int]:
    """Write one batch of games to both DBs. Returns (n_games, n_pos, n_labels)."""

    ensure_game_line_hashes(games_db)
    ts = now_utc()
    unique_batch: list[tuple] = []
    seen_hashes: set[str] = set()
    for game_id, moves, outcome_p0, nn_vals, source in batch:
        line_hash = make_game_line_hash(moves)
        if line_hash in seen_hashes:
            continue
        if games_db.execute(
            "SELECT 1 FROM game_line_hashes WHERE line_hash=?",
            (line_hash,),
        ).fetchone():
            continue
        seen_hashes.add(line_hash)
        unique_batch.append((game_id, moves, outcome_p0, nn_vals, source, line_hash))
    if not unique_batch:
        return 0, 0, 0

    # Collect unique prefixes for this batch
    all_prefixes: list[str] = []
    prefix_index: dict[str, int] = {}
    for game_id, moves, _o, _n, _s, _h in unique_batch:
        for k in range(len(moves)):
            p = " ".join(moves[:k])
            if p not in prefix_index:
                prefix_index[p] = len(all_prefixes)
                all_prefixes.append(p)

    prefix_to_rec = eval_prefixes_parallel(all_prefixes, chunk_size, workers)

    n_games = n_pos = n_labels = 0

    from position_usage_db import ensure_schema as ensure_usage_schema

    ensure_usage_schema(labels_db)
    games_db.execute("BEGIN")
    labels_db.execute("BEGIN")
    try:
        from label_resolution import merge_outcome_sample, stm_from_eval_cp

        for game_id, moves, outcome_p0, nn_vals, source, line_hash in unique_batch:
            pos_rows: list[tuple] = []
            move_rows: list[tuple] = []
            label_rows: list[tuple] = []
            ok = True

            for k, move_alg in enumerate(moves):
                prefix = " ".join(moves[:k])
                rec = prefix_to_rec.get(prefix)
                if rec is None:
                    ok = False; break

                stm = int(rec.get("turn", k % 2))
                key = make_pos_key(rec)
                data = make_pos_data(rec)

                pos_rows.append((key, data, stm))
                move_rows.append((game_id, k, key, move_alg))

                # Hard outcome (flipped to current-player perspective)
                outcome_stm = float(outcome_p0) if stm == 0 else float(-outcome_p0)
                label_rows.append((key, f"{source}_outcome", outcome_stm))

                # Soft NN value if available (already current-player perspective, -1..1)
                if nn_vals is not None and k < len(nn_vals):
                    label_rows.append((key, f"{source}_nn", float(nn_vals[k])))

                eval_cp = rec.get("eval")
                if eval_cp is not None:
                    try:
                        label_rows.append(
                            (key, f"{source}_engine", stm_from_eval_cp(float(eval_cp)))
                        )
                    except (TypeError, ValueError):
                        pass

            if not ok:
                continue

            games_db.execute(
                "INSERT OR IGNORE INTO games VALUES (?,?,?,?,?)",
                (game_id, source, outcome_p0, len(moves), ts),
            )
            games_db.execute(
                "INSERT OR IGNORE INTO game_line_hashes VALUES (?,?,?,?)",
                (line_hash, game_id, len(moves), ts),
            )
            games_db.executemany("INSERT OR IGNORE INTO game_moves VALUES (?,?,?,?)", move_rows)
            games_db.executemany("INSERT OR IGNORE INTO positions VALUES (?,?,?)", pos_rows)

            for pk, lsrc, val in label_rows:
                if lsrc.endswith("_outcome"):
                    row = labels_db.execute(
                        "SELECT value_stm, n_samples FROM labels WHERE pos_key=? AND source=?",
                        (pk, lsrc),
                    ).fetchone()
                    if row is not None:
                        val = merge_outcome_sample(row[0], row[1], val)
                    labels_db.execute(
                        """
                        INSERT INTO labels(pos_key, source, value_stm, n_samples) VALUES (?,?,?,1)
                        ON CONFLICT(pos_key, source) DO UPDATE SET
                          value_stm = (value_stm * n_samples + excluded.value_stm) / (n_samples + 1),
                          n_samples = n_samples + 1
                        """,
                        (pk, lsrc, val),
                    )
                elif lsrc.endswith("_engine"):
                    labels_db.execute(
                        """
                        INSERT INTO labels(pos_key, source, value_stm, n_samples) VALUES (?,?,?,1)
                        ON CONFLICT(pos_key, source) DO UPDATE SET
                          value_stm = excluded.value_stm,
                          n_samples = n_samples + 1
                        """,
                        (pk, lsrc, val),
                    )
                else:
                    labels_db.execute(
                        """
                        INSERT INTO labels(pos_key, source, value_stm, n_samples) VALUES (?,?,?,1)
                        ON CONFLICT(pos_key, source) DO UPDATE SET
                          value_stm = (value_stm * n_samples + excluded.value_stm) / (n_samples + 1),
                          n_samples = n_samples + 1
                        """,
                        (pk, lsrc, val),
                    )
            labels_db.executemany("INSERT OR IGNORE INTO positions VALUES (?,?,?)", pos_rows)

            n_games += 1
            n_pos += len(pos_rows)
            n_labels += len(label_rows)

        from position_usage_db import increment_new_eligible, upsert_positions

        batch_keys: list[str] = []
        for _game_id, moves, _outcome_p0, _nn_vals, _source, _line_hash in unique_batch:
            for k in range(len(moves)):
                prefix = " ".join(moves[:k])
                rec = prefix_to_rec.get(prefix)
                if rec is not None:
                    batch_keys.append("json:" + make_pos_key(rec))
        new_eligible = upsert_positions(labels_db, list(dict.fromkeys(batch_keys)))
        increment_new_eligible(labels_db, new_eligible)

        games_db.execute("COMMIT")
        labels_db.execute("COMMIT")
    except Exception:
        games_db.execute("ROLLBACK")
        labels_db.execute("ROLLBACK")
        raise

    return n_games, n_pos, n_labels


# ─────────────────────────────────────────────────────────────────────────────
# Wallz importer  (streaming, auto-resume, real-time DB commits)
# ─────────────────────────────────────────────────────────────────────────────

def import_wallz(
    games_db: sqlite3.Connection,
    labels_db: sqlite3.Connection,
    game_batch: int = 500,
    chunk_size: int = 8192,
    workers: int = 3,
    limit: int | None = None,
) -> None:
    print(f"[wallz] Reading {WALLZ_GZ} ...", flush=True)
    t_start = time.perf_counter()

    existing = set(
        r[0] for r in games_db.execute("SELECT game_id FROM games WHERE source='wallz'")
    )
    print(f"[wallz] {len(existing):,} games already in DB, skipping them", flush=True)

    new_games: list[tuple] = []
    n_read = n_bad = 0
    with gzip.open(WALLZ_GZ, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                n_bad += 1; continue
            p = obj.get("payload", {})
            if not p.get("winner") or not p.get("moves"):
                continue
            gid = f"wallz_{n_read:07d}"
            n_read += 1
            if limit and n_read > limit:
                break
            if gid in existing:
                continue
            moves = [wallz_move_to_alg(e.get("move", {})) for e in p["moves"]]
            if None in moves:
                n_bad += 1; continue
            outcome_p0 = 1 if p["winner"] == "p1" else -1
            new_games.append((gid, moves, outcome_p0, None, "wallz"))

    print(f"[wallz] {len(new_games):,} new games ({n_bad} skipped as bad)", flush=True)
    if not new_games:
        return

    n_batches = (len(new_games) + game_batch - 1) // game_batch
    total_g = total_p = total_l = 0

    for bi in range(n_batches):
        batch = new_games[bi * game_batch : (bi + 1) * game_batch]
        t0 = time.perf_counter()
        ng, np_, nl = write_batch(games_db, labels_db, batch, chunk_size, workers)
        dt = time.perf_counter() - t0
        total_g += ng; total_p += np_; total_l += nl
        done_games = (bi + 1) * game_batch
        pct = 100.0 * min(done_games, len(new_games)) / len(new_games)
        elapsed = time.perf_counter() - t_start
        eta = elapsed / max(pct / 100, 0.001) * (1 - pct / 100)
        print(
            f"[wallz] batch {bi+1}/{n_batches}  {pct:.1f}%"
            f"  +{ng}g +{np_}p +{nl}L  {dt:.0f}s  ETA {eta/60:.1f}min"
            f"  DB total: {total_g}g/{total_p}p/{total_l}L",
            flush=True,
        )

    print(f"[wallz] Done: {total_g:,}g  {total_p:,}p  {total_l:,}L", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Zero.ink importer
# ─────────────────────────────────────────────────────────────────────────────

def import_zeroink(
    games_db: sqlite3.Connection,
    labels_db: sqlite3.Connection,
    game_batch: int = 500,
    chunk_size: int = 8192,
    workers: int = 3,
) -> None:
    jsonl_files = sorted(ZEROINK_DIR.glob("*.jsonl"))
    if not jsonl_files:
        print(f"[zeroink] No JSONL files in {ZEROINK_DIR}"); return

    existing = set(
        r[0] for r in games_db.execute("SELECT game_id FROM games WHERE source='zeroink'")
    )

    all_records: list[dict] = []
    for f in jsonl_files:
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                all_records.append(json.loads(line))
            except Exception:
                pass

    game_map: dict[str, list[dict]] = {}
    for r in all_records:
        game_map.setdefault(r["game_id"], []).append(r)
    for recs in game_map.values():
        recs.sort(key=lambda r: r["move_num"])

    new_games: list[tuple] = []
    n_bad = 0
    for gid, recs in game_map.items():
        db_id = f"zeroink_{gid}"
        if db_id in existing:
            continue
        moves: list[str] = []
        nn_vals: list[float] = []
        ok = True
        for r in recs:
            alg = zeroink_move_to_alg(r.get("move_chosen", {}))
            if alg is None:
                ok = False; break
            moves.append(alg)
            nn_vals.append(float(r.get("value", 0.0)))
        if not ok or not moves:
            n_bad += 1; continue
        outcome = recs[0].get("game_outcome", {})
        outcome_p0 = 1 if outcome.get("winner") == 0 else -1
        new_games.append((db_id, moves, outcome_p0, nn_vals, "zeroink"))

    print(f"[zeroink] {len(new_games):,} new games  ({n_bad} bad)", flush=True)
    if not new_games:
        return

    n_batches = (len(new_games) + game_batch - 1) // game_batch
    total_g = total_p = total_l = 0
    t_start = time.perf_counter()

    for bi in range(n_batches):
        batch = new_games[bi * game_batch : (bi + 1) * game_batch]
        ng, np_, nl = write_batch(games_db, labels_db, batch, chunk_size, workers)
        total_g += ng; total_p += np_; total_l += nl
        pct = 100.0 * (bi + 1) * game_batch / len(new_games)
        print(f"[zeroink] {pct:.0f}%  {total_g}g/{total_p}p/{total_l}L", flush=True)

    print(f"[zeroink] Done: {total_g:,}g  {total_p:,}p  {total_l:,}L", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Status
# ─────────────────────────────────────────────────────────────────────────────

def print_status() -> None:
    for name, path in [("games.db", GAMES_DB_PATH), ("labels.db", LABELS_DB_PATH)]:
        if not path.exists():
            print(f"  {name}: not created yet"); continue
        con = sqlite3.connect(path)
        size_kb = path.stat().st_size // 1024
        print(f"\n{name}  ({size_kb:,} KB):")
        for tbl in ["games", "positions", "game_moves", "labels"]:
            try:
                cnt = con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
                print(f"  {tbl}: {cnt:,}")
            except Exception:
                pass
        try:
            for src, cnt in con.execute("SELECT source, COUNT(*) FROM games GROUP BY source"):
                print(f"    games[{src}]: {cnt:,}")
        except Exception:
            pass
        try:
            for src, cnt, avg in con.execute(
                "SELECT source, COUNT(*), AVG(value_stm) FROM labels GROUP BY source"
            ):
                print(f"    labels[{src}]: {cnt:,}  avg={avg:+.3f}")
        except Exception:
            pass
        con.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--wallz",   action="store_true")
    ap.add_argument("--zeroink", action="store_true")
    ap.add_argument("--status",  action="store_true")
    ap.add_argument("--limit",   type=int, default=None, help="Max games (for testing)")
    ap.add_argument("--workers", type=int, default=3,    help="Parallel eval-batch workers")
    ap.add_argument("--chunk",   type=int, default=8192, help="Lines per eval-batch call")
    ap.add_argument("--batch",   type=int, default=500,  help="Games per DB commit cycle")
    args = ap.parse_args()

    if args.status or not (args.wallz or args.zeroink):
        print_status()
        return 0

    from prep_guard import guard_real_work

    guard_real_work("labeling", detail="db_import")

    games_db  = open_db(GAMES_DB_PATH,  GAMES_SCHEMA)
    labels_db = open_db(LABELS_DB_PATH, LABELS_SCHEMA)

    if args.wallz:
        import_wallz(games_db, labels_db, args.batch, args.chunk, args.workers, args.limit)
    if args.zeroink:
        import_zeroink(games_db, labels_db, args.batch, args.chunk, args.workers)

    print_status()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
