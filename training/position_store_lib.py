from __future__ import annotations

import base64
import contextlib
import hashlib
import io
import json
import os
import sqlite3
import struct
import tempfile
import time
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from position_store_state import (
    MOVE_SCHEMA_VERSION,
    POSITION_SCHEMA_VERSION,
    PositionState,
    WALL_SLOT_COUNT,
    apply_move,
    cell_to_notation,
    decode_move,
    encode_move,
    iter_wall_slots,
    moves_from_u8_blob,
    moves_to_u8_blob,
    replay_game,
)

ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = ROOT / "training"
DATA_DIR = TRAINING_DIR / "data"
DEFAULT_DB_PATH = DATA_DIR / "position_graph.db"
DEFAULT_REPORT_DIR = DATA_DIR / "position_store_reports"

DATABASE_SCHEMA_VERSION = 1
LABEL_SCHEMA_VERSION = 1
SHARD_FORMAT_VERSION = 1

SHARD_MAGIC = b"TIQSHRD1"
SHARD_TRAILER_MAGIC = b"TIQEND1!"

IMPORTABLE_SUFFIXES = {".db", ".jsonl", ".games", ".zip"}
INVENTORY_SUFFIXES = IMPORTABLE_SUFFIXES | {".json", ".txt", ".md"}


SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS positions (
    position_id INTEGER PRIMARY KEY,
    canonical_hash BLOB NOT NULL,
    fast_hash INTEGER NOT NULL,
    packed_state BLOB NOT NULL,
    side_to_move INTEGER NOT NULL,
    ply_min_seen INTEGER,
    ply_max_seen INTEGER,
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    total_visits INTEGER NOT NULL DEFAULT 0,
    source_flags INTEGER NOT NULL DEFAULT 0,
    schema_version INTEGER NOT NULL,
    UNIQUE(canonical_hash, packed_state)
);

CREATE TABLE IF NOT EXISTS edges (
    parent_position_id INTEGER NOT NULL REFERENCES positions(position_id),
    move_code_u8 INTEGER NOT NULL,
    child_position_id INTEGER NOT NULL REFERENCES positions(position_id),
    visit_count INTEGER NOT NULL DEFAULT 0,
    p0_win_count INTEGER NOT NULL DEFAULT 0,
    p1_win_count INTEGER NOT NULL DEFAULT 0,
    draw_count INTEGER NOT NULL DEFAULT 0,
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    UNIQUE(parent_position_id, move_code_u8, child_position_id)
);

CREATE TABLE IF NOT EXISTS games (
    game_id INTEGER PRIMARY KEY,
    start_position_id INTEGER NOT NULL REFERENCES positions(position_id),
    result INTEGER,
    move_count INTEGER NOT NULL,
    generator_engine_hash TEXT,
    generator_trunk_hash TEXT,
    search_config_hash TEXT,
    random_seed TEXT,
    worker_id TEXT,
    source TEXT NOT NULL,
    created_at TEXT NOT NULL,
    shard_id TEXT,
    game_metadata TEXT
);

CREATE TABLE IF NOT EXISTS game_paths (
    game_id INTEGER PRIMARY KEY REFERENCES games(game_id),
    packed_u8_move_sequence BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS labels (
    label_id INTEGER PRIMARY KEY,
    position_id INTEGER NOT NULL REFERENCES positions(position_id),
    label_type TEXT NOT NULL,
    value REAL,
    score REAL,
    bound TEXT,
    best_move_u8 INTEGER,
    nodes INTEGER,
    completed_depth INTEGER,
    selective_depth INTEGER,
    is_proven INTEGER NOT NULL DEFAULT 0,
    engine_hash TEXT,
    trunk_hash TEXT,
    search_config_hash TEXT,
    label_schema_version INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    quality_rank INTEGER NOT NULL DEFAULT 0,
    source TEXT NOT NULL,
    payload_json TEXT
);

CREATE TABLE IF NOT EXISTS observations (
    position_id INTEGER NOT NULL REFERENCES positions(position_id),
    source_cohort TEXT NOT NULL,
    visit_count INTEGER NOT NULL DEFAULT 0,
    p0_wins INTEGER NOT NULL DEFAULT 0,
    p1_wins INTEGER NOT NULL DEFAULT 0,
    draws INTEGER NOT NULL DEFAULT 0,
    evaluation_summary TEXT,
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    UNIQUE(position_id, source_cohort)
);

CREATE TABLE IF NOT EXISTS relabel_queue (
    queue_id INTEGER PRIMARY KEY,
    position_id INTEGER NOT NULL REFERENCES positions(position_id),
    requested_label_type TEXT NOT NULL,
    requested_node_budget INTEGER,
    priority INTEGER NOT NULL DEFAULT 0,
    reason TEXT NOT NULL,
    required_engine_hash TEXT,
    required_trunk_hash TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    attempt_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS imports (
    import_id INTEGER PRIMARY KEY,
    source_path TEXT NOT NULL,
    source_hash TEXT NOT NULL,
    format TEXT NOT NULL,
    record_count INTEGER NOT NULL DEFAULT 0,
    accepted_count INTEGER NOT NULL DEFAULT 0,
    rejected_count INTEGER NOT NULL DEFAULT 0,
    duplicate_count INTEGER NOT NULL DEFAULT 0,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    importer_version TEXT NOT NULL,
    status TEXT NOT NULL,
    error_report_path TEXT,
    UNIQUE(source_hash, format)
);

CREATE INDEX IF NOT EXISTS idx_positions_fast_hash ON positions(fast_hash);
CREATE INDEX IF NOT EXISTS idx_edges_parent ON edges(parent_position_id);
CREATE INDEX IF NOT EXISTS idx_edges_child ON edges(child_position_id);
CREATE INDEX IF NOT EXISTS idx_games_source ON games(source);
CREATE INDEX IF NOT EXISTS idx_labels_position_type ON labels(position_id, label_type, trunk_hash, engine_hash);
CREATE INDEX IF NOT EXISTS idx_relabel_status_priority ON relabel_queue(status, priority DESC, created_at);
CREATE INDEX IF NOT EXISTS idx_observations_source ON observations(source_cohort, visit_count DESC);
"""


@dataclass
class InventoryRecord:
    path: str
    format: str
    record_count: int | None
    estimated_unique_positions: int | None
    category: str
    labels_exist: bool
    label_type: str | None
    engine_or_network: str | None
    schema_version: str | None
    parse_confidence: str
    migration_status: str
    notes: str


@dataclass
class ImportStats:
    source_path: str
    source_format: str
    record_count: int = 0
    accepted_count: int = 0
    rejected_count: int = 0
    duplicate_count: int = 0
    errors: list[str] | None = None

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []


@dataclass(frozen=True)
class AlphaZeroPositionLabel:
    state: PositionState
    policy_actions: tuple[int, ...]
    policy_values: tuple[float, ...]
    outcome: float | None
    root_value: float | None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def importer_version() -> str:
    return f"position-store-v{DATABASE_SCHEMA_VERSION}"


def sqlite_i64(value: int) -> int:
    value &= (1 << 64) - 1
    if value >= (1 << 63):
        value -= 1 << 64
    return value


def connect_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA_SQL)
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-32768")
    return conn


def init_db(path: Path = DEFAULT_DB_PATH) -> None:
    conn = connect_db(path)
    conn.close()


def begin_import(conn: sqlite3.Connection, source_path: str, source_hash: str, source_format: str) -> int:
    row = conn.execute(
        "SELECT import_id, status FROM imports WHERE source_hash=? AND format=?",
        (source_hash, source_format),
    ).fetchone()
    if row and row["status"] == "completed":
        raise FileExistsError(f"source already imported: {source_path}")
    now = utc_now()
    if row:
        conn.execute(
            "UPDATE imports SET source_path=?, started_at=?, completed_at=NULL, status='running',"
            "record_count=0, accepted_count=0, rejected_count=0, duplicate_count=0, error_report_path=NULL "
            "WHERE import_id=?",
            (source_path, now, row["import_id"]),
        )
        return int(row["import_id"])
    cur = conn.execute(
        "INSERT INTO imports(source_path, source_hash, format, started_at, importer_version, status) "
        "VALUES(?, ?, ?, ?, ?, 'running')",
        (source_path, source_hash, source_format, now, importer_version()),
    )
    return int(cur.lastrowid)


def finish_import(conn: sqlite3.Connection, import_id: int, stats: ImportStats, status: str, error_report_path: str | None = None) -> None:
    conn.execute(
        "UPDATE imports SET record_count=?, accepted_count=?, rejected_count=?, duplicate_count=?, "
        "completed_at=?, status=?, error_report_path=? WHERE import_id=?",
        (
            stats.record_count,
            stats.accepted_count,
            stats.rejected_count,
            stats.duplicate_count,
            utc_now(),
            status,
            error_report_path,
            import_id,
        ),
    )


def ensure_position(
    conn: sqlite3.Connection,
    state: PositionState,
    *,
    ply: int | None,
    source_flag: int = 0,
    source_cohort: str | None = None,
) -> tuple[int, bool]:
    state.validate()
    packed = state.packed_state()
    canonical_hash = state.canonical_hash()
    now = utc_now()
    row = conn.execute(
        "SELECT position_id, total_visits, ply_min_seen, ply_max_seen FROM positions "
        "WHERE canonical_hash=? AND packed_state=?",
        (canonical_hash, packed),
    ).fetchone()
    if row:
        ply_min = row["ply_min_seen"] if row["ply_min_seen"] is not None else ply
        ply_max = row["ply_max_seen"] if row["ply_max_seen"] is not None else ply
        if ply is not None:
            ply_min = ply if ply_min is None else min(int(ply_min), ply)
            ply_max = ply if ply_max is None else max(int(ply_max), ply)
        conn.execute(
            "UPDATE positions SET total_visits=total_visits+1, last_seen_at=?, ply_min_seen=?, ply_max_seen=?, "
            "source_flags=source_flags | ? WHERE position_id=?",
            (now, ply_min, ply_max, source_flag, row["position_id"]),
        )
        if source_cohort:
            bump_observation(conn, int(row["position_id"]), source_cohort, None)
        return int(row["position_id"]), False
    cur = conn.execute(
        "INSERT INTO positions(canonical_hash, fast_hash, packed_state, side_to_move, ply_min_seen, ply_max_seen, "
        "first_seen_at, last_seen_at, total_visits, source_flags, schema_version) "
        "VALUES(?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)",
        (
            canonical_hash,
            sqlite_i64(state.fast_hash()),
            packed,
            state.side_to_move,
            ply,
            ply,
            now,
            now,
            source_flag,
            POSITION_SCHEMA_VERSION,
        ),
    )
    position_id = int(cur.lastrowid)
    if source_cohort:
        bump_observation(conn, position_id, source_cohort, None)
    return position_id, True


def bump_edge(
    conn: sqlite3.Connection,
    parent_id: int,
    move_code: int,
    child_id: int,
    result: int | None,
) -> None:
    now = utc_now()
    row = conn.execute(
        "SELECT child_position_id FROM edges WHERE parent_position_id=? AND move_code_u8=?",
        (parent_id, move_code),
    ).fetchone()
    if row and int(row["child_position_id"]) != child_id:
        raise ValueError(
            f"edge determinism violation: parent={parent_id} move={move_code} "
            f"existing child={row['child_position_id']} new child={child_id}"
        )
    p0 = 1 if result == 1 else 0
    p1 = 1 if result == -1 else 0
    draw = 1 if result == 0 else 0
    cur = conn.execute(
        "UPDATE edges SET visit_count=visit_count+1, p0_win_count=p0_win_count+?, "
        "p1_win_count=p1_win_count+?, draw_count=draw_count+?, last_seen_at=? "
        "WHERE parent_position_id=? AND move_code_u8=? AND child_position_id=?",
        (p0, p1, draw, now, parent_id, move_code, child_id),
    )
    if int(cur.rowcount or 0) == 0:
        conn.execute(
            "INSERT INTO edges(parent_position_id, move_code_u8, child_position_id, visit_count, "
            "p0_win_count, p1_win_count, draw_count, first_seen_at, last_seen_at) "
            "VALUES(?, ?, ?, 1, ?, ?, ?, ?, ?)",
            (parent_id, move_code, child_id, p0, p1, draw, now, now),
        )


def bump_observation(
    conn: sqlite3.Connection,
    position_id: int,
    source_cohort: str,
    result: int | None,
    *,
    evaluation_value: float | None = None,
) -> None:
    now = utc_now()
    row = conn.execute(
        "SELECT visit_count, p0_wins, p1_wins, draws, evaluation_summary FROM observations "
        "WHERE position_id=? AND source_cohort=?",
        (position_id, source_cohort),
    ).fetchone()
    p0 = 1 if result == 1 else 0
    p1 = 1 if result == -1 else 0
    draw = 1 if result == 0 else 0
    eval_summary = None
    if evaluation_value is not None:
        eval_summary = json_dumps({"last_value": evaluation_value})
    if row:
        merged_summary = row["evaluation_summary"] or eval_summary
        conn.execute(
            "UPDATE observations SET visit_count=visit_count+1, p0_wins=p0_wins+?, p1_wins=p1_wins+?, "
            "draws=draws+?, last_seen=?, evaluation_summary=? WHERE position_id=? AND source_cohort=?",
            (p0, p1, draw, now, merged_summary, position_id, source_cohort),
        )
        return
    conn.execute(
        "INSERT INTO observations(position_id, source_cohort, visit_count, p0_wins, p1_wins, draws, "
        "evaluation_summary, first_seen, last_seen) VALUES(?, ?, 1, ?, ?, ?, ?, ?, ?)",
        (position_id, source_cohort, p0, p1, draw, eval_summary, now, now),
    )


def insert_game(
    conn: sqlite3.Connection,
    moves: list[str],
    result: int | None,
    *,
    source: str,
    metadata: dict[str, Any] | None = None,
    source_cohort: str | None = None,
) -> int:
    states = replay_game(moves)
    position_ids: list[int] = []
    for ply, state in enumerate(states):
        pid, _ = ensure_position(conn, state, ply=ply, source_cohort=source_cohort)
        position_ids.append(pid)
        bump_observation(conn, pid, source_cohort or source, result)
    move_blob = moves_to_u8_blob(moves)
    for idx, move_code in enumerate(move_blob):
        bump_edge(conn, position_ids[idx], move_code, position_ids[idx + 1], result)
    cur = conn.execute(
        "INSERT INTO games(start_position_id, result, move_count, source, created_at, game_metadata) "
        "VALUES(?, ?, ?, ?, ?, ?)",
        (
            position_ids[0],
            result,
            len(moves),
            source,
            utc_now(),
            json_dumps(metadata or {}),
        ),
    )
    game_id = int(cur.lastrowid)
    conn.execute(
        "INSERT INTO game_paths(game_id, packed_u8_move_sequence) VALUES(?, ?)",
        (game_id, move_blob),
    )
    return game_id


def add_label(
    conn: sqlite3.Connection,
    position_id: int,
    *,
    label_type: str,
    source: str,
    value: float | None = None,
    score: float | None = None,
    bound: str | None = None,
    best_move_u8: int | None = None,
    nodes: int | None = None,
    completed_depth: int | None = None,
    selective_depth: int | None = None,
    is_proven: bool = False,
    engine_hash: str | None = None,
    trunk_hash: str | None = None,
    search_config_hash: str | None = None,
    quality_rank: int = 0,
    payload: dict[str, Any] | None = None,
) -> int:
    cur = conn.execute(
        "INSERT INTO labels(position_id, label_type, value, score, bound, best_move_u8, nodes, completed_depth, "
        "selective_depth, is_proven, engine_hash, trunk_hash, search_config_hash, label_schema_version, created_at, "
        "quality_rank, source, payload_json) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            position_id,
            label_type,
            value,
            score,
            bound,
            best_move_u8,
            nodes,
            completed_depth,
            selective_depth,
            1 if is_proven else 0,
            engine_hash,
            trunk_hash,
            search_config_hash,
            LABEL_SCHEMA_VERSION,
            utc_now(),
            quality_rank,
            source,
            json_dumps(payload or {}),
        ),
    )
    return int(cur.lastrowid)


def queue_relabel(
    conn: sqlite3.Connection,
    position_id: int,
    *,
    requested_label_type: str,
    priority: int,
    reason: str,
    requested_node_budget: int | None = None,
    required_engine_hash: str | None = None,
    required_trunk_hash: str | None = None,
) -> None:
    now = utc_now()
    conn.execute(
        "INSERT INTO relabel_queue(position_id, requested_label_type, requested_node_budget, priority, reason, "
        "required_engine_hash, required_trunk_hash, status, attempt_count, created_at, updated_at) "
        "VALUES(?, ?, ?, ?, ?, ?, ?, 'pending', 0, ?, ?)",
        (
            position_id,
            requested_label_type,
            requested_node_budget,
            priority,
            reason,
            required_engine_hash,
            required_trunk_hash,
            now,
            now,
        ),
    )


def parse_games_text(text: str) -> list[tuple[list[str], int | None]]:
    lines = text.splitlines()
    out: list[tuple[list[str], int | None]] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("GAME "):
            moves = line.split()[1:]
            result = None
            if i + 1 < len(lines):
                nxt = lines[i + 1].strip()
                if nxt.startswith("RESULT "):
                    token = nxt.split()[1]
                    if token == "W":
                        result = 1
                    elif token == "B":
                        result = -1
                    elif token in {"0", "D", "DRAW"}:
                        result = 0
                    i += 1
            out.append((moves, result))
        i += 1
    return out


def load_old_moves_from_row(moves_text: str | None, moves_bin: bytes | None) -> list[str]:
    from move_codec import moves_from_row

    return moves_from_row(moves_text, moves_bin)


def decode_moves_bin_base64(encoded: str) -> list[str]:
    from move_codec import unpack_moves

    return unpack_moves(base64.b64decode(encoded))


def jsonl_first_object(path: Path) -> dict[str, Any] | None:
    opener: Any
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line in handle:
            if line.strip():
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    return None
    return None


def classify_jsonl_object(obj: dict[str, Any]) -> tuple[str, str | None, bool]:
    schema = str(obj.get("schema", ""))
    if "state" in obj and ("policyActions" in obj or "policy" in obj):
        return "alpha-selfplay-jsonl", "policy+value", True
    if schema.startswith("zero-search-budget"):
        return "zero-search-budget-jsonl", "teacher_value+policy", True
    if schema.startswith("leaf-search-pressure"):
        return "search-pressure-jsonl", "search_pressure", True
    if schema.startswith("titanium-reduction-counterfactual"):
        return "reduction-counterfactual-jsonl", "reduction_counterfactual", True
    if {"turn", "pawn0", "pawn1", "wl0", "wl1"}.issubset(obj.keys()):
        return "expanded-position-jsonl", "game_result", True
    if {"moves", "score", "cp"}.issubset(obj.keys()):
        return "ka-cache-jsonl", "teacher_value", True
    return "jsonl-unknown", None, False


def inventory_scan(root: Path = ROOT) -> list[InventoryRecord]:
    candidates: list[Path] = []
    for base in (root / "training" / "data", root / "KaAiData", root / "site" / "benchmark" / "overnight"):
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file() and path.suffix.lower() in INVENTORY_SUFFIXES:
                candidates.append(path)
    candidates.sort()
    out: list[InventoryRecord] = []
    for path in candidates:
        rel = str(path.relative_to(root))
        suffix = path.suffix.lower()
        if suffix == ".db":
            try:
                conn = sqlite3.connect(str(path))
                row = conn.execute(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='games'"
                ).fetchone()
                has_games = bool(row and row[0])
                count = None
                notes = "sqlite"
                if has_games:
                    count = int(conn.execute("SELECT COUNT(*) FROM games").fetchone()[0])
                    fmt = "sqlite-games-v1"
                    category = "games"
                    labels = False
                    label_type = None
                else:
                    fmt = "sqlite-unknown"
                    category = "unknown"
                    labels = False
                    label_type = None
                conn.close()
                out.append(
                    InventoryRecord(
                        path=rel,
                        format=fmt,
                        record_count=count,
                        estimated_unique_positions=None,
                        category=category,
                        labels_exist=labels,
                        label_type=label_type,
                        engine_or_network=None,
                        schema_version=None,
                        parse_confidence="high",
                        migration_status="pending",
                        notes=notes,
                    )
                )
            except Exception as exc:
                out.append(
                    InventoryRecord(
                        path=rel,
                        format="sqlite-unparseable",
                        record_count=None,
                        estimated_unique_positions=None,
                        category="unknown",
                        labels_exist=False,
                        label_type=None,
                        engine_or_network=None,
                        schema_version=None,
                        parse_confidence="low",
                        migration_status="manual",
                        notes=str(exc),
                    )
                )
            continue
        if suffix == ".games":
            text = path.read_text(encoding="utf-8", errors="replace")
            games = parse_games_text(text)
            out.append(
                InventoryRecord(
                    path=rel,
                    format="games-text-v1",
                    record_count=len(games),
                    estimated_unique_positions=None,
                    category="games",
                    labels_exist=False,
                    label_type=None,
                    engine_or_network=None,
                    schema_version=None,
                    parse_confidence="high",
                    migration_status="pending",
                    notes="GAME/RESULT text",
                )
            )
            continue
        if suffix == ".jsonl":
            obj = jsonl_first_object(path)
            if obj is None:
                fmt, label_type, labels = ("jsonl-unparseable", None, False)
                confidence = "low"
            else:
                fmt, label_type, labels = classify_jsonl_object(obj)
                confidence = "high" if fmt != "jsonl-unknown" else "medium"
            line_count = 0
            with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
                for line in handle:
                    if line.strip():
                        line_count += 1
            category = "positions" if "jsonl" in fmt and fmt != "jsonl-unparseable" else "unknown"
            if fmt == "search-pressure-jsonl" or fmt == "reduction-counterfactual-jsonl":
                category = "positions"
            if fmt == "jsonl-unknown":
                category = "unknown"
            out.append(
                InventoryRecord(
                    path=rel,
                    format=fmt,
                    record_count=line_count,
                    estimated_unique_positions=None,
                    category=category,
                    labels_exist=labels,
                    label_type=label_type,
                    engine_or_network=obj.get("teacher") if obj else None,
                    schema_version=str(obj.get("schema")) if obj and "schema" in obj else None,
                    parse_confidence=confidence,
                    migration_status="pending" if fmt != "jsonl-unparseable" else "manual",
                    notes="jsonl",
                )
            )
            continue
        if suffix == ".zip" and "selfplay_iters_" in path.name:
            out.append(
                InventoryRecord(
                    path=rel,
                    format="alpha-selfplay-zip",
                    record_count=None,
                    estimated_unique_positions=None,
                    category="positions",
                    labels_exist=True,
                    label_type="policy+value",
                    engine_or_network="friend_selfplay",
                    schema_version=None,
                    parse_confidence="medium",
                    migration_status="pending",
                    notes="zip of JSONL self-play shards",
                )
            )
            continue
        if suffix in {".json", ".txt"} and "benchmark" in rel.lower():
            out.append(
                InventoryRecord(
                    path=rel,
                    format="benchmark-report",
                    record_count=1,
                    estimated_unique_positions=None,
                    category="games",
                    labels_exist=False,
                    label_type=None,
                    engine_or_network=None,
                    schema_version=None,
                    parse_confidence="medium",
                    migration_status="inventory-only",
                    notes="benchmark artifact",
                )
            )
    return out


def write_inventory_report(records: list[InventoryRecord], out_dir: Path = DEFAULT_REPORT_DIR) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = out_dir / f"inventory-{stamp}.json"
    md_path = out_dir / f"inventory-{stamp}.md"
    json_path.write_text(json.dumps([asdict(r) for r in records], indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Position Store Inventory",
        "",
        "| path | format | count | category | labels | label type | parse | migration | notes |",
        "|---|---:|---:|---|---|---|---|---|---|",
    ]
    for r in records:
        lines.append(
            f"| `{r.path}` | `{r.format}` | {r.record_count if r.record_count is not None else ''} | "
            f"{r.category} | {'yes' if r.labels_exist else 'no'} | {r.label_type or ''} | "
            f"{r.parse_confidence} | {r.migration_status} | {r.notes} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def _alpha_selfplay_sample_to_label(data: dict[str, Any]) -> AlphaZeroPositionLabel:
    state_obj = data["state"]
    state = PositionState(
        player0_cell=int(state_obj["player0Cell"]),
        player1_cell=int(state_obj["player1Cell"]),
        player0_walls=int(state_obj["player0Walls"]),
        player1_walls=int(state_obj["player1Walls"]),
        horizontal_walls=int(state_obj["horizontalWalls"]),
        vertical_walls=int(state_obj["verticalWalls"]),
        side_to_move=int(state_obj["currentPlayer"]),
    )
    state.validate()
    if "policyActions" in data and "policyValues" in data:
        actions = tuple(int(x) for x in data["policyActions"])
        values = tuple(float(x) for x in data["policyValues"])
    else:
        dense = data["policy"]
        actions_list: list[int] = []
        values_list: list[float] = []
        for idx, value in enumerate(dense):
            if float(value) > 0:
                actions_list.append(idx)
                values_list.append(float(value))
        actions = tuple(actions_list)
        values = tuple(values_list)
    outcome = data.get("outcome")
    root_value = data.get("rootValue")
    return AlphaZeroPositionLabel(
        state=state,
        policy_actions=actions,
        policy_values=values,
        outcome=float(outcome) if outcome is not None else None,
        root_value=float(root_value) if root_value is not None else None,
    )


def _alpha_action_to_move_u8(state: PositionState, action: int) -> int:
    if 0 <= action <= 80:
        move = cell_to_notation(action)
        return encode_move(state, move)
    if 81 <= action <= 144:
        slot = action - 81
        return slot
    if 145 <= action <= 208:
        slot = action - 145
        return 64 + slot
    raise ValueError(f"unsupported alpha action id: {action}")


def _compact_search_pressure_payload(obj: dict[str, Any]) -> dict[str, Any]:
    shallow = obj.get("shallow") or {}
    deep = obj.get("deep") or {}
    return {
        "schema": obj.get("schema", "leaf-search-pressure-v1"),
        "src": obj.get("src"),
        "source_game_key": obj.get("source_game_key"),
        "teacher": obj.get("teacher"),
        "ply": obj.get("ply"),
        "engine": obj.get("engine"),
        "outcome": obj.get("outcome"),
        "search_pressure": obj.get("search_pressure"),
        "target_components": obj.get("target_components"),
        "shallow": {
            "best": shallow.get("best"),
            "score": shallow.get("score"),
            "depth": shallow.get("depth"),
            "nodes": shallow.get("nodes"),
        },
        "deep": {
            "best": deep.get("best"),
            "score": deep.get("score"),
            "depth": deep.get("depth"),
            "nodes": deep.get("nodes"),
        },
    }


def _compact_zero_teacher_payload(obj: dict[str, Any]) -> dict[str, Any]:
    search = obj.get("search") or {}
    shallow = search.get("shallow") or {}
    deep = search.get("deep") or {}
    disagreement = search.get("disagreement") or {}
    stream_last = obj.get("stream_last") or {}
    return {
        "schema": obj.get("schema", "zero-search-budget-v2"),
        "src": obj.get("src"),
        "source_game_key": obj.get("source_game_key"),
        "teacher": obj.get("teacher"),
        "ply": obj.get("ply"),
        "settings": obj.get("settings"),
        "shallow": {
            "root_value": shallow.get("root_value"),
            "total_visits": shallow.get("total_visits"),
            "top_visit_fraction": shallow.get("top_visit_fraction"),
            "visit_entropy": shallow.get("visit_entropy"),
            "prior_visit_gap": shallow.get("prior_visit_gap"),
        },
        "deep": {
            "root_value": deep.get("root_value"),
            "total_visits": deep.get("total_visits"),
            "top_visit_fraction": deep.get("top_visit_fraction"),
            "visit_entropy": deep.get("visit_entropy"),
            "prior_visit_gap": deep.get("prior_visit_gap"),
        },
        "disagreement": disagreement,
        "stream_last": {
            "rootValue": stream_last.get("rootValue"),
            "totalVisits": stream_last.get("totalVisits"),
            "rootChildVisits": stream_last.get("rootChildVisits"),
            "depth": stream_last.get("depth"),
        },
    }


def _compact_reduction_payload(obj: dict[str, Any]) -> dict[str, Any]:
    keep = (
        "schema_version",
        "source",
        "proposal_source",
        "position_ply",
        "depth",
        "move",
        "move_index",
        "base_reduction",
        "move_class",
        "legal_move_bucket",
        "decision_preserved",
        "safe_plus_one_reduction",
        "verification_triggered",
        "baseline_nodes",
        "counterfactual_nodes",
        "net_nodes_saved",
        "net_savings_ratio",
        "sample_status",
        "activate_plus_one",
        "engine_commit",
        "trunk_sha256",
    )
    return {key: obj.get(key) for key in keep if key in obj}


def _compact_alpha_payload(
    sample: AlphaZeroPositionLabel,
    *,
    move_codes_u8: list[int],
    source_label: str,
) -> dict[str, Any]:
    return {
        "schema": "friend-selfplay-v1",
        "source": source_label,
        "policy_move_codes_u8": move_codes_u8,
        "policy_values": list(sample.policy_values),
        "outcome": sample.outcome,
        "root_value": sample.root_value,
    }


def import_all_games_db(
    conn: sqlite3.Connection,
    source_db: Path,
    *,
    dry_run: bool,
) -> ImportStats:
    stats = ImportStats(source_path=str(source_db), source_format="sqlite-games-v1")
    source = sqlite3.connect(str(source_db))
    source.row_factory = sqlite3.Row
    rows = source.execute(
        "SELECT g.id, g.moves, g.moves_bin, g.outcome, COALESCE(s.name, '') AS src "
        "FROM games g LEFT JOIN sources s ON s.id=g.src_id ORDER BY g.id"
    ).fetchall()
    for row in rows:
        stats.record_count += 1
        try:
            moves = load_old_moves_from_row(row["moves"], row["moves_bin"])
            insert_game(
                conn,
                moves,
                int(row["outcome"]) if row["outcome"] is not None else None,
                source=f"legacy-db:{row['src'] or 'unknown'}",
                metadata={"legacy_game_id": int(row["id"]), "legacy_source": row["src"]},
                source_cohort=row["src"] or "legacy-db",
            )
            stats.accepted_count += 1
        except Exception as exc:
            stats.rejected_count += 1
            stats.errors.append(f"games.id={row['id']}: {exc}")
    source.close()
    return stats


def import_games_file(
    conn: sqlite3.Connection,
    path: Path,
    *,
    dry_run: bool,
) -> ImportStats:
    stats = ImportStats(source_path=str(path), source_format="games-text-v1")
    records = parse_games_text(path.read_text(encoding="utf-8", errors="replace"))
    for idx, (moves, result) in enumerate(records, start=1):
        stats.record_count += 1
        try:
            insert_game(
                conn,
                moves,
                result,
                source=f"games-file:{path.name}",
                metadata={"record_index": idx, "source_path": str(path)},
                source_cohort=path.stem,
            )
            stats.accepted_count += 1
        except Exception as exc:
            stats.rejected_count += 1
            stats.errors.append(f"record {idx}: {exc}")
    return stats


def _position_from_moves_payload(obj: dict[str, Any]) -> tuple[PositionState, list[str]]:
    if obj.get("moves_bin"):
        moves = decode_moves_bin_base64(str(obj["moves_bin"]))
    elif obj.get("moves"):
        if isinstance(obj["moves"], list):
            moves = [str(x) for x in obj["moves"]]
        else:
            moves = str(obj["moves"]).split()
    else:
        raise ValueError("no moves or moves_bin present")
    states = replay_game(moves)
    return states[-1], moves


def import_search_pressure_jsonl(
    conn: sqlite3.Connection,
    path: Path,
    *,
    dry_run: bool,
) -> ImportStats:
    stats = ImportStats(source_path=str(path), source_format="search-pressure-jsonl")
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            stats.record_count += 1
            try:
                obj = json.loads(line)
                state, moves = _position_from_moves_payload(obj)
                pos_id, _ = ensure_position(conn, state, ply=obj.get("ply"), source_cohort=str(obj.get("src", "search-pressure")))
                add_label(
                    conn,
                    pos_id,
                    label_type="search_pressure",
                    source=str(obj.get("teacher", "titanium-native")),
                    value=float(obj.get("search_pressure")) if obj.get("search_pressure") is not None else None,
                    engine_hash=str(obj.get("engine")) if obj.get("engine") is not None else None,
                    payload=_compact_search_pressure_payload(obj),
                )
                stats.accepted_count += 1
            except Exception as exc:
                stats.rejected_count += 1
                stats.errors.append(f"line {line_no}: {exc}")
    return stats


def import_zero_teacher_jsonl(
    conn: sqlite3.Connection,
    path: Path,
    *,
    dry_run: bool,
) -> ImportStats:
    stats = ImportStats(source_path=str(path), source_format="zero-search-budget-jsonl")
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            stats.record_count += 1
            try:
                obj = json.loads(line)
                state, _moves = _position_from_moves_payload(obj)
                pos_id, _ = ensure_position(conn, state, ply=obj.get("ply"), source_cohort=str(obj.get("src", "zero-teacher")))
                best_move_u8 = None
                top_moves = obj.get("search", {}).get("deep", {}).get("top_moves") or obj.get("search", {}).get("top_moves")
                if top_moves:
                    top = top_moves[0]
                    move = top.get("move")
                    if isinstance(move, str):
                        best_move_u8 = encode_move(state, move)
                add_label(
                    conn,
                    pos_id,
                    label_type="teacher_value",
                    source=str(obj.get("teacher", "quoridor-zero.ink")),
                    value=float(obj.get("search", {}).get("deep", {}).get("root_value"))
                    if obj.get("search", {}).get("deep", {}).get("root_value") is not None
                    else float(obj.get("search", {}).get("root_value"))
                    if obj.get("search", {}).get("root_value") is not None
                    else None,
                    best_move_u8=best_move_u8,
                    payload=_compact_zero_teacher_payload(obj),
                )
                stats.accepted_count += 1
            except Exception as exc:
                stats.rejected_count += 1
                stats.errors.append(f"line {line_no}: {exc}")
    return stats


def import_reduction_counterfactual_jsonl(
    conn: sqlite3.Connection,
    path: Path,
    *,
    dry_run: bool,
) -> ImportStats:
    stats = ImportStats(source_path=str(path), source_format="reduction-counterfactual-jsonl")
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            stats.record_count += 1
            try:
                obj = json.loads(line)
                state, _ = _position_from_moves_payload(obj)
                pos_id, _ = ensure_position(conn, state, ply=obj.get("position_ply"), source_cohort=str(obj.get("source", "reduction-counterfactual")))
                best_move_u8 = encode_move(state, obj["move"]) if obj.get("move") else None
                add_label(
                    conn,
                    pos_id,
                    label_type="reduction_counterfactual",
                    source=str(obj.get("source", "reduction-counterfactual")),
                    value=1.0 if obj.get("activate_plus_one") else 0.0,
                    best_move_u8=best_move_u8,
                    nodes=int(obj.get("baseline_nodes")) if obj.get("baseline_nodes") is not None else None,
                    completed_depth=int(obj.get("depth")) if obj.get("depth") is not None else None,
                    engine_hash=obj.get("engine_commit"),
                    trunk_hash=obj.get("trunk_sha256"),
                    payload=_compact_reduction_payload(obj),
                )
                stats.accepted_count += 1
            except Exception as exc:
                stats.rejected_count += 1
                stats.errors.append(f"line {line_no}: {exc}")
    return stats


def import_alpha_selfplay_jsonl(
    conn: sqlite3.Connection,
    path: Path,
    *,
    dry_run: bool,
) -> ImportStats:
    stats = ImportStats(source_path=str(path), source_format="alpha-selfplay-jsonl")
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            stats.record_count += 1
            try:
                obj = json.loads(line)
                sample = _alpha_selfplay_sample_to_label(obj)
                pos_id, created = ensure_position(conn, sample.state, ply=None, source_cohort="friend_selfplay")
                best_move_u8 = None
                if sample.policy_actions:
                    best_idx = max(range(len(sample.policy_actions)), key=lambda i: sample.policy_values[i])
                    best_move_u8 = _alpha_action_to_move_u8(sample.state, sample.policy_actions[best_idx])
                move_codes_u8 = [_alpha_action_to_move_u8(sample.state, action) for action in sample.policy_actions]
                add_label(
                    conn,
                    pos_id,
                    label_type="teacher_value",
                    source="friend_selfplay",
                    value=sample.root_value,
                    best_move_u8=best_move_u8,
                    payload=_compact_alpha_payload(sample, move_codes_u8=move_codes_u8, source_label="friend_selfplay"),
                )
                result = None
                if sample.outcome is not None and sample.outcome in (-1.0, 0.0, 1.0):
                    result = int(sample.outcome)
                bump_observation(conn, pos_id, "friend_selfplay", result, evaluation_value=sample.root_value)
                if created and best_move_u8 is None:
                    queue_relabel(
                        conn,
                        pos_id,
                        requested_label_type="value_label_missing",
                        priority=10,
                        reason="unlabeled imported isolated position",
                    )
                stats.accepted_count += 1
            except Exception as exc:
                stats.rejected_count += 1
                stats.errors.append(f"line {line_no}: {exc}")
    return stats


def iter_selfplay_zip_jsonl(zip_path: Path) -> Iterable[tuple[str, str]]:
    with zipfile.ZipFile(zip_path) as archive:
        for name in sorted(archive.namelist()):
            if name.endswith(".jsonl"):
                with archive.open(name) as raw:
                    yield name, raw.read().decode("utf-8-sig", errors="replace")


def import_alpha_selfplay_zip(
    conn: sqlite3.Connection,
    path: Path,
    *,
    dry_run: bool,
) -> ImportStats:
    stats = ImportStats(source_path=str(path), source_format="alpha-selfplay-zip")
    for member, text in iter_selfplay_zip_jsonl(path):
        tmp_stats = import_alpha_selfplay_text(
            conn,
            text,
            source_label=f"{path.name}!{member}",
        )
        stats.record_count += tmp_stats.record_count
        stats.accepted_count += tmp_stats.accepted_count
        stats.rejected_count += tmp_stats.rejected_count
        stats.duplicate_count += tmp_stats.duplicate_count
        stats.errors.extend(tmp_stats.errors or [])
    return stats


def import_alpha_selfplay_text(conn: sqlite3.Connection, text: str, *, source_label: str) -> ImportStats:
    stats = ImportStats(source_path=source_label, source_format="alpha-selfplay-jsonl")
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        stats.record_count += 1
        try:
            obj = json.loads(line)
            sample = _alpha_selfplay_sample_to_label(obj)
            pos_id, created = ensure_position(conn, sample.state, ply=None, source_cohort="friend_selfplay")
            best_move_u8 = None
            if sample.policy_actions:
                best_idx = max(range(len(sample.policy_actions)), key=lambda i: sample.policy_values[i])
                best_move_u8 = _alpha_action_to_move_u8(sample.state, sample.policy_actions[best_idx])
            move_codes_u8 = [_alpha_action_to_move_u8(sample.state, action) for action in sample.policy_actions]
            add_label(
                conn,
                pos_id,
                label_type="teacher_value",
                source=source_label,
                value=sample.root_value,
                best_move_u8=best_move_u8,
                payload=_compact_alpha_payload(sample, move_codes_u8=move_codes_u8, source_label=source_label),
            )
            result = None
            if sample.outcome is not None and sample.outcome in (-1.0, 0.0, 1.0):
                result = int(sample.outcome)
            bump_observation(conn, pos_id, "friend_selfplay", result, evaluation_value=sample.root_value)
            if created and sample.root_value is None:
                queue_relabel(
                    conn,
                    pos_id,
                    requested_label_type="value_label_missing",
                    priority=10,
                    reason="unlabeled imported isolated position",
                )
            stats.accepted_count += 1
        except Exception as exc:
            stats.rejected_count += 1
            stats.errors.append(f"{source_label}:{line_no}: {exc}")
    return stats


def detect_import_format(path: Path) -> str:
    if path.suffix.lower() == ".db":
        return "sqlite-games-v1"
    if path.suffix.lower() == ".games":
        return "games-text-v1"
    if path.suffix.lower() == ".zip":
        return "alpha-selfplay-zip"
    if path.suffix.lower() != ".jsonl":
        raise ValueError(f"unsupported import path: {path}")
    obj = jsonl_first_object(path)
    if obj is None:
        return "jsonl-unparseable"
    return classify_jsonl_object(obj)[0]


def import_path(
    db_path: Path,
    source_path: Path,
    *,
    dry_run: bool = False,
    report_dir: Path = DEFAULT_REPORT_DIR,
) -> ImportStats:
    source_format = detect_import_format(source_path)
    if source_format == "jsonl-unparseable":
        raise ValueError(f"unparseable JSONL: {source_path}")
    conn = connect_db(db_path)
    source_hash = sha256_file(source_path)
    import_id = begin_import(conn, str(source_path), source_hash, source_format)
    stats = ImportStats(source_path=str(source_path), source_format=source_format)
    error_report_path: Path | None = None
    try:
        conn.commit()
        conn.execute("BEGIN")
        if source_format == "sqlite-games-v1":
            stats = import_all_games_db(conn, source_path, dry_run=dry_run)
        elif source_format == "games-text-v1":
            stats = import_games_file(conn, source_path, dry_run=dry_run)
        elif source_format == "search-pressure-jsonl":
            stats = import_search_pressure_jsonl(conn, source_path, dry_run=dry_run)
        elif source_format == "zero-search-budget-jsonl":
            stats = import_zero_teacher_jsonl(conn, source_path, dry_run=dry_run)
        elif source_format == "reduction-counterfactual-jsonl":
            stats = import_reduction_counterfactual_jsonl(conn, source_path, dry_run=dry_run)
        elif source_format == "alpha-selfplay-jsonl":
            stats = import_alpha_selfplay_jsonl(conn, source_path, dry_run=dry_run)
        elif source_format == "alpha-selfplay-zip":
            stats = import_alpha_selfplay_zip(conn, source_path, dry_run=dry_run)
        else:
            raise ValueError(f"no importer for format {source_format}")
        if stats.errors:
            report_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            error_report_path = report_dir / f"errors-{source_path.stem}-{stamp}.json"
            error_report_path.write_text(json.dumps(stats.errors, indent=2) + "\n", encoding="utf-8")
        if dry_run:
            conn.rollback()
            finish_import(conn, import_id, stats, "dry-run", str(error_report_path) if error_report_path else None)
        else:
            conn.commit()
            finish_import(conn, import_id, stats, "completed", str(error_report_path) if error_report_path else None)
            conn.commit()
    except Exception:
        conn.rollback()
        finish_import(conn, import_id, stats, "failed", str(error_report_path) if error_report_path else None)
        conn.commit()
        conn.close()
        raise
    conn.close()
    return stats


def import_all_known(
    db_path: Path = DEFAULT_DB_PATH,
    *,
    dry_run: bool = False,
) -> list[ImportStats]:
    records = inventory_scan(ROOT)
    stats: list[ImportStats] = []
    for record in records:
        if record.migration_status not in {"pending"}:
            continue
        if record.format not in {
            "sqlite-games-v1",
            "games-text-v1",
            "search-pressure-jsonl",
            "zero-search-budget-jsonl",
            "reduction-counterfactual-jsonl",
            "alpha-selfplay-jsonl",
            "alpha-selfplay-zip",
        }:
            continue
        try:
            stats.append(import_path(db_path, ROOT / record.path, dry_run=dry_run))
        except FileExistsError:
            continue
    return stats


def db_summary(db_path: Path = DEFAULT_DB_PATH) -> dict[str, Any]:
    conn = connect_db(db_path)
    summary = {
        "positions": int(conn.execute("SELECT COUNT(*) FROM positions").fetchone()[0]),
        "edges": int(conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]),
        "games": int(conn.execute("SELECT COUNT(*) FROM games").fetchone()[0]),
        "labels": int(conn.execute("SELECT COUNT(*) FROM labels").fetchone()[0]),
        "observations": int(conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0]),
        "imports": int(conn.execute("SELECT COUNT(*) FROM imports").fetchone()[0]),
        "relabel_queue": int(conn.execute("SELECT COUNT(*) FROM relabel_queue").fetchone()[0]),
        "bytes": db_path.stat().st_size if db_path.exists() else 0,
    }
    conn.close()
    return summary


def storage_report(
    db_path: Path = DEFAULT_DB_PATH,
    *,
    source_paths: Iterable[Path] | None = None,
) -> dict[str, Any]:
    conn = connect_db(db_path)
    summary = db_summary(db_path)
    total_path_bytes = conn.execute(
        "SELECT COALESCE(SUM(LENGTH(packed_u8_move_sequence)), 0) FROM game_paths"
    ).fetchone()[0]
    packed_state_bytes = conn.execute(
        "SELECT COALESCE(SUM(LENGTH(packed_state)), 0) FROM positions"
    ).fetchone()[0]
    label_payload_bytes = conn.execute(
        "SELECT COALESCE(SUM(LENGTH(payload_json)), 0) FROM labels"
    ).fetchone()[0]
    report: dict[str, Any] = {
        "database_path": str(db_path),
        "database_bytes": summary["bytes"],
        "positions": summary["positions"],
        "games": summary["games"],
        "labels": summary["labels"],
        "edges": summary["edges"],
        "packed_state_bytes": int(packed_state_bytes),
        "game_path_bytes": int(total_path_bytes),
        "label_payload_bytes": int(label_payload_bytes),
        "avg_bytes_per_game_path": (float(total_path_bytes) / summary["games"]) if summary["games"] else 0.0,
        "avg_bytes_per_unique_position": (float(summary["bytes"]) / summary["positions"]) if summary["positions"] else 0.0,
        "avg_bytes_per_label": (float(label_payload_bytes) / summary["labels"]) if summary["labels"] else 0.0,
    }
    if source_paths:
        sources = []
        total_source_bytes = 0
        for source_path in source_paths:
            path = Path(source_path)
            if not path.exists():
                continue
            size = path.stat().st_size
            total_source_bytes += size
            sources.append({"path": str(path), "bytes": size})
        report["sources"] = sources
        report["source_bytes_total"] = total_source_bytes
        report["db_vs_sources_ratio"] = (
            float(summary["bytes"]) / total_source_bytes if total_source_bytes else None
        )
    conn.close()
    return report


def audit_database(db_path: Path = DEFAULT_DB_PATH) -> dict[str, Any]:
    conn = connect_db(db_path)
    issues: list[str] = []
    sqlite_integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
    if sqlite_integrity != "ok":
        issues.append(f"sqlite_integrity_check={sqlite_integrity}")
    dup_positions = conn.execute(
        "SELECT COUNT(*) FROM (SELECT canonical_hash, packed_state, COUNT(*) c FROM positions "
        "GROUP BY canonical_hash, packed_state HAVING c > 1)"
    ).fetchone()[0]
    if dup_positions:
        issues.append(f"duplicate_positions={dup_positions}")
    bad_edges = conn.execute(
        "SELECT COUNT(*) FROM edges e LEFT JOIN positions p ON p.position_id=e.parent_position_id "
        "LEFT JOIN positions c ON c.position_id=e.child_position_id "
        "WHERE p.position_id IS NULL OR c.position_id IS NULL"
    ).fetchone()[0]
    if bad_edges:
        issues.append(f"orphan_edges={bad_edges}")
    replay_failures = 0
    rows = conn.execute(
        "SELECT g.game_id, gp.packed_u8_move_sequence FROM games g JOIN game_paths gp ON gp.game_id=g.game_id"
    ).fetchall()
    for row in rows:
        try:
            moves_from_u8_blob(row["packed_u8_move_sequence"])
        except Exception:
            replay_failures += 1
    if replay_failures:
        issues.append(f"game_replay_failures={replay_failures}")
    edge_conflicts = 0
    edge_rows = conn.execute(
        "SELECT e.parent_position_id, e.move_code_u8, e.child_position_id, pp.packed_state AS parent_state, "
        "cp.packed_state AS child_state FROM edges e "
        "JOIN positions pp ON pp.position_id=e.parent_position_id "
        "JOIN positions cp ON cp.position_id=e.child_position_id"
    ).fetchall()
    for row in edge_rows:
        parent = PositionState.unpack_state(row["parent_state"])
        child = PositionState.unpack_state(row["child_state"])
        try:
            move = decode_move(parent, int(row["move_code_u8"]))
            rebuilt = apply_move(parent, move)
            if rebuilt.packed_state() != child.packed_state():
                edge_conflicts += 1
        except Exception:
            edge_conflicts += 1
    if edge_conflicts:
        issues.append(f"edge_reconstruction_failures={edge_conflicts}")
    unlabeled_positions = conn.execute(
        "SELECT COUNT(*) FROM positions p LEFT JOIN labels l ON l.position_id=p.position_id WHERE l.position_id IS NULL"
    ).fetchone()[0]
    result = {
        "sqlite_integrity_check": sqlite_integrity,
        "issues": issues,
        "summary": db_summary(db_path),
        "unlabeled_positions": int(unlabeled_positions),
    }
    conn.close()
    return result


def export_training_rows(
    db_path: Path = DEFAULT_DB_PATH,
    *,
    out_path: Path,
    label_type: str = "teacher_value",
    limit: int | None = None,
) -> int:
    conn = connect_db(db_path)
    query = (
        "SELECT p.packed_state, p.total_visits, l.value, l.source, l.trunk_hash, l.engine_hash "
        "FROM positions p JOIN labels l ON l.position_id=p.position_id "
        "WHERE l.label_type=? ORDER BY p.total_visits DESC, l.quality_rank DESC"
    )
    if limit is not None:
        query += f" LIMIT {int(limit)}"
    rows = conn.execute(query, (label_type,)).fetchall()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            state = PositionState.unpack_state(row["packed_state"])
            handle.write(
                json_dumps(
                    {
                        "packed_state_hex": row["packed_state"].hex(),
                        "canonical_hash": state.canonical_hash().hex(),
                        "sample_weight": row["total_visits"],
                        "value_target": row["value"],
                        "label_source": row["source"],
                        "trunk_hash": row["trunk_hash"],
                        "engine_hash": row["engine_hash"],
                    }
                )
                + "\n"
            )
            count += 1
    conn.close()
    return count


@dataclass
class BinaryShardGame:
    result: int
    move_codes: bytes
    start_state: PositionState
    metadata: dict[str, Any]


class BinaryShardWriter:
    def __init__(
        self,
        output_dir: Path,
        *,
        engine_hash: str,
        trunk_hash: str,
        search_config_hash: str,
        worker_id: str,
        random_seed_range: str,
    ) -> None:
        self.output_dir = output_dir
        self.engine_hash = engine_hash
        self.trunk_hash = trunk_hash
        self.search_config_hash = search_config_hash
        self.worker_id = worker_id
        self.random_seed_range = random_seed_range
        self.games: list[BinaryShardGame] = []

    def add_game(
        self,
        moves: list[str],
        *,
        result: int,
        start_state: PositionState | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        state = start_state or PositionState.initial()
        move_blob = moves_to_u8_blob(moves, start=state)
        self.games.append(BinaryShardGame(result=result, move_codes=move_blob, start_state=state, metadata=metadata or {}))

    def write_ready(self, stem: str) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "shard_format_version": SHARD_FORMAT_VERSION,
            "engine_hash": self.engine_hash,
            "trunk_hash": self.trunk_hash,
            "search_config_hash": self.search_config_hash,
            "worker_id": self.worker_id,
            "random_seed_range": self.random_seed_range,
            "created_at": utc_now(),
            "game_count": len(self.games),
        }
        meta_bytes = json_dumps(metadata).encode("utf-8")
        partial = self.output_dir / f"{stem}.partial"
        ready = self.output_dir / f"{stem}.ready"
        digest = hashlib.sha256()
        with partial.open("wb") as handle:
            handle.write(SHARD_MAGIC)
            handle.write(struct.pack("<I", len(meta_bytes)))
            handle.write(meta_bytes)
            digest.update(SHARD_MAGIC)
            digest.update(struct.pack("<I", len(meta_bytes)))
            digest.update(meta_bytes)
            for game in self.games:
                record_payload = bytearray()
                record_payload.extend(struct.pack("<bH", game.result, len(game.move_codes)))
                packed_start = game.start_state.packed_state()
                record_payload.extend(struct.pack("<H", len(packed_start)))
                record_payload.extend(packed_start)
                record_payload.extend(struct.pack("<I", len(json_dumps(game.metadata).encode("utf-8"))))
                meta_payload = json_dumps(game.metadata).encode("utf-8")
                record_payload.extend(meta_payload)
                record_payload.extend(game.move_codes)
                handle.write(struct.pack("<I", len(record_payload)))
                handle.write(record_payload)
                digest.update(struct.pack("<I", len(record_payload)))
                digest.update(record_payload)
            handle.write(SHARD_TRAILER_MAGIC)
            handle.write(digest.digest())
        partial.replace(ready)
        return ready


def iter_binary_shard_games(path: Path) -> tuple[dict[str, Any], list[BinaryShardGame]]:
    data = path.read_bytes()
    stream = io.BytesIO(data)
    digest = hashlib.sha256()
    magic = stream.read(8)
    if magic != SHARD_MAGIC:
        raise ValueError("bad shard magic")
    meta_len = struct.unpack("<I", stream.read(4))[0]
    meta_bytes = stream.read(meta_len)
    metadata = json.loads(meta_bytes.decode("utf-8"))
    digest.update(magic)
    digest.update(struct.pack("<I", meta_len))
    digest.update(meta_bytes)
    games: list[BinaryShardGame] = []
    while True:
        peek = stream.read(len(SHARD_TRAILER_MAGIC))
        if peek == SHARD_TRAILER_MAGIC:
            trailer_digest = stream.read(32)
            if trailer_digest != digest.digest():
                raise ValueError("shard checksum mismatch")
            break
        stream.seek(-len(SHARD_TRAILER_MAGIC), io.SEEK_CUR)
        record_len = struct.unpack("<I", stream.read(4))[0]
        record = stream.read(record_len)
        digest.update(struct.pack("<I", record_len))
        digest.update(record)
        rec = io.BytesIO(record)
        result, move_count = struct.unpack("<bH", rec.read(3))
        start_len = struct.unpack("<H", rec.read(2))[0]
        start_state = PositionState.unpack_state(rec.read(start_len))
        metadata_len = struct.unpack("<I", rec.read(4))[0]
        game_meta = json.loads(rec.read(metadata_len).decode("utf-8"))
        move_codes = rec.read(move_count)
        if len(move_codes) != move_count:
            raise ValueError("truncated move blob in shard")
        games.append(
            BinaryShardGame(
                result=result,
                move_codes=move_codes,
                start_state=start_state,
                metadata=game_meta,
            )
        )
    return metadata, games


def import_binary_shard(
    db_path: Path,
    shard_path: Path,
    *,
    dry_run: bool = False,
) -> ImportStats:
    if shard_path.suffix != ".ready":
        raise ValueError("ingester only accepts .ready shards")
    source_hash = sha256_file(shard_path)
    conn = connect_db(db_path)
    import_id = begin_import(conn, str(shard_path), source_hash, "binary-shard-v1")
    stats = ImportStats(source_path=str(shard_path), source_format="binary-shard-v1")
    metadata, games = iter_binary_shard_games(shard_path)
    try:
        conn.commit()
        conn.execute("BEGIN")
        for idx, game in enumerate(games, start=1):
            stats.record_count += 1
            try:
                moves = moves_from_u8_blob(game.move_codes, start=game.start_state)
                insert_game(
                    conn,
                    moves,
                    game.result,
                    source=f"binary-shard:{metadata.get('worker_id', 'worker')}",
                    metadata={"shard_metadata": metadata, "record_index": idx, "game_metadata": game.metadata},
                    source_cohort=f"shard:{metadata.get('worker_id', 'worker')}",
                )
                stats.accepted_count += 1
            except Exception as exc:
                stats.rejected_count += 1
                stats.errors.append(f"record {idx}: {exc}")
        if dry_run:
            conn.rollback()
            finish_import(conn, import_id, stats, "dry-run")
        else:
            conn.commit()
            finish_import(conn, import_id, stats, "completed")
            conn.commit()
            shard_path.rename(shard_path.with_suffix(".imported"))
    except Exception:
        conn.rollback()
        finish_import(conn, import_id, stats, "failed")
        conn.commit()
        conn.close()
        raise
    conn.close()
    return stats
