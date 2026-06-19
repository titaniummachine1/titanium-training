from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sqlite3
import struct
import tempfile
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from position_store_state import PositionState

COMPACT_DATABASE_SCHEMA_VERSION = 2
COMPACT_LABEL_SCHEMA_VERSION = 2
COMPACT_EXPORT_SCHEMA_VERSION = 1

EVAL_UNIT_NAME = "centitempo"
EVAL_UNITS_PER_TEMPO = 100
EVAL_POSITIVE_CONVENTION = "side-to-move favorable"
EVAL_NORMAL_RANGE = [-32767, 32767]
EVAL_TRUE_MATE_SCORE = 100_000
EVAL_RACE_PROOF_SCORE = 32_000

TARGET_KIND_UNKNOWN = 0
TARGET_KIND_EVAL_I16 = 1
TARGET_KIND_PROBABILITY_U16 = 2
TARGET_KIND_SCALAR_Q15 = 3
TARGET_KIND_OUTCOME = 4

LABEL_TYPE_UNKNOWN = 0
LABEL_TYPE_SEARCH_VALUE = 1
LABEL_TYPE_TEACHER_VALUE = 2
LABEL_TYPE_SEARCH_PRESSURE = 3
LABEL_TYPE_REDUCTION_COUNTERFACTUAL = 4
LABEL_TYPE_GAME_RESULT = 5

PAYLOAD_KIND_JSON_ZLIB = 1
PAYLOAD_KIND_SPARSE_POLICY_U16 = 2

BOUND_UNKNOWN = 0
BOUND_EXACT = 1
BOUND_LOWER = 2
BOUND_UPPER = 3

FLAG_PROVEN = 1 << 2
FLAG_TERMINAL = 1 << 3
FLAG_SATURATED = 1 << 4
FLAG_HAS_BEST_MOVE = 1 << 5
FLAG_HAS_DEPTH = 1 << 6
FLAG_HAS_NODES = 1 << 7

OUTCOME_UNKNOWN = 0
OUTCOME_P0_WIN = 1
OUTCOME_DRAW = 2
OUTCOME_P1_WIN = 3

SPARSE_POLICY_MAGIC = b"TIPOL1"
JSON_BLOB_MAGIC = b"TIJZ01"
EXPORT_MAGIC = b"TICLBL1\0"


COMPACT_SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

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

CREATE TABLE IF NOT EXISTS sources (
    source_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS engine_versions (
    engine_version_id INTEGER PRIMARY KEY,
    engine_hash BLOB NOT NULL UNIQUE,
    display_name TEXT
);

CREATE TABLE IF NOT EXISTS trunk_versions (
    trunk_version_id INTEGER PRIMARY KEY,
    trunk_hash BLOB NOT NULL UNIQUE,
    display_name TEXT
);

CREATE TABLE IF NOT EXISTS search_configs (
    search_config_id INTEGER PRIMARY KEY,
    config_hash BLOB NOT NULL UNIQUE,
    canonical_json TEXT
);

CREATE TABLE IF NOT EXISTS games (
    game_id INTEGER PRIMARY KEY,
    start_position_id INTEGER NOT NULL REFERENCES positions(position_id),
    result_code INTEGER NOT NULL DEFAULT 0,
    move_count INTEGER NOT NULL,
    generator_engine_version_id INTEGER REFERENCES engine_versions(engine_version_id),
    generator_trunk_version_id INTEGER REFERENCES trunk_versions(trunk_version_id),
    search_config_id INTEGER REFERENCES search_configs(search_config_id),
    random_seed TEXT,
    worker_id TEXT,
    source_id INTEGER REFERENCES sources(source_id),
    created_at TEXT NOT NULL,
    shard_id TEXT,
    game_metadata TEXT
);

CREATE TABLE IF NOT EXISTS game_paths (
    game_id INTEGER PRIMARY KEY REFERENCES games(game_id),
    packed_u8_move_sequence BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS payload_refs (
    payload_ref_id INTEGER PRIMARY KEY,
    payload_kind INTEGER NOT NULL,
    payload_hash BLOB NOT NULL UNIQUE,
    storage_path TEXT NOT NULL,
    byte_offset INTEGER NOT NULL,
    raw_bytes INTEGER NOT NULL,
    stored_bytes INTEGER NOT NULL,
    compression_code INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS label_sets (
    label_set_id INTEGER PRIMARY KEY,
    label_type_code INTEGER NOT NULL,
    target_kind INTEGER NOT NULL,
    engine_version_id INTEGER REFERENCES engine_versions(engine_version_id),
    trunk_version_id INTEGER REFERENCES trunk_versions(trunk_version_id),
    search_config_id INTEGER REFERENCES search_configs(search_config_id),
    label_schema_version INTEGER NOT NULL,
    source_id INTEGER REFERENCES sources(source_id),
    created_at TEXT NOT NULL,
    UNIQUE(label_type_code, target_kind, engine_version_id, trunk_version_id, search_config_id, label_schema_version, source_id)
);

CREATE TABLE IF NOT EXISTS canonical_labels (
    label_id INTEGER PRIMARY KEY,
    position_id INTEGER NOT NULL REFERENCES positions(position_id),
    label_set_id INTEGER NOT NULL REFERENCES label_sets(label_set_id),
    target_kind INTEGER NOT NULL,
    eval_i16 INTEGER,
    scalar_q15 INTEGER,
    probability_u16 INTEGER,
    outcome_code INTEGER NOT NULL DEFAULT 0,
    distance_to_terminal INTEGER,
    best_move_u8 INTEGER,
    completed_depth INTEGER,
    selective_depth INTEGER,
    nodes INTEGER,
    flags_u16 INTEGER NOT NULL DEFAULT 0,
    quality_rank INTEGER NOT NULL DEFAULT 0,
    confidence_u8 INTEGER NOT NULL DEFAULT 0,
    score_stability_i16 INTEGER,
    best_move_stable_depths_u8 INTEGER,
    pv_change_count_u8 INTEGER,
    policy_margin_i16 INTEGER,
    payload_ref_id INTEGER REFERENCES payload_refs(payload_ref_id),
    observation_count INTEGER NOT NULL DEFAULT 1,
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    UNIQUE(position_id, label_set_id, target_kind)
);

CREATE TABLE IF NOT EXISTS label_observations (
    position_id INTEGER NOT NULL REFERENCES positions(position_id),
    label_set_id INTEGER NOT NULL REFERENCES label_sets(label_set_id),
    target_kind INTEGER NOT NULL,
    sample_count INTEGER NOT NULL DEFAULT 0,
    exact_count INTEGER NOT NULL DEFAULT 0,
    proven_count INTEGER NOT NULL DEFAULT 0,
    disagreement_count INTEGER NOT NULL DEFAULT 0,
    eval_min_i16 INTEGER,
    eval_max_i16 INTEGER,
    eval_sum_i64 INTEGER,
    scalar_min_i16 INTEGER,
    scalar_max_i16 INTEGER,
    scalar_sum_i64 INTEGER,
    probability_min_u16 INTEGER,
    probability_max_u16 INTEGER,
    probability_sum_i64 INTEGER,
    best_move_mode_u8 INTEGER,
    best_move_agreement_count INTEGER NOT NULL DEFAULT 0,
    payload_ref_count INTEGER NOT NULL DEFAULT 0,
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    PRIMARY KEY(position_id, label_set_id, target_kind)
);

CREATE TABLE IF NOT EXISTS observations (
    position_id INTEGER NOT NULL REFERENCES positions(position_id),
    source_id INTEGER NOT NULL REFERENCES sources(source_id),
    visit_count INTEGER NOT NULL DEFAULT 0,
    p0_win_count INTEGER NOT NULL DEFAULT 0,
    draw_count INTEGER NOT NULL DEFAULT 0,
    p1_win_count INTEGER NOT NULL DEFAULT 0,
    eval_count INTEGER NOT NULL DEFAULT 0,
    last_eval_q15 INTEGER,
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    PRIMARY KEY(position_id, source_id)
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
CREATE INDEX IF NOT EXISTS idx_games_source_id ON games(source_id);
CREATE INDEX IF NOT EXISTS idx_label_sets_lookup ON label_sets(label_type_code, target_kind, source_id);
CREATE INDEX IF NOT EXISTS idx_canonical_labels_lookup ON canonical_labels(position_id, label_set_id, target_kind);
CREATE INDEX IF NOT EXISTS idx_canonical_labels_hot ON canonical_labels(label_set_id, confidence_u8 DESC, quality_rank DESC);
CREATE INDEX IF NOT EXISTS idx_observations_source_id ON observations(source_id, visit_count DESC);
CREATE INDEX IF NOT EXISTS idx_relabel_pending ON relabel_queue(priority DESC, created_at) WHERE status='pending';
"""


@dataclass
class CompactMigrationStats:
    source_db: str
    dest_db: str
    sidecar_dir: str
    positions: int = 0
    edges: int = 0
    games: int = 0
    labels_seen: int = 0
    canonical_labels: int = 0
    label_observation_groups: int = 0
    payload_refs: int = 0
    observations: int = 0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_bytes(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def identity_blob(text: str | None) -> bytes | None:
    if text is None:
        return None
    value = str(text).strip()
    if not value:
        return None
    try:
        if len(value) % 2 == 0:
            decoded = bytes.fromhex(value)
            if decoded.hex() == value.lower():
                return decoded
    except ValueError:
        pass
    return sha256_bytes(value.encode("utf-8"))


def open_sqlite(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_compact_db(path: Path) -> None:
    conn = open_sqlite(path)
    conn.executescript(COMPACT_SCHEMA_SQL)
    conn.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)",
        ("database_schema_version", str(COMPACT_DATABASE_SCHEMA_VERSION)),
    )
    conn.commit()
    conn.close()


def score_semantics_report() -> dict[str, Any]:
    return {
        "eval_unit_name": EVAL_UNIT_NAME,
        "eval_units_per_tempo": EVAL_UNITS_PER_TEMPO,
        "positive_side_convention": EVAL_POSITIVE_CONVENTION,
        "normal_eval_range_i16": EVAL_NORMAL_RANGE,
        "race_proof_score_band": {
            "base": EVAL_RACE_PROOF_SCORE,
            "encoding": "race win/loss in k plies => +/- (32000 - k)",
        },
        "true_mate_score_band": {
            "base": EVAL_TRUE_MATE_SCORE,
            "encoding": "mate win/loss in k plies => +/- (100000 - k)",
        },
        "draw_encoding": "not used by Quoridor game records",
        "unknown_encoding": "flags + NULL target field",
    }


def encode_scalar_q15(value: float | None) -> int | None:
    if value is None:
        return None
    clamped = max(-1.0, min(1.0, float(value)))
    return int(round(clamped * 32767.0))


def encode_probability_u16(value: float | None) -> int | None:
    if value is None:
        return None
    clamped = max(0.0, min(1.0, float(value)))
    return int(round(clamped * 65535.0))


def clamp_eval_i16(value: int | float | None) -> tuple[int | None, bool]:
    if value is None:
        return None, False
    iv = int(round(float(value)))
    saturated = iv < -32768 or iv > 32767
    iv = max(-32768, min(32767, iv))
    return iv, saturated


def outcome_to_code(result: int | None) -> int:
    if result == 1:
        return OUTCOME_P0_WIN
    if result == -1:
        return OUTCOME_P1_WIN
    if result == 0:
        return OUTCOME_DRAW
    return OUTCOME_UNKNOWN


def label_type_code(name: str) -> int:
    mapping = {
        "search_value": LABEL_TYPE_SEARCH_VALUE,
        "deep_relabel": LABEL_TYPE_SEARCH_VALUE,
        "teacher_value": LABEL_TYPE_TEACHER_VALUE,
        "search_pressure": LABEL_TYPE_SEARCH_PRESSURE,
        "reduction_counterfactual": LABEL_TYPE_REDUCTION_COUNTERFACTUAL,
        "game_result": LABEL_TYPE_GAME_RESULT,
    }
    return mapping.get(name, LABEL_TYPE_UNKNOWN)


def infer_target_kind(label_type: str, value: Any) -> int:
    if label_type == "search_pressure":
        return TARGET_KIND_SCALAR_Q15
    if label_type == "reduction_counterfactual":
        return TARGET_KIND_PROBABILITY_U16
    if label_type == "game_result":
        return TARGET_KIND_OUTCOME
    if label_type == "teacher_value":
        if value is None:
            return TARGET_KIND_UNKNOWN
        fv = float(value)
        if -1.0001 <= fv <= 1.0001:
            return TARGET_KIND_SCALAR_Q15
        return TARGET_KIND_EVAL_I16
    if label_type == "search_value" or label_type == "deep_relabel":
        return TARGET_KIND_EVAL_I16
    return TARGET_KIND_UNKNOWN


def bound_bits(bound: str | None) -> int:
    mapping = {
        None: BOUND_UNKNOWN,
        "": BOUND_UNKNOWN,
        "exact": BOUND_EXACT,
        "lower": BOUND_LOWER,
        "upper": BOUND_UPPER,
    }
    return mapping.get(bound, BOUND_UNKNOWN)


def compute_label_confidence(
    *,
    target_kind: int,
    flags_u16: int,
    nodes: int | None,
    completed_depth: int | None,
    quality_rank: int,
) -> int:
    confidence = 32
    if flags_u16 & FLAG_PROVEN:
        confidence += 120
    if (flags_u16 & 0b11) == BOUND_EXACT:
        confidence += 50
    if completed_depth is not None:
        confidence += min(40, max(0, int(completed_depth)) * 4)
    if nodes is not None and nodes > 0:
        confidence += min(40, int(math.log10(nodes + 1) * 16))
    confidence += min(12, max(0, quality_rank))
    if target_kind == TARGET_KIND_OUTCOME:
        confidence = max(confidence, 220)
    return max(0, min(255, confidence))


def canonical_label_better(candidate: dict[str, Any], current: sqlite3.Row) -> bool:
    def rank_tuple(item: dict[str, Any] | sqlite3.Row) -> tuple:
        flags = int(item["flags_u16"])
        exact = 1 if (flags & 0b11) == BOUND_EXACT else 0
        proven = 1 if (flags & FLAG_PROVEN) else 0
        quality = int(item["quality_rank"] or 0)
        nodes = int(item["nodes"] or 0)
        depth = int(item["completed_depth"] or 0)
        confidence = int(item["confidence_u8"] or 0)
        created = str(item["last_seen_at"])
        return (proven, exact, quality, nodes, depth, confidence, created)

    return rank_tuple(candidate) > rank_tuple(current)


class CompactSidecarWriter:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def store(self, kind_name: str, payload_kind: int, raw: bytes) -> tuple[str, int, int]:
        filename = f"{kind_name}.bin"
        path = self.root / filename
        payload_hash = sha256_bytes(raw)
        if path.exists():
            pass
        compressed = zlib.compress(raw, 9)
        with path.open("ab") as handle:
            offset = handle.tell()
            handle.write(struct.pack("<6sII32s", b"REC001", len(raw), len(compressed), payload_hash))
            handle.write(compressed)
        return filename, offset, len(compressed) + 46


def _maybe_sparse_policy_payload(payload: dict[str, Any]) -> bytes | None:
    move_codes = payload.get("policy_move_codes_u8")
    policy_values = payload.get("policy_values")
    if not isinstance(move_codes, list) or not isinstance(policy_values, list):
        return None
    if len(move_codes) != len(policy_values) or len(move_codes) > 255:
        return None
    out = bytearray()
    out.extend(SPARSE_POLICY_MAGIC)
    out.append(len(move_codes))
    for move_code, policy_value in zip(move_codes, policy_values):
        out.append(int(move_code) & 0xFF)
        out.extend(struct.pack("<H", encode_probability_u16(float(policy_value)) or 0))
    return bytes(out)


def _payload_blob_for_label(label_type: str, payload: dict[str, Any]) -> tuple[int, bytes] | None:
    if not payload:
        return None
    if label_type == "teacher_value":
        sparse = _maybe_sparse_policy_payload(payload)
        if sparse is not None:
            return PAYLOAD_KIND_SPARSE_POLICY_U16, sparse
    raw = JSON_BLOB_MAGIC + json_dumps(payload).encode("utf-8")
    return PAYLOAD_KIND_JSON_ZLIB, raw


def _get_or_create_source(conn: sqlite3.Connection, name: str | None) -> int | None:
    if name is None or not str(name).strip():
        return None
    row = conn.execute("SELECT source_id FROM sources WHERE name=?", (str(name),)).fetchone()
    if row:
        return int(row["source_id"])
    cur = conn.execute("INSERT INTO sources(name) VALUES(?)", (str(name),))
    return int(cur.lastrowid)


def _get_or_create_engine(conn: sqlite3.Connection, value: str | None) -> int | None:
    blob = identity_blob(value)
    if blob is None:
        return None
    row = conn.execute("SELECT engine_version_id FROM engine_versions WHERE engine_hash=?", (blob,)).fetchone()
    if row:
        return int(row["engine_version_id"])
    cur = conn.execute(
        "INSERT INTO engine_versions(engine_hash, display_name) VALUES(?, ?)",
        (blob, value),
    )
    return int(cur.lastrowid)


def _get_or_create_trunk(conn: sqlite3.Connection, value: str | None) -> int | None:
    blob = identity_blob(value)
    if blob is None:
        return None
    row = conn.execute("SELECT trunk_version_id FROM trunk_versions WHERE trunk_hash=?", (blob,)).fetchone()
    if row:
        return int(row["trunk_version_id"])
    cur = conn.execute(
        "INSERT INTO trunk_versions(trunk_hash, display_name) VALUES(?, ?)",
        (blob, value),
    )
    return int(cur.lastrowid)


def _get_or_create_search_config(conn: sqlite3.Connection, value: str | None) -> int | None:
    blob = identity_blob(value)
    if blob is None:
        return None
    row = conn.execute("SELECT search_config_id FROM search_configs WHERE config_hash=?", (blob,)).fetchone()
    if row:
        return int(row["search_config_id"])
    cur = conn.execute(
        "INSERT INTO search_configs(config_hash, canonical_json) VALUES(?, ?)",
        (blob, value),
    )
    return int(cur.lastrowid)


def _get_or_create_label_set(
    conn: sqlite3.Connection,
    *,
    label_type: str,
    target_kind: int,
    source: str | None,
    engine_hash: str | None,
    trunk_hash: str | None,
    search_config_hash: str | None,
    label_schema_version: int,
    created_at: str,
) -> int:
    source_id = _get_or_create_source(conn, source)
    engine_version_id = _get_or_create_engine(conn, engine_hash)
    trunk_version_id = _get_or_create_trunk(conn, trunk_hash)
    search_config_id = _get_or_create_search_config(conn, search_config_hash)
    code = label_type_code(label_type)
    row = conn.execute(
        "SELECT label_set_id FROM label_sets WHERE label_type_code=? AND target_kind=? AND "
        "engine_version_id IS ? AND trunk_version_id IS ? AND search_config_id IS ? AND "
        "label_schema_version=? AND source_id IS ?",
        (code, target_kind, engine_version_id, trunk_version_id, search_config_id, label_schema_version, source_id),
    ).fetchone()
    if row:
        return int(row["label_set_id"])
    cur = conn.execute(
        "INSERT INTO label_sets(label_type_code, target_kind, engine_version_id, trunk_version_id, search_config_id, "
        "label_schema_version, source_id, created_at) VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
        (code, target_kind, engine_version_id, trunk_version_id, search_config_id, label_schema_version, source_id, created_at),
    )
    return int(cur.lastrowid)


def _get_or_create_payload_ref(
    conn: sqlite3.Connection,
    writer: CompactSidecarWriter,
    *,
    label_type: str,
    payload: dict[str, Any] | None,
) -> int | None:
    packed = _payload_blob_for_label(label_type, payload or {})
    if packed is None:
        return None
    payload_kind, raw = packed
    payload_hash = sha256_bytes(raw)
    row = conn.execute("SELECT payload_ref_id FROM payload_refs WHERE payload_hash=?", (payload_hash,)).fetchone()
    if row:
        return int(row["payload_ref_id"])
    kind_name = "json" if payload_kind == PAYLOAD_KIND_JSON_ZLIB else "policy"
    storage_path, byte_offset, stored_bytes = writer.store(kind_name, payload_kind, raw)
    cur = conn.execute(
        "INSERT INTO payload_refs(payload_kind, payload_hash, storage_path, byte_offset, raw_bytes, stored_bytes, compression_code, created_at) "
        "VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
        (payload_kind, payload_hash, storage_path, byte_offset, len(raw), stored_bytes, 1, utc_now()),
    )
    return int(cur.lastrowid)


def _convert_v1_label_row(row: sqlite3.Row) -> dict[str, Any]:
    label_type = str(row["label_type"])
    target_kind = infer_target_kind(label_type, row["value"])
    flags = bound_bits(row["bound"])
    if int(row["is_proven"] or 0):
        flags |= FLAG_PROVEN
    if row["best_move_u8"] is not None:
        flags |= FLAG_HAS_BEST_MOVE
    if row["completed_depth"] is not None:
        flags |= FLAG_HAS_DEPTH
    if row["nodes"] is not None:
        flags |= FLAG_HAS_NODES

    eval_i16 = None
    scalar_q15 = None
    probability_u16 = None
    outcome_code = OUTCOME_UNKNOWN
    distance_to_terminal = None
    saturated = False

    if target_kind == TARGET_KIND_EVAL_I16:
        eval_i16, saturated = clamp_eval_i16(row["value"])
    elif target_kind == TARGET_KIND_SCALAR_Q15:
        scalar_q15 = encode_scalar_q15(row["value"])
    elif target_kind == TARGET_KIND_PROBABILITY_U16:
        probability_u16 = encode_probability_u16(row["value"])
    elif target_kind == TARGET_KIND_OUTCOME:
        outcome_code = outcome_to_code(int(row["value"]) if row["value"] is not None else None)

    if saturated:
        flags |= FLAG_SATURATED
    confidence = compute_label_confidence(
        target_kind=target_kind,
        flags_u16=flags,
        nodes=int(row["nodes"]) if row["nodes"] is not None else None,
        completed_depth=int(row["completed_depth"]) if row["completed_depth"] is not None else None,
        quality_rank=int(row["quality_rank"] or 0),
    )
    return {
        "position_id": int(row["position_id"]),
        "label_type": label_type,
        "target_kind": target_kind,
        "eval_i16": eval_i16,
        "scalar_q15": scalar_q15,
        "probability_u16": probability_u16,
        "outcome_code": outcome_code,
        "distance_to_terminal": distance_to_terminal,
        "best_move_u8": row["best_move_u8"],
        "completed_depth": row["completed_depth"],
        "selective_depth": row["selective_depth"],
        "nodes": row["nodes"],
        "flags_u16": flags,
        "quality_rank": int(row["quality_rank"] or 0),
        "confidence_u8": confidence,
        "score_stability_i16": None,
        "best_move_stable_depths_u8": None,
        "pv_change_count_u8": None,
        "policy_margin_i16": None,
        "source": row["source"],
        "engine_hash": row["engine_hash"],
        "trunk_hash": row["trunk_hash"],
        "search_config_hash": row["search_config_hash"],
        "label_schema_version": int(row["label_schema_version"] or 1),
        "created_at": str(row["created_at"]),
        "payload": json.loads(row["payload_json"]) if row["payload_json"] else None,
    }


def _upsert_label_observation(conn: sqlite3.Connection, item: dict[str, Any], label_set_id: int, payload_present: bool) -> None:
    key = (item["position_id"], label_set_id, item["target_kind"])
    row = conn.execute(
        "SELECT * FROM label_observations WHERE position_id=? AND label_set_id=? AND target_kind=?",
        key,
    ).fetchone()
    exact_count = 1 if (int(item["flags_u16"]) & 0b11) == BOUND_EXACT else 0
    proven_count = 1 if (int(item["flags_u16"]) & FLAG_PROVEN) else 0
    eval_val = item["eval_i16"]
    scalar_val = item["scalar_q15"]
    prob_val = item["probability_u16"]
    if row:
        disagreement = int(row["disagreement_count"])
        if eval_val is not None and row["eval_min_i16"] is not None and int(row["eval_min_i16"]) != int(eval_val):
            disagreement += 1
        if scalar_val is not None and row["scalar_min_i16"] is not None and int(row["scalar_min_i16"]) != int(scalar_val):
            disagreement += 1
        if prob_val is not None and row["probability_min_u16"] is not None and int(row["probability_min_u16"]) != int(prob_val):
            disagreement += 1
        conn.execute(
            "UPDATE label_observations SET sample_count=sample_count+1, exact_count=exact_count+?, proven_count=proven_count+?, "
            "disagreement_count=?, eval_min_i16=COALESCE(MIN(eval_min_i16, ?), ?), eval_max_i16=COALESCE(MAX(eval_max_i16, ?), ?), "
            "eval_sum_i64=COALESCE(eval_sum_i64, 0) + COALESCE(?, 0), scalar_min_i16=COALESCE(MIN(scalar_min_i16, ?), ?), "
            "scalar_max_i16=COALESCE(MAX(scalar_max_i16, ?), ?), scalar_sum_i64=COALESCE(scalar_sum_i64, 0) + COALESCE(?, 0), "
            "probability_min_u16=COALESCE(MIN(probability_min_u16, ?), ?), probability_max_u16=COALESCE(MAX(probability_max_u16, ?), ?), "
            "probability_sum_i64=COALESCE(probability_sum_i64, 0) + COALESCE(?, 0), best_move_mode_u8=COALESCE(best_move_mode_u8, ?), "
            "best_move_agreement_count=best_move_agreement_count + CASE WHEN best_move_mode_u8 IS ? THEN 1 ELSE 0 END, "
            "payload_ref_count=payload_ref_count + ?, last_seen_at=? WHERE position_id=? AND label_set_id=? AND target_kind=?",
            (
                exact_count,
                proven_count,
                disagreement,
                eval_val, eval_val, eval_val, eval_val,
                eval_val,
                scalar_val, scalar_val, scalar_val, scalar_val,
                scalar_val,
                prob_val, prob_val, prob_val, prob_val,
                prob_val,
                item["best_move_u8"],
                item["best_move_u8"],
                1 if payload_present else 0,
                item["created_at"],
                item["position_id"],
                label_set_id,
                item["target_kind"],
            ),
        )
        return
    conn.execute(
        "INSERT INTO label_observations(position_id, label_set_id, target_kind, sample_count, exact_count, proven_count, disagreement_count, "
        "eval_min_i16, eval_max_i16, eval_sum_i64, scalar_min_i16, scalar_max_i16, scalar_sum_i64, probability_min_u16, "
        "probability_max_u16, probability_sum_i64, best_move_mode_u8, best_move_agreement_count, payload_ref_count, first_seen_at, last_seen_at) "
        "VALUES(?, ?, ?, 1, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            item["position_id"], label_set_id, item["target_kind"], exact_count, proven_count,
            eval_val, eval_val, eval_val,
            scalar_val, scalar_val, scalar_val,
            prob_val, prob_val, prob_val,
            item["best_move_u8"], 1 if item["best_move_u8"] is not None else 0,
            1 if payload_present else 0,
            item["created_at"], item["created_at"],
        ),
    )


def _upsert_canonical_label(
    conn: sqlite3.Connection,
    item: dict[str, Any],
    *,
    label_set_id: int,
    payload_ref_id: int | None,
) -> None:
    row = conn.execute(
        "SELECT * FROM canonical_labels WHERE position_id=? AND label_set_id=? AND target_kind=?",
        (item["position_id"], label_set_id, item["target_kind"]),
    ).fetchone()
    if row:
        conn.execute(
            "UPDATE canonical_labels SET observation_count=observation_count+1, last_seen_at=? WHERE label_id=?",
            (item["created_at"], row["label_id"]),
        )
        merged = dict(item)
        merged["last_seen_at"] = item["created_at"]
        if canonical_label_better(merged, row):
            conn.execute(
                "UPDATE canonical_labels SET eval_i16=?, scalar_q15=?, probability_u16=?, outcome_code=?, distance_to_terminal=?, "
                "best_move_u8=?, completed_depth=?, selective_depth=?, nodes=?, flags_u16=?, quality_rank=?, confidence_u8=?, "
                "score_stability_i16=?, best_move_stable_depths_u8=?, pv_change_count_u8=?, policy_margin_i16=?, payload_ref_id=?, "
                "last_seen_at=? WHERE label_id=?",
                (
                    item["eval_i16"], item["scalar_q15"], item["probability_u16"], item["outcome_code"], item["distance_to_terminal"],
                    item["best_move_u8"], item["completed_depth"], item["selective_depth"], item["nodes"], item["flags_u16"],
                    item["quality_rank"], item["confidence_u8"], item["score_stability_i16"], item["best_move_stable_depths_u8"],
                    item["pv_change_count_u8"], item["policy_margin_i16"], payload_ref_id, item["created_at"], row["label_id"],
                ),
            )
        return
    conn.execute(
        "INSERT INTO canonical_labels(position_id, label_set_id, target_kind, eval_i16, scalar_q15, probability_u16, outcome_code, "
        "distance_to_terminal, best_move_u8, completed_depth, selective_depth, nodes, flags_u16, quality_rank, confidence_u8, "
        "score_stability_i16, best_move_stable_depths_u8, pv_change_count_u8, policy_margin_i16, payload_ref_id, observation_count, "
        "first_seen_at, last_seen_at) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)",
        (
            item["position_id"], label_set_id, item["target_kind"], item["eval_i16"], item["scalar_q15"], item["probability_u16"],
            item["outcome_code"], item["distance_to_terminal"], item["best_move_u8"], item["completed_depth"], item["selective_depth"],
            item["nodes"], item["flags_u16"], item["quality_rank"], item["confidence_u8"], item["score_stability_i16"],
            item["best_move_stable_depths_u8"], item["pv_change_count_u8"], item["policy_margin_i16"], payload_ref_id,
            item["created_at"], item["created_at"],
        ),
    )


def _copy_table(src: sqlite3.Connection, dst: sqlite3.Connection, table: str, columns: list[str], insert_sql: str) -> int:
    rows = src.execute(f"SELECT {', '.join(columns)} FROM {table}").fetchall()
    dst.executemany(insert_sql, [tuple(row[col] for col in columns) for row in rows])
    return len(rows)


def rebuild_compact_db(src_db: Path, dst_db: Path, *, sidecar_dir: Path | None = None) -> dict[str, Any]:
    if dst_db.exists():
        raise FileExistsError(f"destination already exists: {dst_db}")
    if sidecar_dir is None:
        sidecar_dir = dst_db.with_suffix(".sidecars")
    init_compact_db(dst_db)
    src = sqlite3.connect(str(src_db))
    src.row_factory = sqlite3.Row
    dst = open_sqlite(dst_db)
    writer = CompactSidecarWriter(sidecar_dir)
    stats = CompactMigrationStats(str(src_db), str(dst_db), str(sidecar_dir))

    dst.execute("BEGIN")
    stats.positions = _copy_table(
        src,
        dst,
        "positions",
        [
            "position_id", "canonical_hash", "fast_hash", "packed_state", "side_to_move", "ply_min_seen",
            "ply_max_seen", "first_seen_at", "last_seen_at", "total_visits", "source_flags", "schema_version",
        ],
        "INSERT INTO positions(position_id, canonical_hash, fast_hash, packed_state, side_to_move, ply_min_seen, ply_max_seen, "
        "first_seen_at, last_seen_at, total_visits, source_flags, schema_version) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
    )
    stats.edges = _copy_table(
        src,
        dst,
        "edges",
        [
            "parent_position_id", "move_code_u8", "child_position_id", "visit_count", "p0_win_count", "p1_win_count",
            "draw_count", "first_seen_at", "last_seen_at",
        ],
        "INSERT INTO edges(parent_position_id, move_code_u8, child_position_id, visit_count, p0_win_count, p1_win_count, draw_count, first_seen_at, last_seen_at) "
        "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)",
    )

    game_rows = src.execute("SELECT * FROM games ORDER BY game_id").fetchall()
    for row in game_rows:
        source_id = _get_or_create_source(dst, row["source"])
        eng_id = _get_or_create_engine(dst, row["generator_engine_hash"])
        trunk_id = _get_or_create_trunk(dst, row["generator_trunk_hash"])
        cfg_id = _get_or_create_search_config(dst, row["search_config_hash"])
        dst.execute(
            "INSERT INTO games(game_id, start_position_id, result_code, move_count, generator_engine_version_id, generator_trunk_version_id, "
            "search_config_id, random_seed, worker_id, source_id, created_at, shard_id, game_metadata) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                row["game_id"], row["start_position_id"], outcome_to_code(row["result"]), row["move_count"], eng_id, trunk_id, cfg_id,
                row["random_seed"], row["worker_id"], source_id, row["created_at"], row["shard_id"], row["game_metadata"],
            ),
        )
    stats.games = len(game_rows)
    stats.labels_seen = int(src.execute("SELECT COUNT(*) FROM labels").fetchone()[0])

    game_path_rows = src.execute("SELECT * FROM game_paths").fetchall()
    dst.executemany(
        "INSERT INTO game_paths(game_id, packed_u8_move_sequence) VALUES(?, ?)",
        [(row["game_id"], row["packed_u8_move_sequence"]) for row in game_path_rows],
    )

    obs_rows = src.execute("SELECT * FROM observations").fetchall()
    for row in obs_rows:
        source_id = _get_or_create_source(dst, row["source_cohort"])
        last_eval_q15 = None
        if row["evaluation_summary"]:
            try:
                obj = json.loads(row["evaluation_summary"])
                last_eval_q15 = encode_scalar_q15(obj.get("last_value"))
            except Exception:
                last_eval_q15 = None
        dst.execute(
            "INSERT INTO observations(position_id, source_id, visit_count, p0_win_count, draw_count, p1_win_count, eval_count, last_eval_q15, first_seen, last_seen) "
            "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                row["position_id"], source_id, row["visit_count"], row["p0_wins"], row["draws"], row["p1_wins"],
                1 if last_eval_q15 is not None else 0, last_eval_q15, row["first_seen"], row["last_seen"],
            ),
        )
    stats.observations = len(obs_rows)

    label_rows = src.execute("SELECT * FROM labels ORDER BY label_id").fetchall()
    for row in label_rows:
        item = _convert_v1_label_row(row)
        label_set_id = _get_or_create_label_set(
            dst,
            label_type=item["label_type"],
            target_kind=item["target_kind"],
            source=item["source"],
            engine_hash=item["engine_hash"],
            trunk_hash=item["trunk_hash"],
            search_config_hash=item["search_config_hash"],
            label_schema_version=item["label_schema_version"],
            created_at=item["created_at"],
        )
        payload_ref_id = _get_or_create_payload_ref(dst, writer, label_type=item["label_type"], payload=item["payload"])
        _upsert_label_observation(dst, item, label_set_id, payload_ref_id is not None)
        _upsert_canonical_label(dst, item, label_set_id=label_set_id, payload_ref_id=payload_ref_id)

    stats.canonical_labels = int(dst.execute("SELECT COUNT(*) FROM canonical_labels").fetchone()[0])
    stats.label_observation_groups = int(dst.execute("SELECT COUNT(*) FROM label_observations").fetchone()[0])
    stats.payload_refs = int(dst.execute("SELECT COUNT(*) FROM payload_refs").fetchone()[0])

    import_rows = src.execute("SELECT * FROM imports").fetchall()
    if import_rows:
        dst.executemany(
            "INSERT INTO imports(import_id, source_path, source_hash, format, record_count, accepted_count, rejected_count, duplicate_count, "
            "started_at, completed_at, importer_version, status, error_report_path) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    row["import_id"], row["source_path"], row["source_hash"], row["format"], row["record_count"], row["accepted_count"],
                    row["rejected_count"], row["duplicate_count"], row["started_at"], row["completed_at"], row["importer_version"],
                    row["status"], row["error_report_path"],
                )
                for row in import_rows
            ],
        )
    rq_rows = src.execute("SELECT * FROM relabel_queue").fetchall()
    if rq_rows:
        dst.executemany(
            "INSERT INTO relabel_queue(queue_id, position_id, requested_label_type, requested_node_budget, priority, reason, required_engine_hash, "
            "required_trunk_hash, status, attempt_count, created_at, updated_at) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    row["queue_id"], row["position_id"], row["requested_label_type"], row["requested_node_budget"], row["priority"],
                    row["reason"], row["required_engine_hash"], row["required_trunk_hash"], row["status"], row["attempt_count"],
                    row["created_at"], row["updated_at"],
                )
                for row in rq_rows
            ],
        )

    dst.commit()
    dst.execute("VACUUM")
    dst.commit()
    src.close()
    dst.close()
    return {
        "migration": stats.__dict__,
        "score_semantics": score_semantics_report(),
    }


def _ensure_dbstat(conn: sqlite3.Connection) -> bool:
    try:
        conn.execute("CREATE VIRTUAL TABLE temp.dbstat USING dbstat(main)")
        return True
    except sqlite3.DatabaseError:
        return False


def _vacuum_copy_size(db_path: Path, drop_sql: Iterable[str] | None = None) -> int:
    temp_dir = Path(tempfile.mkdtemp(prefix="tiq_audit_"))
    work_db = temp_dir / "work.db"
    vacuum_db = temp_dir / "vacuum.db"
    try:
        shutil.copyfile(db_path, work_db)
        conn = sqlite3.connect(str(work_db))
        try:
            if drop_sql:
                for stmt in drop_sql:
                    conn.execute(stmt)
                conn.commit()
            quoted = str(vacuum_db).replace("'", "''")
            conn.execute(f"VACUUM INTO '{quoted}'")
        finally:
            conn.close()
        return vacuum_db.stat().st_size
    finally:
        for child in temp_dir.iterdir():
            child.unlink()
        temp_dir.rmdir()


def storage_audit(db_path: Path, *, vacuum_measure: bool = True) -> dict[str, Any]:
    conn = open_sqlite(db_path)
    page_size = int(conn.execute("PRAGMA page_size").fetchone()[0])
    page_count = int(conn.execute("PRAGMA page_count").fetchone()[0])
    freelist_count = int(conn.execute("PRAGMA freelist_count").fetchone()[0])
    total_bytes = db_path.stat().st_size if db_path.exists() else 0
    live_bytes = (page_count - freelist_count) * page_size
    free_bytes = freelist_count * page_size

    sqlite_objects = {
        row["name"]: row["type"]
        for row in conn.execute("SELECT name, type FROM sqlite_master WHERE type IN ('table','index')")
    }
    per_object: dict[str, dict[str, Any]] = {}
    byte_method = "dbstat"
    if _ensure_dbstat(conn):
        rows = conn.execute("SELECT name, SUM(pgsize) AS bytes FROM temp.dbstat GROUP BY name").fetchall()
        for row in rows:
            name = str(row["name"])
            per_object[name] = {
                "type": sqlite_objects.get(name, "other"),
                "bytes": int(row["bytes"] or 0),
            }
    else:
        byte_method = "vacuum-differential"
        baseline_vacuum_bytes = _vacuum_copy_size(db_path)
        explicit_indexes = [
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_autoindex_%' AND sql IS NOT NULL"
            ).fetchall()
        ]
        explicit_index_bytes: dict[str, int] = {}
        for index_name in explicit_indexes:
            size_without = _vacuum_copy_size(db_path, [f'DROP INDEX "{index_name}"'])
            explicit_index_bytes[index_name] = max(0, baseline_vacuum_bytes - size_without)
            per_object[index_name] = {"type": "index", "bytes": explicit_index_bytes[index_name]}
        table_names = [
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        ]
        for table_name in table_names:
            table_indexes = [
                row["name"]
                for row in conn.execute(f'PRAGMA index_list("{table_name}")').fetchall()
                if row["name"] in explicit_index_bytes
            ]
            drop_explicit = [f'DROP INDEX "{name}"' for name in table_indexes]
            size_without_explicit = _vacuum_copy_size(db_path, drop_explicit) if table_indexes else baseline_vacuum_bytes
            size_without_table = _vacuum_copy_size(db_path, drop_explicit + [f'DROP TABLE "{table_name}"'])
            per_object[table_name] = {
                "type": "table",
                "bytes": max(0, size_without_explicit - size_without_table),
                "explicit_index_bytes": sum(explicit_index_bytes.get(name, 0) for name in table_indexes),
            }

    def table_summary(name: str) -> dict[str, Any]:
        bytes_used = per_object.get(name, {}).get("bytes", 0)
        row_count = 0
        try:
            row_count = int(conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0])
        except sqlite3.DatabaseError:
            row_count = 0
        return {
            "rows": row_count,
            "bytes": bytes_used,
            "explicit_index_bytes": int(per_object.get(name, {}).get("explicit_index_bytes", 0)),
            "avg_bytes_per_row": (float(bytes_used) / row_count) if row_count else 0.0,
        }

    tables = {}
    for name, kind in sqlite_objects.items():
        if kind == "table" and not name.startswith("sqlite_"):
            tables[name] = table_summary(name)

    indexes = {
        name: data for name, data in per_object.items() if data.get("type") == "index"
    }

    label_distributions: dict[str, Any] = {}
    if "labels" in tables:
        label_rows = conn.execute(
            "SELECT label_type, COUNT(*) AS c, SUM(CASE WHEN value IS NULL THEN 1 ELSE 0 END) AS null_values, "
            "AVG(LENGTH(payload_json)) AS avg_payload, MAX(LENGTH(payload_json)) AS max_payload, "
            "SUM(LENGTH(payload_json)) AS payload_bytes, AVG(LENGTH(COALESCE(engine_hash,''))) AS avg_engine_len, "
            "AVG(LENGTH(COALESCE(trunk_hash,''))) AS avg_trunk_len, AVG(LENGTH(COALESCE(search_config_hash,''))) AS avg_cfg_len, "
            "AVG(LENGTH(COALESCE(source,''))) AS avg_source_len "
            "FROM labels GROUP BY label_type ORDER BY c DESC"
        ).fetchall()
        label_distributions = {
            "label_type_counts": [dict(row) for row in label_rows],
            "duplicate_identity_groups": int(
                conn.execute(
                    "SELECT COUNT(*) FROM (SELECT position_id, label_type, source, engine_hash, trunk_hash, search_config_hash, "
                    "COALESCE(best_move_u8,-1), COALESCE(nodes,-1), COALESCE(completed_depth,-1), COALESCE(selective_depth,-1), "
                    "COALESCE(bound,''), COALESCE(value, 9e99), COUNT(*) c FROM labels "
                    "GROUP BY 1,2,3,4,5,6,7,8,9,10,11,12 HAVING c > 1)"
                ).fetchone()[0]
            ),
            "duplicate_payload_groups": int(
                conn.execute(
                    "SELECT COUNT(*) FROM (SELECT payload_json, COUNT(*) c FROM labels WHERE payload_json IS NOT NULL GROUP BY payload_json HAVING c > 1)"
                ).fetchone()[0]
            ),
            "distinct_engine_hashes": int(conn.execute("SELECT COUNT(DISTINCT engine_hash) FROM labels").fetchone()[0]),
            "distinct_trunk_hashes": int(conn.execute("SELECT COUNT(DISTINCT trunk_hash) FROM labels").fetchone()[0]),
            "distinct_search_configs": int(conn.execute("SELECT COUNT(DISTINCT search_config_hash) FROM labels").fetchone()[0]),
            "distinct_sources": int(conn.execute("SELECT COUNT(DISTINCT source) FROM labels").fetchone()[0]),
        }
    elif "canonical_labels" in tables:
        label_distributions = {
            "label_set_count": int(conn.execute("SELECT COUNT(*) FROM label_sets").fetchone()[0]),
            "payload_ref_count": int(conn.execute("SELECT COUNT(*) FROM payload_refs").fetchone()[0]),
            "canonical_labels_by_type": [
                dict(row)
                for row in conn.execute(
                    "SELECT ls.label_type_code, COUNT(*) c, AVG(cl.confidence_u8) avg_conf, AVG(cl.observation_count) avg_obs "
                    "FROM canonical_labels cl JOIN label_sets ls ON ls.label_set_id=cl.label_set_id "
                    "GROUP BY ls.label_type_code ORDER BY c DESC"
                ).fetchall()
            ],
            "payload_bytes_total": int(
                conn.execute("SELECT COALESCE(SUM(raw_bytes),0) FROM payload_refs").fetchone()[0]
            ),
        }

    vacuum_bytes = None
    if vacuum_measure:
        temp_dir = Path(tempfile.mkdtemp(prefix="tiq_vacuum_"))
        try:
            vacuum_path = temp_dir / "vacuum.db"
            quoted = str(vacuum_path).replace("'", "''")
            conn.execute(f"VACUUM INTO '{quoted}'")
            vacuum_bytes = vacuum_path.stat().st_size
        finally:
            for child in temp_dir.iterdir():
                child.unlink()
            temp_dir.rmdir()

    sidecar_dir = db_path.with_suffix(".sidecars")
    sidecar_files = []
    sidecar_total_bytes = 0
    if sidecar_dir.exists():
        for path in sorted(sidecar_dir.glob("*")):
            if not path.is_file():
                continue
            size = path.stat().st_size
            sidecar_total_bytes += size
            sidecar_files.append({"path": str(path), "bytes": size})

    avg_bytes_per_label = 0.0
    avg_bytes_per_observation = 0.0
    if "labels" in tables and tables["labels"]["rows"]:
        avg_bytes_per_label = float(tables["labels"]["bytes"]) / tables["labels"]["rows"]
    elif "canonical_labels" in tables and tables["canonical_labels"]["rows"]:
        avg_bytes_per_label = float(tables["canonical_labels"]["bytes"]) / tables["canonical_labels"]["rows"]
    if "observations" in tables and tables["observations"]["rows"]:
        avg_bytes_per_observation = float(tables["observations"]["bytes"]) / tables["observations"]["rows"]

    conn.close()
    return {
        "database_path": str(db_path),
        "schema_kind": "compact-v2" if "canonical_labels" in tables else "position-store-v1",
        "database_bytes": total_bytes,
        "sidecar_total_bytes": sidecar_total_bytes,
        "hot_plus_sidecar_bytes": total_bytes + sidecar_total_bytes,
        "page_size": page_size,
        "page_count": page_count,
        "freelist_count": freelist_count,
        "live_bytes": live_bytes,
        "free_bytes": free_bytes,
        "vacuum_into_bytes": vacuum_bytes,
        "byte_accounting_method": byte_method,
        "average_bytes_per_label_row": avg_bytes_per_label,
        "average_bytes_per_observation_row": avg_bytes_per_observation,
        "tables": tables,
        "indexes": indexes,
        "label_distributions": label_distributions,
        "sidecar_files": sidecar_files,
    }


def export_training_binary(
    db_path: Path,
    *,
    out_path: Path,
    label_type_code_filter: int = LABEL_TYPE_TEACHER_VALUE,
    limit: int | None = None,
) -> dict[str, Any]:
    conn = open_sqlite(db_path)
    query = (
        "SELECT p.packed_state, p.total_visits, cl.target_kind, cl.eval_i16, cl.scalar_q15, cl.probability_u16, cl.best_move_u8, "
        "cl.flags_u16, cl.confidence_u8, cl.quality_rank, cl.outcome_code, ls.label_type_code "
        "FROM canonical_labels cl JOIN positions p ON p.position_id=cl.position_id "
        "JOIN label_sets ls ON ls.label_set_id=cl.label_set_id WHERE ls.label_type_code=? "
        "ORDER BY cl.confidence_u8 DESC, cl.quality_rank DESC, p.total_visits DESC"
    )
    if limit is not None:
        query += f" LIMIT {int(limit)}"
    rows = conn.execute(query, (label_type_code_filter,)).fetchall()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as handle:
        handle.write(EXPORT_MAGIC)
        handle.write(struct.pack("<HHI", COMPACT_EXPORT_SCHEMA_VERSION, label_type_code_filter, len(rows)))
        for row in rows:
            packed_state = bytes(row["packed_state"])
            if len(packed_state) != 24:
                raise ValueError("packed_state must remain 24 bytes in export")
            best_move = 255 if row["best_move_u8"] is None else int(row["best_move_u8"]) & 0xFF
            handle.write(packed_state)
            handle.write(
                struct.pack(
                    "<BhhHBBBHf",
                    int(row["target_kind"]),
                    int(row["eval_i16"] or 0),
                    int(row["scalar_q15"] or 0),
                    int(row["probability_u16"] or 0),
                    best_move,
                    int(row["confidence_u8"] or 0) & 0xFF,
                    int(row["quality_rank"] or 0) & 0xFF,
                    int(row["outcome_code"] or 0) & 0xFF,
                    float(max(1, int(row["total_visits"] or 1))) ** 0.5,
                )
            )
            handle.write(struct.pack("<H", int(row["flags_u16"] or 0) & 0xFFFF))
    conn.close()
    return {"rows": len(rows), "path": str(out_path)}
