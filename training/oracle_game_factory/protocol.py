"""Shared protocol helpers for Oracle game-factory and laptop importer."""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import GAME_SCHEMA_VERSION, PROTOCOL_VERSION, WEIGHT_SCHEMA

try:
    from titanium_training.store.website_games import parse_text_wire
except ModuleNotFoundError:
    parse_text_wire = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def atomic_write_json(path: Path, obj: dict[str, Any]) -> None:
    atomic_write_bytes(path, (json.dumps(obj, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def parse_website_game_wire(raw: bytes) -> dict[str, Any]:
    text = raw.decode("utf-8").strip()
    if not text:
        return {}
    if text.startswith("{"):
        return json.loads(text)
    if parse_text_wire is None:
        raise ValueError("text website-game wire requires titanium_training parser")
    parsed = parse_text_wire(text)
    return {"moves": list(parsed.moves), "result": parsed.result, "source": parsed.source}


def new_token() -> str:
    return secrets.token_urlsafe(32)


def game_payload_checksum(payload: dict[str, Any]) -> str:
    clone = dict(payload)
    clone.pop("payload_checksum", None)
    raw = json.dumps(clone, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(raw)


def validate_game_payload(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = [
        "game_id",
        "protocol_version",
        "schema_version",
        "engine_build_hash",
        "current_weight_hash",
        "generation_id",
        "matchup_type",
        "worker_id",
        "seed",
        "moves",
        "result",
        "termination_reason",
        "plies",
        "time_control",
        "search",
        "started_at",
        "finished_at",
        "payload_checksum",
    ]
    for key in required:
        if key not in payload:
            errors.append(f"missing {key}")
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        errors.append("protocol_version mismatch")
    if payload.get("schema_version") != GAME_SCHEMA_VERSION:
        errors.append("schema_version mismatch")
    moves = payload.get("moves")
    if not isinstance(moves, list) or not all(isinstance(m, str) and m for m in moves):
        errors.append("moves must be non-empty string list")
    if payload.get("plies") != len(moves or []):
        errors.append("plies does not match moves length")
    if payload.get("result") not in ("P0", "P1", "DRAW"):
        errors.append("result must be P0, P1, or DRAW")
    checksum = payload.get("payload_checksum")
    if checksum and checksum != game_payload_checksum(payload):
        errors.append("payload_checksum mismatch")
    return errors


def make_generation_manifest(
    *,
    generation_id: str,
    current_path: Path,
    prior_path: Path,
    engine_build_hash: str,
    search: dict[str, Any],
    source_promotion_epoch: int | None,
) -> dict[str, Any]:
    current_hash = sha256_file(current_path)
    prior_hash = sha256_file(prior_path)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "generation_id": generation_id,
        "current_deployed_hash": current_hash,
        "prior_deployed_hash": prior_hash,
        "prior_is_distinct": current_hash != prior_hash,
        "engine_build_hash": engine_build_hash,
        "weight_schema": WEIGHT_SCHEMA,
        "search_settings": search,
        "created_at": utc_now(),
        "source_promotion_epoch": source_promotion_epoch,
    }
