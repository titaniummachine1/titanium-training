"""Laptop-side Oracle result pull/import/ack client.

The laptop remains the training authority.  This module downloads Oracle game
records through the SSH tunnel, validates them, imports into canonical SQLite,
synchronizes the active teacher dataset, and only then acknowledges Oracle.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sqlite3
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_TRAINING = Path(__file__).resolve().parent
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

from db_import import (
    GAMES_DB_PATH,
    GAMES_SCHEMA,
    LABELS_DB_PATH,
    LABELS_SCHEMA,
    ensure_game_line_hashes,
    make_game_line_hash,
    open_db,
    write_batch,
)
from oracle_game_factory import GAME_SCHEMA_VERSION, PROTOCOL_VERSION, WEIGHT_SCHEMA
from oracle_game_factory.protocol import game_payload_checksum, validate_game_payload
from sync_overnight_to_teacher import load_synced_ids, pool_teacher_dir, sync_single_game


@dataclass
class OracleClientConfig:
    base_url: str = "http://127.0.0.1:8765"
    token: str = ""
    inbox: Path = _TRAINING / "data" / "oracle_inbox"
    batch_limit: int = 25
    poll_sec: float = 30.0
    # Optional external locks to serialize DB and teacher-dataset writes with the
    # host pool's own _db_lock / _teacher_lock, preventing parquet corruption.
    db_lock: object = None
    teacher_lock: object = None


class OracleHttpClient:
    def __init__(self, cfg: OracleClientConfig):
        self.cfg = cfg

    def _request(self, method: str, path: str, data: dict[str, Any] | None = None) -> bytes:
        raw = None if data is None else json.dumps(data).encode("utf-8")
        req = urllib.request.Request(
            self.cfg.base_url.rstrip("/") + path,
            data=raw,
            method=method,
            headers={
                "Authorization": f"Bearer {self.cfg.token}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.read()

    def json(self, method: str, path: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
        return json.loads(self._request(method, path, data).decode("utf-8"))

    def bytes(self, path: str) -> bytes:
        return self._request("GET", path)


def _game_exists(game_id: str) -> bool:
    if not GAMES_DB_PATH.is_file():
        return False
    con = sqlite3.connect(str(GAMES_DB_PATH), timeout=30)
    try:
        return con.execute("SELECT 1 FROM games WHERE game_id=?", (game_id,)).fetchone() is not None
    finally:
        con.close()


def _line_hash_exists(moves: list[str]) -> bool:
    """True if this exact move sequence is already stored under a different
    game_id. Common for early/common opening lines from independent workers --
    NOT a validation failure, just a duplicate to skip (and still ack, so the
    Oracle FIFO queue doesn't get stuck re-serving it forever)."""
    if not GAMES_DB_PATH.is_file():
        return False
    con = sqlite3.connect(str(GAMES_DB_PATH), timeout=30)
    try:
        ensure_game_line_hashes(con)
        line_hash = make_game_line_hash(moves)
        return con.execute(
            "SELECT 1 FROM game_line_hashes WHERE line_hash=?", (line_hash,)
        ).fetchone() is not None
    finally:
        con.close()


def _result_to_outcome_p0(result: str) -> int:
    if result == "P0":
        return 1
    if result == "P1":
        return -1
    return 0


def validate_remote_game(payload: dict[str, Any]) -> list[str]:
    errors = validate_game_payload(payload)
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        errors.append("unsupported protocol")
    if payload.get("schema_version") != GAME_SCHEMA_VERSION:
        errors.append("unsupported game schema")
    if payload.get("weight_schema") and payload.get("weight_schema") != WEIGHT_SCHEMA:
        errors.append("unsupported weight schema")
    return errors


def import_remote_game(payload: dict[str, Any], *, db_lock=None, teacher_lock=None) -> dict[str, Any]:
    errors = validate_remote_game(payload)
    if errors:
        raise ValueError("; ".join(errors))
    game_id = str(payload["game_id"])
    moves = list(payload["moves"])
    matchup_type = str(payload["matchup_type"])
    if matchup_type == "website_public":
        source = "website_public"
    else:
        source = "oracle_mixed" if matchup_type != "current_current" else "oracle_selfplay"

    def _do_db_write() -> bool:
        """Returns True only when this exact move line is already stored under
        a DIFFERENT game_id (a duplicate -- common for shared opening lines
        from independent workers). That case must skip teacher sync too,
        since sync_single_game() looks the game up by this exact game_id,
        which was deliberately never inserted. Ack it anyway below so
        Oracle's FIFO queue can move on instead of re-serving it forever.

        When _game_exists(game_id) is already True, the row IS present under
        this game_id (e.g. a previous run finished the db write but crashed
        before teacher sync) -- fall through so teacher sync still runs/retries."""
        if _game_exists(game_id):
            return False
        if _line_hash_exists(moves):
            return True
        games_db = open_db(GAMES_DB_PATH, GAMES_SCHEMA)
        labels_db = open_db(LABELS_DB_PATH, LABELS_SCHEMA)
        try:
            written_games, _written_positions, _written_labels = write_batch(
                games_db,
                labels_db,
                [(game_id, moves, _result_to_outcome_p0(str(payload["result"])), None, source)],
                chunk_size=512,
                workers=1,
            )
        finally:
            games_db.close()
            labels_db.close()
        if written_games <= 0:
            raise ValueError(f"engine eval-batch rejected remote game {game_id}")
        return False

    if db_lock is not None:
        with db_lock:
            is_duplicate = _do_db_write()
    else:
        is_duplicate = _do_db_write()

    if is_duplicate:
        return {"game_id": game_id, "synced": {"game_id": game_id, "new_positions": 0, "duplicate": True}}

    def _do_teacher_sync():
        if game_id not in load_synced_ids():
            return sync_single_game(game_id, dataset_dir=pool_teacher_dir())
        return {"game_id": game_id, "new_positions": 0, "skipped": True}

    if teacher_lock is not None:
        with teacher_lock:
            sync_result = _do_teacher_sync()
    else:
        sync_result = _do_teacher_sync()

    return {"game_id": game_id, "synced": sync_result}


def pull_once(cfg: OracleClientConfig) -> dict[str, Any]:
    cfg.inbox.mkdir(parents=True, exist_ok=True)
    client = OracleHttpClient(cfg)
    listing = client.json("GET", f"/results?limit={cfg.batch_limit}")
    imported: list[str] = []
    failed: list[dict[str, str]] = []
    for item in listing.get("results", []):
        game_id = item["game_id"]
        try:
            data = client.bytes(f"/results/{game_id}")
            inbox_file = cfg.inbox / f"{game_id}.json.gz"
            inbox_file.write_bytes(data)
            payload = json.loads(gzip.decompress(data).decode("utf-8"))
            result = import_remote_game(payload, db_lock=cfg.db_lock, teacher_lock=cfg.teacher_lock)
            client.json("POST", "/ack", {"game_id": game_id})
            imported.append(
                {
                    "game_id": game_id,
                    "new_positions": int(result.get("synced", {}).get("new_positions", 0) or 0),
                    "generation_id": payload.get("generation_id"),
                    "matchup_type": payload.get("matchup_type"),
                }
            )
        except Exception as exc:
            failed.append({"game_id": game_id, "error": str(exc)})
    return {"imported": imported, "failed": failed, "seen": len(listing.get("results", []))}


class OracleImportThread(threading.Thread):
    def __init__(self, cfg: OracleClientConfig, on_import=None):
        super().__init__(daemon=True, name="oracle-import")
        self.cfg = cfg
        self.on_import = on_import
        self.stop_event = threading.Event()
        self.last_result: dict[str, Any] = {}

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                self.last_result = pull_once(self.cfg)
                if self.on_import:
                    for item in self.last_result.get("imported", []):
                        self.on_import(item)
            except Exception as exc:
                self.last_result = {"error": str(exc)}
            self.stop_event.wait(self.cfg.poll_sec)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8765")
    ap.add_argument("--token", required=True)
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--poll-sec", type=float, default=30.0)
    args = ap.parse_args()
    cfg = OracleClientConfig(base_url=args.url, token=args.token, batch_limit=args.limit, poll_sec=args.poll_sec)
    if not args.watch:
        print(json.dumps(pull_once(cfg), indent=2))
        return 0
    while True:
        print(json.dumps(pull_once(cfg), indent=2))
        time.sleep(args.poll_sec)


if __name__ == "__main__":
    raise SystemExit(main())
