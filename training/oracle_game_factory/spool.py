"""Filesystem-backed durable result spool."""
from __future__ import annotations

import gzip
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .protocol import atomic_write_bytes, read_json, utc_now, validate_game_payload

# archive/ only ever grows (every acknowledged game + its .ack.json sidecar
# lands there permanently) and status()->backpressure() used to re-walk it in
# full on every single authenticated /status call. Confirmed live, 2026-07-07:
# once archive passed ~118k files, that walk alone made GET /status hang past
# any reasonable client timeout (every unauthenticated request stayed
# instant, since it short-circuits before touching the spool at all) --
# every local monitor (training_status.py, the tunnel watchdog's own health
# probe notwithstanding, since that one doesn't send a token and never
# reaches this code) read the worker as unreachable even though the process
# and the SSH tunnel were both fine. archive's total only matters for the
# slow-moving warn/stop disk-usage thresholds, not for the real-time
# ready/tmp backlog signal, so cache it instead of re-scanning every call.
_ARCHIVE_SIZE_CACHE_TTL_SEC = 60.0


@dataclass
class SpoolConfig:
    root: Path
    warn_bytes: int = 10 * 1024**3
    stop_bytes: int = 20 * 1024**3


def _folder_bytes(folder: Path) -> int:
    total = 0
    if not folder.is_dir():
        return 0
    stack = [folder]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                for entry in it:
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(entry.path)
                    elif entry.is_file(follow_symlinks=False):
                        total += entry.stat().st_size
        except FileNotFoundError:
            continue
    return total


class DurableSpool:
    def __init__(self, cfg: SpoolConfig):
        self.cfg = cfg
        self.root = cfg.root
        self.tmp = self.root / "tmp"
        self.ready = self.root / "ready"
        self.archive = self.root / "archive"
        self.quarantine = self.root / "quarantine"
        for path in (self.tmp, self.ready, self.archive, self.quarantine):
            path.mkdir(parents=True, exist_ok=True)
        self._archive_bytes_cache: int = 0
        self._archive_bytes_cached_at: float = 0.0

    def _archive_bytes(self) -> int:
        now = time.time()
        if now - self._archive_bytes_cached_at > _ARCHIVE_SIZE_CACHE_TTL_SEC:
            self._archive_bytes_cache = _folder_bytes(self.archive)
            self._archive_bytes_cached_at = now
        return self._archive_bytes_cache

    def size_bytes(self) -> int:
        return _folder_bytes(self.ready) + _folder_bytes(self.tmp) + self._archive_bytes()

    def backpressure(self) -> dict[str, Any]:
        size = self.size_bytes()
        return {
            "bytes": size,
            "warn": size >= self.cfg.warn_bytes,
            "stop": size >= self.cfg.stop_bytes,
            "warn_bytes": self.cfg.warn_bytes,
            "stop_bytes": self.cfg.stop_bytes,
        }

    def result_path(self, game_id: str) -> Path:
        return self.ready / f"{game_id}.json.gz"

    def archive_path(self, game_id: str) -> Path:
        return self.archive / f"{game_id}.json.gz"

    def write_game(self, payload: dict[str, Any]) -> Path:
        errors = validate_game_payload(payload)
        if errors:
            raise ValueError("; ".join(errors))
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        data = gzip.compress(raw, compresslevel=6)
        path = self.result_path(str(payload["game_id"]))
        atomic_write_bytes(path, data)
        return path

    def list_ready(self, *, after: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for path in sorted(self.ready.glob("*.json.gz")):
            game_id = path.name.removesuffix(".json.gz")
            if after and game_id <= after:
                continue
            out.append(
                {
                    "game_id": game_id,
                    "bytes": path.stat().st_size,
                    "mtime": path.stat().st_mtime,
                    "path": path.name,
                }
            )
            if len(out) >= limit:
                break
        return out

    def read_game_bytes(self, game_id: str) -> bytes:
        path = self.result_path(game_id)
        if not path.is_file():
            archived = self.archive_path(game_id)
            if archived.is_file():
                return archived.read_bytes()
            raise FileNotFoundError(game_id)
        return path.read_bytes()

    def read_game_payload(self, game_id: str) -> dict[str, Any]:
        return json.loads(gzip.decompress(self.read_game_bytes(game_id)).decode("utf-8"))

    def ack(self, game_id: str) -> dict[str, Any]:
        src = self.result_path(game_id)
        dst = self.archive_path(game_id)
        if dst.is_file() and not src.is_file():
            return {"game_id": game_id, "status": "already_acknowledged"}
        if not src.is_file():
            return {"game_id": game_id, "status": "missing"}
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.replace(src, dst)
        meta = self.archive / f"{game_id}.ack.json"
        meta.write_text(json.dumps({"game_id": game_id, "acked_at": utc_now()}, indent=2), encoding="utf-8")
        return {"game_id": game_id, "status": "acknowledged"}

    def cleanup_archive(self, *, keep: int = 1000) -> int:
        files = sorted(self.archive.glob("*.json.gz"), key=lambda p: p.stat().st_mtime)
        removed = 0
        for path in files[:-keep]:
            path.unlink(missing_ok=True)
            ack = self.archive / f"{path.name.removesuffix('.json.gz')}.ack.json"
            ack.unlink(missing_ok=True)
            removed += 1
        return removed

