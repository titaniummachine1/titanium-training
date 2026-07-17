"""Localhost-only HTTP API for Oracle game factory."""
from __future__ import annotations

import argparse
import json
import shutil
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .generation import GenerationStore
from . import GAME_SCHEMA_VERSION, PROTOCOL_VERSION, WEIGHT_SCHEMA
from .protocol import game_payload_checksum, new_token, parse_website_game_wire, read_json, utc_now
from .supervisor import GameSupervisor, SupervisorConfig


def load_or_create_token(path: Path) -> str:
    if path.is_file():
        return path.read_text(encoding="utf-8").strip()
    token = new_token()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(token + "\n", encoding="utf-8")
    try:
        path.chmod(0o600)
    except Exception:
        pass
    return token


class ApiHandler(BaseHTTPRequestHandler):
    supervisor: GameSupervisor
    token: str
    website_submit_token: str

    def _auth(self) -> bool:
        header = self.headers.get("Authorization", "")
        return header == f"Bearer {self.token}"

    def _website_submit_auth(self) -> bool:
        token = self.headers.get("X-Website-Submit-Token", "")
        header = self.headers.get("Authorization", "")
        return (
            bool(self.website_submit_token)
            and (token == self.website_submit_token or header == f"Bearer {self.website_submit_token}")
        )

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Website-Submit-Token")

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        if self.path.startswith("/submit/"):
            self._cors()
        self.end_headers()
        self.wfile.write(raw)

    def _bytes(self, status: int, data: bytes, *, content_type: str = "application/octet-stream") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self) -> dict[str, Any]:
        n = int(self.headers.get("Content-Length", "0") or "0")
        if n <= 0:
            return {}
        return json.loads(self.rfile.read(n).decode("utf-8"))

    def _read_website_game(self) -> dict[str, Any]:
        n = int(self.headers.get("Content-Length", "0") or "0")
        if n <= 0:
            return {}
        return parse_website_game_wire(self.rfile.read(n))

    def do_OPTIONS(self) -> None:  # noqa: N802
        if urlparse(self.path).path == "/submit/website-game":
            self.send_response(204)
            self._cors()
            self.end_headers()
            return
        self.send_response(404)
        self.end_headers()

    def _website_result(self, body: dict[str, Any]) -> str:
        result = body.get("result")
        winner = body.get("winner")
        if result in (1, "1") or winner == "white":
            return "P0"
        if result in (-1, "-1") or winner == "black":
            return "P1"
        if result in (0, "0") or winner == "draw":
            return "DRAW"
        raise ValueError("need result 1/-1/0 or winner white/black/draw")

    def _spool_website_game(self, body: dict[str, Any]) -> dict[str, Any]:
        moves = body.get("moves")
        if not isinstance(moves, list) or not all(isinstance(m, str) and m for m in moves):
            raise ValueError("moves must be a non-empty string list")
        result = self._website_result(body)
        now = utc_now()
        metadata = body.get("metadata") if isinstance(body.get("metadata"), dict) else {}
        game_id = f"website-{uuid.uuid4().hex}"
        payload: dict[str, Any] = {
            "game_id": game_id,
            "protocol_version": PROTOCOL_VERSION,
            "schema_version": GAME_SCHEMA_VERSION,
            "weight_schema": WEIGHT_SCHEMA,
            "engine_build_hash": str(body.get("engine_build_hash") or "website-client"),
            "current_weight_hash": str(body.get("current_weight_hash") or "website-public"),
            "generation_id": "website-public",
            "matchup_type": "website_public",
            "worker_id": str(metadata.get("sessionId") or metadata.get("userAgent") or "website"),
            "seed": str(metadata.get("website_signature") or uuid.uuid4().hex),
            "moves": moves,
            "result": result,
            "termination_reason": "website_finished_game",
            "plies": len(moves),
            "time_control": metadata.get("timeControl") or {},
            "search": {
                "source": "website_finished_game",
                "players": metadata.get("players"),
                "engine_labels": metadata.get("engineLabels"),
            },
            "started_at": str(metadata.get("startedAt") or now),
            "finished_at": str(metadata.get("finishedAt") or now),
            "stats": {"website_metadata": metadata},
        }
        payload["payload_checksum"] = game_payload_checksum(payload)
        self.supervisor.spool.write_game(payload)
        return {"ok": True, "queued": True, "game_id": game_id, "result": result, "plies": len(moves)}

    def do_GET(self) -> None:  # noqa: N802
        if not self._auth():
            self._json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
            return
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)
        try:
            if path == "/health":
                self._json(200, {"ok": True})
            elif path == "/status":
                self._json(200, self.supervisor.status())
            elif path == "/generation":
                self._json(200, self.supervisor.generations.active_manifest())
            elif path == "/results":
                limit = int((qs.get("limit") or ["100"])[0])
                after = (qs.get("after") or [None])[0]
                self._json(200, {"results": self.supervisor.spool.list_ready(after=after, limit=limit)})
            elif path.startswith("/results/"):
                game_id = path.rsplit("/", 1)[1]
                self._bytes(200, self.supervisor.spool.read_game_bytes(game_id), content_type="application/gzip")
            else:
                self._json(404, {"error": "not found"})
        except Exception as exc:
            self._json(500, {"error": str(exc)})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/submit/website-game":
            if not self._website_submit_auth():
                self._json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
                return
            try:
                self._json(200, self._spool_website_game(self._read_website_game()))
            except Exception as exc:
                self._json(400, {"error": str(exc)})
            return

        if not self._auth():
            self._json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
            return
        try:
            body = self._read_json()
            if path == "/generation/stage":
                self._json(200, self.supervisor.generations.stage(Path(body["path"])))
            elif path == "/generation/activate":
                self._json(200, self.supervisor.generations.activate(str(body["generation_id"])))
            elif path == "/ack":
                game_ids = body.get("game_ids") or [body.get("game_id")]
                self._json(200, {"acks": [self.supervisor.spool.ack(str(g)) for g in game_ids if g]})
            elif path == "/pause":
                self.supervisor.pause()
                self._json(200, {"paused": True})
            elif path == "/resume":
                self.supervisor.resume()
                self._json(200, {"paused": False})
            elif path == "/drain":
                self.supervisor.pause()
                self._json(200, {"draining": True})
            else:
                self._json(404, {"error": "not found"})
        except Exception as exc:
            self._json(500, {"error": str(exc)})


def main() -> int:
    from prep_guard import guard_real_work

    guard_real_work("oracle_factory", detail="oracle_game_factory.server")
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--data-dir", type=Path, default=Path("/var/lib/titanium-game-factory"))
    ap.add_argument("--engine-bin", type=Path, default=Path("/opt/titanium-game-factory/engine/target/release/titanium"))
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--move-time", type=float, default=2.0)
    ap.add_argument("--node-budget", type=int, default=0,
                    help="Fixed node budget per move (0=disabled, use --move-time only)")
    args = ap.parse_args()

    cfg = SupervisorConfig(
        data_dir=args.data_dir,
        engine_bin=args.engine_bin,
        workers=args.workers,
        move_time=args.move_time,
        node_budget=args.node_budget,
    )
    supervisor = GameSupervisor(cfg)
    token = load_or_create_token(args.data_dir / "api_token")
    website_submit_token = load_or_create_token(args.data_dir / "website_submit_token")
    ApiHandler.supervisor = supervisor
    ApiHandler.token = token
    ApiHandler.website_submit_token = website_submit_token
    supervisor.start()
    httpd = ThreadingHTTPServer((args.host, args.port), ApiHandler)
    try:
        httpd.serve_forever()
    finally:
        supervisor.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
