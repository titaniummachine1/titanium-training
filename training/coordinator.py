#!/usr/bin/env python3
"""Localhost coordinator — single writer for manifest matchups + games DB.

Parallel match workers POST upserts here instead of fighting over manifest.json,
.ingested_offset sidecars, or sqlite from many processes.

  POST /api/matchup   upsert cumulative W/L for a pairing
  GET  /api/matchup   lookup prior a_wins / b_wins
  POST /api/game      insert one game into all_games.db (SQLite only)
  POST /api/claim-pairing  atomically pick next game for a free slot
  POST /api/release-remote free remote slot after crash/skip
  GET  /api/scoreboard
  GET  /health
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from datagen import DB_PATH, insert_single_game, insert_single_game_idempotent, validate_game
from color_rotation import claim_local_game
from opponent_curriculum import claim_game, load_state as load_curriculum, record_result
from manifest import format_scoreboard, format_scoreboard_compact, load_manifest, lookup_prior_wins, save_manifest, update_matchup
from swiss_tournament import (
    KA_TIME_CONTROLS,
    MAX_KA_PER_TC,
    OUR_TIME_CONTROLS,
    max_remote_parallel,
    pairing_slot_tag,
    pick_one_pairing,
    pairing_game_entry,
    pool_slots,
)

ROOT = Path(__file__).resolve().parent.parent

_lock = threading.Lock()
_active_remotes: dict[str, str] = {}  # game_id -> remote tc_b (short/medium/long)
_active_slots: dict[str, str] = {}  # game_id -> slot tag (ka:short, js, frozen:5s, …)
_ka_search_holders: dict[str, str | None] = {}  # tc_b -> game_id holding active go/search
_ka_search_since: dict[str, float] = {}  # tc_b -> monotonic when holder acquired
_ka_search_conds: dict[str, threading.Condition] = {}
DEFAULT_PORT = 8765
KA_SEARCH_ACQUIRE_TIMEOUT_SEC = 900
# Force-release Ka search lock if worker died mid-think (prevents 15min queue stalls).
KA_SEARCH_HOLD_MAX_SEC: dict[str, float] = {
    "intuition": 90,
    "short": 120,
    "medium": 180,
    "long": 240,
}


def _long_remotes_in_flight() -> int:
    return sum(1 for tc in _active_remotes.values() if tc == "long")


def _ka_tc_in_flight(tc: str) -> int:
    return sum(1 for t in _active_remotes.values() if t == tc)


def _ka_tc_slots_free() -> dict[str, bool]:
    return {tc: _ka_tc_in_flight(tc) < MAX_KA_PER_TC for tc in KA_TIME_CONTROLS}


def _ka_search_cond(tc_b: str) -> threading.Condition:
    if tc_b not in _ka_search_conds:
        _ka_search_conds[tc_b] = threading.Condition(_lock)
    return _ka_search_conds[tc_b]


def _evict_stale_ka_search(tc_b: str | None = None) -> list[str]:
    """Drop Ka search locks held longer than preset max (crashed worker / hung WS)."""
    now = time.monotonic()
    evicted: list[str] = []
    for tc in list(_ka_search_holders.keys()):
        if tc_b is not None and tc != tc_b:
            continue
        gid = _ka_search_holders.get(tc)
        if not gid:
            continue
        since = _ka_search_since.get(tc, now)
        max_hold = KA_SEARCH_HOLD_MAX_SEC.get(tc, 120)
        if now - since <= max_hold:
            continue
        _ka_search_holders[tc] = None
        _ka_search_since.pop(tc, None)
        evicted.append(f"{tc}:{gid}")
        _ka_search_cond(tc).notify_all()
    return evicted


def _acquire_ka_search(tc_b: str, game_id: str, timeout_sec: float = KA_SEARCH_ACQUIRE_TIMEOUT_SEC) -> dict:
    """FIFO-style master: one active Ka `go` per time preset."""
    cond = _ka_search_cond(tc_b)
    with _lock:
        _evict_stale_ka_search(tc_b)
        deadline = time.monotonic() + timeout_sec
        while _ka_search_holders.get(tc_b) not in (None, game_id):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return {"ok": False, "error": "ka search queue timeout"}
            cond.wait(timeout=remaining)
            _evict_stale_ka_search(tc_b)
        _ka_search_holders[tc_b] = game_id
        _ka_search_since[tc_b] = time.monotonic()
        return {"ok": True, "tc_b": tc_b, "game_id": game_id}


def _release_ka_search(tc_b: str, game_id: str) -> dict:
    cond = _ka_search_cond(tc_b)
    with _lock:
        if _ka_search_holders.get(tc_b) == game_id:
            _ka_search_holders[tc_b] = None
            _ka_search_since.pop(tc_b, None)
            cond.notify_all()
        return {"ok": True}


def _count_slots() -> dict:
    ka = {tc: 0 for tc in KA_TIME_CONTROLS}
    js = 0
    frozen = {tc: 0 for tc in OUR_TIME_CONTROLS}
    ti_pure_10s = 0
    self_10s = 0
    zero = 0
    for tag in _active_slots.values():
        if tag.startswith("ka:"):
            ka[tag[3:]] = ka.get(tag[3:], 0) + 1
        elif tag == "js":
            js += 1
        elif tag.startswith("frozen:"):
            tc = tag.split(":", 1)[1]
            frozen[tc] = frozen.get(tc, 0) + 1
        elif tag == "ti-pure:10s":
            ti_pure_10s += 1
        elif tag == "self:10s":
            self_10s += 1
        elif tag == "zero:adaptive":
            zero += 1
    return {
        "ka": ka,
        "js": js,
        "frozen": frozen,
        "ti_pure_10s": ti_pure_10s,
        "self_10s": self_10s,
        "zero": zero,
    }


def _claim_pairing() -> dict | None:
    global _active_remotes, _active_slots
    manifest = load_manifest()
    counts = _count_slots()
    pairing = pick_one_pairing(
        manifest,
        slot_counts=counts,
        n_pool_slots=pool_slots(),
    )
    if pairing is None:
        return None
    game_id = uuid.uuid4().hex[:8]
    entry = pairing_game_entry(pairing, game_id, ROOT / "training" / "data")
    if pairing.opponent_profile == "adaptive":
        curriculum = claim_game(pairing.engine_b)
        entry.update(curriculum)
        entry["opponent_profile"] = "adaptive"
        entry["source_tag"] = f"adaptive-{pairing.engine_b}"
        entry["display_label"] = (
            f"v15@5s vs {pairing.engine_b}@{curriculum['opponent_visits']}v"
        )
    elif pairing.kind == "local":
        entry.update(claim_local_game(entry["source_tag"]))
    _active_slots[game_id] = pairing_slot_tag(pairing)
    if pairing.kind == "remote":
        _active_remotes[game_id] = pairing.tc_b
        entry["release_remote"] = True
    return entry


def _release_game_slot(game_id: str | None = None) -> None:
    global _active_remotes, _active_slots
    if game_id:
        _active_slots.pop(game_id, None)
        _release_remote(game_id)
    elif _active_slots:
        gid = next(iter(_active_slots))
        _active_slots.pop(gid, None)
        _release_remote(gid)


def _release_remote(game_id: str | None = None) -> None:
    global _active_remotes
    if game_id:
        _active_remotes.pop(game_id, None)
        for tc in list(_ka_search_holders.keys()):
            if _ka_search_holders.get(tc) == game_id:
                _release_ka_search(tc, game_id)
    elif _active_remotes:
        gid, tc = next(iter(_active_remotes.items()))
        _active_remotes.pop(gid, None)
        if _ka_search_holders.get(tc) == gid:
            _release_ka_search(tc, gid)


def _record_game_done() -> None:
    manifest = load_manifest()
    t = manifest.setdefault("tournament", {})
    t["mode"] = "random-pool"
    t["games"] = int(t.get("games", 0)) + 1
    save_manifest(manifest)


class CoordinatorHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args) -> None:
        return

    def _send_json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length))

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"

        if path == "/health":
            self._send_json(200, {"ok": True, "pool": True, "version": 2})
            return

        if path == "/api/pool-status":
            from swiss_tournament import eligible_pairings
            with _lock:
                pool = eligible_pairings(load_manifest())
                local_n = sum(1 for p in pool if p.kind == "local")
                remote_n = sum(1 for p in pool if p.kind == "remote")
                slot_counts = _count_slots()
            self._send_json(200, {
                "pairings": len(pool),
                "local_pairings": local_n,
                "remote_pairings": remote_n,
                "remote_in_flight": len(_active_remotes),
                "slots_in_flight": len(_active_slots),
                "slot_counts": slot_counts,
                "ka_long_in_flight": _long_remotes_in_flight(),
                "ka_per_tc_in_flight": {tc: _ka_tc_in_flight(tc) for tc in KA_TIME_CONTROLS},
                "ka_per_tc_max": MAX_KA_PER_TC,
                "ka_search_holders": dict(_ka_search_holders),
                "max_remote_parallel": max_remote_parallel(),
                "curriculum": load_curriculum(),
            })
            return

        if path == "/api/matchup":
            q = parse_qs(parsed.query)
            def one(key: str) -> str | None:
                v = q.get(key)
                return v[0] if v else None

            with _lock:
                a_w, b_w = lookup_prior_wins(
                    one("engine_a") or "",
                    one("engine_b") or "",
                    one("tc_a"),
                    one("tc_b"),
                )
            self._send_json(200, {"a_wins": a_w, "b_wins": b_w})
            return

        if path == "/api/scoreboard":
            q = parse_qs(parsed.query)
            compact = (q.get("compact") or ["0"])[0] in ("1", "true", "yes")
            with _lock:
                manifest = load_manifest()
                text = format_scoreboard_compact(manifest) if compact else format_scoreboard(manifest)
            self._send_json(200, {"text": text, "compact": compact})
            return

        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        body = self._read_json()

        if path == "/api/matchup":
            required = ("engine_a", "engine_b", "a_wins", "b_wins")
            if any(k not in body for k in required):
                self._send_json(400, {"error": f"need {required}"})
                return
            with _lock:
                entry = update_matchup(
                    body["engine_a"],
                    body["engine_b"],
                    int(body["a_wins"]),
                    int(body["b_wins"]),
                    body.get("tc_a"),
                    body.get("tc_b"),
                    games_file=body.get("games_file"),
                    source=body.get("source"),
                )
            self._send_json(200, entry)
            return

        if path == "/api/claim-pairing":
            with _lock:
                entry = _claim_pairing()
            if entry is None:
                self._send_json(503, {"error": "no pairing available"})
                return
            self._send_json(200, entry)
            return

        if path == "/api/release-remote":
            game_id = body.get("game_id")
            with _lock:
                _release_game_slot(game_id)
            self._send_json(200, {"ok": True})
            return

        if path == "/api/ka-search-acquire":
            tc_b = body.get("tc_b") or body.get("tc")
            game_id = body.get("game_id")
            if not tc_b or not game_id:
                self._send_json(400, {"error": "need tc_b and game_id"})
                return
            timeout = float(body.get("timeout_sec") or KA_SEARCH_ACQUIRE_TIMEOUT_SEC)
            result = _acquire_ka_search(str(tc_b), str(game_id), timeout)
            code = 200 if result.get("ok") else 503
            self._send_json(code, result)
            return

        if path == "/api/ka-search-release":
            tc_b = body.get("tc_b") or body.get("tc")
            game_id = body.get("game_id")
            if not tc_b or not game_id:
                self._send_json(400, {"error": "need tc_b and game_id"})
                return
            result = _release_ka_search(str(tc_b), str(game_id))
            self._send_json(200, result)
            return

        if path == "/api/game":
            moves = body.get("moves")
            result = body.get("result")
            if not isinstance(moves, list) or result not in ("W", "B"):
                self._send_json(400, {"error": "need moves[] and result W|B"})
                return
            outcome = 1 if result == "W" else -1
            err = validate_game(moves, outcome)
            if err:
                self._send_json(400, {"error": err})
                return
            tag = body.get("tag") or body.get("source_tag") or ""
            opponent = body.get("curriculum_opponent")
            if opponent in ("ka", "zero") and not isinstance(body.get("our_win"), bool):
                self._send_json(400, {"error": "adaptive game needs boolean our_win"})
                return

            with _lock:
                try:
                    request_id = str(body.get("game_id") or "")
                    if request_id:
                        gid, inserted = insert_single_game_idempotent(
                            moves,
                            outcome,
                            request_id=request_id,
                            out_path=DB_PATH,
                            tag=tag,
                        )
                    else:
                        gid = insert_single_game(moves, outcome, DB_PATH, tag)
                        inserted = True
                except ValueError as e:
                    self._send_json(400, {"error": str(e)})
                    return
                if body.get("release_remote") or body.get("game_id"):
                    _release_game_slot(body.get("game_id"))
                curriculum_update = None
                if inserted and opponent in ("ka", "zero"):
                    curriculum_update = record_result(
                        opponent,
                        our_win=body["our_win"],
                        game_id=request_id,
                        visits=int(body.get("opponent_visits") or 1),
                    )
                if inserted:
                    _record_game_done()
            self._send_json(200, {
                "ok": True,
                "game_id": gid,
                "inserted": inserted,
                "curriculum": curriculum_update,
            })
            return

        self._send_json(404, {"error": "not found"})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    save_manifest(load_manifest())

    server = ThreadingHTTPServer((args.host, args.port), CoordinatorHandler)
    print(f"coordinator listening on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
