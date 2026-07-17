#!/usr/bin/env python3
"""Build a frozen list of paired openings legal in BOTH Claustrophobia and Titanium.

Every candidate line is replayed from standard start:
  1) each ply must be in Claustrophobia /api/state legal set;
  2) full prefix must EngineSession.sync through Titanium.

Output: openings manifest with stable opening_id + content hash.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "training"))

from engine_session import EngineSession  # noqa: E402

OURS = os.environ.get("CLAUSTRO_HTTP", "http://127.0.0.1:9171")
DEFAULT_ENGINE = (
    REPO / "engine" / "target-catv5-accepted-03856fe" / "release" / "titanium.exe"
)
DEFAULT_WEIGHTS = REPO / "training" / "runs" / "v16" / "accepted" / "epoch_0002.bin"
OUT_DEFAULT = Path(__file__).resolve().parent / "frozen_openings" / "claustro_titanium_openings_v1.json"

# Candidate pool — only lines that pass dual validation are kept.
CANDIDATE_LINES: list[tuple[str, ...]] = [
    (),
    ("e2", "e8"),
    ("e2", "e8", "e3", "e7"),
    ("e2", "e8", "d2", "d8"),
    ("e2", "e8", "f2", "f8"),
    ("d1", "d9"),
    ("f1", "f9"),
    ("e2", "d9"),
    ("e2", "f9"),
    ("d1", "e8"),
    ("f1", "e8"),
    ("d1", "f9"),
    ("f1", "d9"),
    ("e2", "e8", "e3", "d7"),
    ("e2", "e8", "e3", "f7"),
    ("e2", "e8", "d2", "e7"),
    ("e2", "e8", "f2", "e7"),
    ("e2", "e8", "e3", "e7", "d2", "d8"),
    ("e2", "e8", "e3", "e7", "f2", "f8"),
    ("e2", "e8", "d2", "d8", "e3", "e7"),
    ("e2", "e8", "f2", "f8", "e3", "e7"),
    ("e2", "e8", "e3", "e7", "e4", "e6"),
    ("d1", "d9", "e2", "e8"),
    ("f1", "f9", "e2", "e8"),
    ("e2", "e8", "d3", "d7"),
    ("e2", "e8", "f3", "f7"),
    ("e2", "e8", "c2", "c8"),
    ("e2", "e8", "g2", "g8"),
    ("e2", "e8", "e3h"),
    ("e2", "e8", "d3h"),
    ("e2", "e8", "f3h"),
    ("e2", "e8", "e2v"),
    ("e2", "e8", "d2v"),
    ("e2", "e8", "f2v"),
]


def post(path: str, obj: dict | None = None) -> dict:
    data = json.dumps(obj or {}).encode()
    req = urllib.request.Request(
        OURS + path, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def get(path: str) -> dict:
    with urllib.request.urlopen(OURS + path, timeout=30) as r:
        return json.loads(r.read())


def legal_notations(st: dict) -> set[str]:
    out: set[str] = set()
    for item in st.get("legal") or []:
        if isinstance(item, str):
            out.add(item)
        elif isinstance(item, dict) and "notation" in item:
            out.add(str(item["notation"]))
    return out


def expand_candidates_bfs(max_lines: int = 80, max_depth: int = 4) -> list[tuple[str, ...]]:
    """Grow candidate openings by dual-blind BFS on Claustrophobia legal moves only.
    Titanium validation still happens later for each accepted line.
    """
    lines: list[tuple[str, ...]] = [()]
    seen = {()}
    queue: list[tuple[str, ...]] = [()]
    while queue and len(lines) < max_lines:
        prefix = queue.pop(0)
        if len(prefix) >= max_depth:
            continue
        post("/api/new", {})
        for mv in prefix:
            post("/api/move", {"move": mv})
        st = get("/api/state")
        if st.get("terminal"):
            continue
        # Prefer pawn replies, then a few walls, for diversity without explosion.
        legal = sorted(legal_notations(st))
        pawns = [m for m in legal if len(m) == 2]
        walls = [m for m in legal if len(m) == 3]
        choices = pawns[:6] + walls[:4]
        for mv in choices:
            nxt = prefix + (mv,)
            if nxt in seen:
                continue
            seen.add(nxt)
            lines.append(nxt)
            queue.append(nxt)
            if len(lines) >= max_lines:
                break
    return lines


def validate_line_claustrophobia(moves: tuple[str, ...]) -> tuple[bool, str]:
    post("/api/new", {})
    for i, mv in enumerate(moves):
        st = get("/api/state")
        legal = legal_notations(st)
        if mv not in legal:
            return False, f"claustrophobia_illegal_at_ply_{i}:{mv}"
        r = post("/api/move", {"move": mv})
        if r.get("error"):
            return False, f"claustrophobia_reject_at_ply_{i}:{mv}:{r.get('error')}"
    return True, "ok"


def validate_line_titanium(sess: EngineSession, moves: tuple[str, ...]) -> tuple[bool, str]:
    # Incremental prefixes must all sync.
    if not sess.sync([]):
        return False, "titanium_sync_startpos_failed"
    for i in range(1, len(moves) + 1):
        prefix = list(moves[:i])
        if not sess.sync(prefix):
            return False, f"titanium_illegal_or_sync_fail_at_ply_{i-1}:{moves[i-1]}"
    return True, "ok"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=20)
    ap.add_argument("--titanium-bin", type=Path, default=DEFAULT_ENGINE)
    ap.add_argument("--titanium-weights", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--version", default="claustro-titanium-openings-v1")
    args = ap.parse_args()

    # Connectivity
    try:
        get("/api/state")
    except Exception as e:
        print(f"BLOCKED: Claustrophobia not reachable at {OURS}: {e}", file=sys.stderr)
        return 2

    sess = EngineSession(
        "titanium-v17",
        args.titanium_weights,
        threads=1,
        engine_bin=args.titanium_bin,
    )
    accepted: list[dict] = []
    rejected: list[dict] = []
    try:
        pool = list(CANDIDATE_LINES) + expand_candidates_bfs(max_lines=100, max_depth=4)
        seen: set[tuple[str, ...]] = set()
        for line in pool:
            if line in seen:
                continue
            seen.add(line)
            ok_c, reason_c = validate_line_claustrophobia(line)
            if not ok_c:
                rejected.append({"moves": list(line), "reason": reason_c})
                continue
            ok_t, reason_t = validate_line_titanium(sess, line)
            if not ok_t:
                rejected.append({"moves": list(line), "reason": reason_t})
                continue
            opening_id = f"open-{len(accepted):04d}"
            accepted.append(
                {
                    "opening_id": opening_id,
                    "moves": list(line),
                    "plies": len(line),
                    "validated_by": ["claustrophobia_legal_replay", "titanium_session_sync"],
                }
            )
            if len(accepted) >= args.count:
                break
    finally:
        sess.close()

    if len(accepted) < args.count:
        print(
            f"BLOCKED: only {len(accepted)}/{args.count} dual-legal openings; "
            f"rejected={len(rejected)}",
            file=sys.stderr,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        (args.out.parent / "rejected_candidates.json").write_text(
            json.dumps(rejected, indent=2) + "\n", encoding="utf-8"
        )
        return 3

    payload = {
        "version": args.version,
        "n_openings": len(accepted),
        "claustrophobia_http": OURS,
        "titanium_bin": str(args.titanium_bin),
        "titanium_weights": str(args.titanium_weights),
        "validation": {
            "from_standard_start": True,
            "claustrophobia_legal_each_ply": True,
            "titanium_session_sync_each_prefix": True,
        },
        "openings": accepted,
        "rejected_count": len(rejected),
    }
    # Manifest hash over stable content (ids + moves), not absolute paths.
    hash_body = {
        "version": payload["version"],
        "openings": [{"opening_id": o["opening_id"], "moves": o["moves"]} for o in accepted],
    }
    blob = json.dumps(hash_body, sort_keys=True, separators=(",", ":")).encode()
    payload["manifest_sha256"] = hashlib.sha256(blob).hexdigest()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (args.out.parent / "rejected_candidates.json").write_text(
        json.dumps(rejected, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "ok": True,
                "n": len(accepted),
                "manifest_sha256": payload["manifest_sha256"],
                "out": str(args.out),
                "rejected": len(rejected),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
