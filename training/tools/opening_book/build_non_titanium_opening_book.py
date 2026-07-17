#!/usr/bin/env python3
"""Build a 15-ply opening book from non-Titanium engine games in game_store.db.

Filters out any game where Titanium (or v15/v16 aliases) appears in the source tag
or in website game metadata players.

Outputs:
  training/data/opening_book/non_titanium_opening_dag.db — position DAG (packed state + u8 edges + win rate)
  training/data/opening_book/non_titanium_10ply.json  — JSON mirror for inspection
  training/data/opening_book/non_titanium_book_lines.txt — human-readable main lines
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from titanium_training.store.config import GAME_STORE_DB, TRAINING_DIR
from titanium_training.store.state import (
    PositionState,
    apply_move,
    encode_move,
    moves_from_u8_blob,
)

OUT_DIR = TRAINING_DIR / "data" / "opening_book"
DEFAULT_DAG = OUT_DIR / "non_titanium_opening_dag.db"
DEFAULT_JSON = OUT_DIR / "non_titanium_10ply.json"
DEFAULT_LINES = OUT_DIR / "non_titanium_book_lines.txt"

DAG_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS positions (
    position_id   INTEGER PRIMARY KEY,
    ply           INTEGER NOT NULL,
    side_to_move  INTEGER NOT NULL,
    packed_state  BLOB NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS edges (
    parent_position_id INTEGER NOT NULL REFERENCES positions(position_id),
    move_code_u8       INTEGER NOT NULL CHECK(move_code_u8 BETWEEN 0 AND 255),
    child_position_id  INTEGER NOT NULL REFERENCES positions(position_id),
    visit_count        INTEGER NOT NULL DEFAULT 0,
    wins_stm           INTEGER NOT NULL DEFAULT 0,
    losses_stm         INTEGER NOT NULL DEFAULT 0,
    draws              INTEGER NOT NULL DEFAULT 0,
    win_rate           REAL NOT NULL DEFAULT 0.5,
    PRIMARY KEY (parent_position_id, move_code_u8)
);

CREATE TABLE IF NOT EXISTS path_codes (
    position_id INTEGER PRIMARY KEY REFERENCES positions(position_id),
    path_u8     BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS metadata (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_edges_parent ON edges(parent_position_id);
CREATE INDEX IF NOT EXISTS idx_positions_ply ON positions(ply);
"""

MAX_PLIES = 15
MOVE_RE = re.compile(r"^[a-i][1-9][hv]?$")

# Refuted wall-fest — drop from DAG when White plays these at exact prefixes.
# Black may keep e3h etc. in other lines. Line: e2 e8 e3 e7 e4 e6 h3h e6h e3h …
DENIED_WHITE_BOOK_MOVES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("e2", "e8", "e3", "e7", "e4", "e6"), "h3h"),
    (("e2", "e8", "e3", "e7", "e4", "e6", "h3h", "e6h"), "e3h"),
    (
        ("e2", "e8", "e3", "e7", "e4", "e6", "h3h", "e6h", "e3h", "c6h"),
        "g2h",
    ),
)

# Refuted Black continuation — cut DAG after g3h (no f6).
DENIED_BLACK_BOOK_MOVES: tuple[tuple[tuple[str, ...], str], ...] = (
    (
        (
            "e2",
            "e8",
            "e3",
            "e7",
            "e4",
            "e6",
            "e3h",
            "f6h",
            "c3h",
            "d6h",
            "g3h",
        ),
        "f6",
    ),
)


def white_book_move_denied(moves: list[str], side_to_move: int, next_move: str) -> bool:
    """side_to_move: 0 = White (matches engine GameState.turn)."""
    if side_to_move != 0:
        return False
    for prefix, denied in DENIED_WHITE_BOOK_MOVES:
        if len(moves) == len(prefix) and moves == list(prefix) and next_move == denied:
            return True
    return False


def black_book_move_denied(moves: list[str], side_to_move: int, next_move: str) -> bool:
    if side_to_move != 1:
        return False
    for prefix, denied in DENIED_BLACK_BOOK_MOVES:
        if len(moves) == len(prefix) and moves == list(prefix) and next_move == denied:
            return True
    return False


def opening_book_move_denied(moves: list[str], side_to_move: int, next_move: str) -> bool:
    return white_book_move_denied(
        moves, side_to_move, next_move
    ) or black_book_move_denied(moves, side_to_move, next_move)


def touches_denied_opening_prefix(moves: list[str]) -> bool:
    """Drop whole games that follow the refuted White wall-fest mainline."""
    wall_fest = (
        "e2",
        "e8",
        "e3",
        "e7",
        "e4",
        "e6",
        "h3h",
        "e6h",
        "e3h",
    )
    if len(moves) >= len(wall_fest) and moves[: len(wall_fest)] == list(wall_fest):
        return True
    return False

# Titanium branding in this repo: explicit name or v15/v16 production aliases.
TITANIUM_RE = re.compile(
    r"titanium|(?:^|[-_/])v1[56](?:[-_/]|$)|grafted|titanium-v\d+",
    re.IGNORECASE,
)


@dataclass
class MoveStats:
    games: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0

    def record(self, outcome_for_mover: int) -> None:
        self.games += 1
        if outcome_for_mover > 0:
            self.wins += 1
        elif outcome_for_mover < 0:
            self.losses += 1
        else:
            self.draws += 1

    @property
    def win_rate(self) -> float:
        decided = self.wins + self.losses
        if decided == 0:
            return 0.5
        return self.wins / decided

    def to_dict(self) -> dict[str, Any]:
        return {
            "games": self.games,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "winRate": round(self.win_rate, 4),
        }


@dataclass
class BookNode:
    """Prefix position → stats for each child move."""

    moves: dict[str, MoveStats] = field(default_factory=lambda: defaultdict(MoveStats))
    total_games: int = 0

    def to_dict(self) -> dict[str, Any]:
        children = {
            mv: stats.to_dict()
            for mv, stats in sorted(self.moves.items(), key=lambda kv: (-kv[1].games, kv[0]))
        }
        best = self.best_move(min_games=1)
        return {
            "totalGames": self.total_games,
            "moves": children,
            "bookMove": best[0] if best else None,
            "bookWinRate": round(best[1], 4) if best else None,
            "bookGames": best[2] if best else 0,
        }

    def best_move(self, min_games: int = 3) -> tuple[str, float, int] | None:
        candidates: list[tuple[str, MoveStats]] = [
            (mv, st) for mv, st in self.moves.items() if st.games >= min_games
        ]
        if candidates:
            mv, st = max(candidates, key=lambda kv: (kv[1].win_rate, kv[1].games, kv[0]))
            return mv, st.win_rate, st.games
        if not self.moves:
            return None
        mv, st = max(self.moves.items(), key=lambda kv: (kv[1].games, kv[1].win_rate, kv[0]))
        return mv, st.win_rate, st.games


def _parse_json(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def _engines_from_source(source: str) -> tuple[str | None, str | None]:
    s = source
    if s.startswith("legacy-db:"):
        s = s[len("legacy-db:") :]
    for prefix in ("pool-", "random-"):
        if s.startswith(prefix):
            s = s[len(prefix) :]
            break
    m = re.match(r"(.+?)-vs-(.+?)(?:-\d+s)?(?:-[0-9a-f]{8})?(?:-s\d+)?$", s, re.I)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return None, None


def _players_from_metadata(meta: dict[str, Any]) -> tuple[str | None, str | None]:
    players = meta.get("players")
    if isinstance(players, list) and len(players) >= 2:
        return str(players[0]), str(players[1])
    white = meta.get("white") or meta.get("whiteEngine") or meta.get("engineWhite")
    black = meta.get("black") or meta.get("blackEngine") or meta.get("engineBlack")
    if white or black:
        return (str(white) if white else None, str(black) if black else None)
    return None, None


def involves_titanium(source: str, meta: dict[str, Any]) -> bool:
    if TITANIUM_RE.search(source):
        return True
    ea, eb = _engines_from_source(source)
    if _is_titanium_name(ea) or _is_titanium_name(eb):
        return True
    pa, pb = _players_from_metadata(meta)
    if _is_titanium_name(pa) or _is_titanium_name(pb):
        return True
    for key in ("engineA", "engineB", "engine_a", "engine_b", "stoppedBy", "mode"):
        val = meta.get(key)
        if _is_titanium_name(str(val) if val is not None else None):
            return True
    return False


def _is_titanium_name(name: str | None) -> bool:
    if not name:
        return False
    return bool(TITANIUM_RE.search(name))


def _outcome_for_mover(result: int | None, ply: int) -> int:
    """+1 win, -1 loss, 0 draw/unknown from mover's perspective."""
    if result is None:
        return 0
    if result == 0:
        return 0
    mover_is_p0 = ply % 2 == 0
    if result == 1:
        return 1 if mover_is_p0 else -1
    if result == -1:
        return -1 if mover_is_p0 else 1
    return 0


def _valid_moves(moves: list[str]) -> bool:
    return bool(moves) and all(MOVE_RE.fullmatch(m) for m in moves)


def load_games(db_path: Path) -> list[tuple[list[str], int | None, str]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT g.source, g.result, g.game_metadata, gp.packed_u8_move_sequence
        FROM games g
        JOIN game_paths gp ON gp.game_id = g.game_id
        """
    ).fetchall()
    conn.close()

    out: list[tuple[list[str], int | None, str]] = []
    skipped = defaultdict(int)
    for row in rows:
        source = row["source"] or ""
        meta = _parse_json(row["game_metadata"])
        if involves_titanium(source, meta):
            skipped["titanium"] += 1
            continue
        blob = row["packed_u8_move_sequence"]
        if blob is None:
            skipped["no_path"] += 1
            continue
        data = bytes(blob)
        if not data:
            skipped["empty_path"] += 1
            continue
        try:
            moves = moves_from_u8_blob(data)
        except (ValueError, KeyError, IndexError):
            skipped["decode_error"] += 1
            continue
        if not _valid_moves(moves):
            skipped["invalid_moves"] += 1
            continue
        if touches_denied_opening_prefix(moves):
            skipped["denied_opening"] += 1
            continue
        out.append((moves, row["result"], source))
    print(f"loaded {len(out)} games; skipped {dict(skipped)}")
    return out


def build_tree(
    games: list[tuple[list[str], int | None, str]],
    max_plies: int,
) -> dict[str, BookNode]:
    """Key = space-joined move prefix ('' for root)."""
    nodes: dict[str, BookNode] = defaultdict(BookNode)

    for moves, result, _source in games:
        limit = min(max_plies, len(moves))
        for ply in range(limit):
            prefix = moves[:ply]
            move = moves[ply]
            side = ply % 2
            if opening_book_move_denied(prefix, side, move):
                break
            key = " ".join(prefix)
            node = nodes[key]
            node.total_games += 1
            node.moves[move].record(_outcome_for_mover(result, ply))

    return nodes


def build_book_lines(nodes: dict[str, BookNode], max_plies: int, min_games: int) -> list[dict[str, Any]]:
    """Greedy main line: best win rate among moves with enough samples, else most popular."""
    lines: list[dict[str, Any]] = []
    prefix: list[str] = []

    for ply in range(max_plies):
        key = " ".join(prefix)
        node = nodes.get(key)
        if not node:
            break
        floor = min_games
        if ply == 0:
            floor = max(min_games, 20)
        pick = node.best_move(min_games=floor)
        if not pick:
            break
        mv, wr, n = pick
        prefix.append(mv)
        lines.append(
            {
                "ply": ply + 1,
                "prefix": list(prefix[:-1]),
                "move": mv,
                "winRate": round(wr, 4),
                "games": n,
                "side": "white" if ply % 2 == 0 else "black",
            }
        )
    return lines


@dataclass
class _EdgeAcc:
    visit: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0

    def record(self, outcome_for_mover: int) -> None:
        self.visit += 1
        if outcome_for_mover > 0:
            self.wins += 1
        elif outcome_for_mover < 0:
            self.losses += 1
        else:
            self.draws += 1

    @property
    def win_rate(self) -> float:
        decided = self.wins + self.losses
        if decided == 0:
            return 0.5
        return self.wins / decided


def build_opening_dag(
    games: list[tuple[list[str], int | None, str]],
    max_plies: int,
    out_db: Path,
) -> dict[str, int]:
    """Position DAG: 1 byte move_code_u8 per ply, aggregated win rate per edge."""
    positions: dict[bytes, PositionState] = {}
    paths: dict[bytes, bytes] = {}
    edges: dict[tuple[bytes, int], _EdgeAcc] = {}
    child_of: dict[tuple[bytes, int], bytes] = {}

    root = PositionState.initial()
    root_key = root.packed_state()
    positions[root_key] = root
    paths[root_key] = b""

    for moves, result, _source in games:
        state = PositionState.initial()
        parent_key = state.packed_state()
        path = bytearray()
        limit = min(max_plies, len(moves))
        for ply in range(limit):
            mv = moves[ply]
            code = encode_move(state, mv)
            acc = edges.setdefault((parent_key, code), _EdgeAcc())
            acc.record(_outcome_for_mover(result, ply))
            state = apply_move(state, mv)
            path.append(code)
            child_key = state.packed_state()
            positions[child_key] = state
            paths[child_key] = bytes(path)
            child_of[(parent_key, code)] = child_key
            parent_key = child_key

    out_db.parent.mkdir(parents=True, exist_ok=True)
    if out_db.exists():
        out_db.unlink()
    conn = sqlite3.connect(out_db)
    conn.executescript(DAG_SCHEMA)

    # Stable ids: sort by ply then packed bytes.
    ordered = sorted(
        positions.items(),
        key=lambda kv: (paths[kv[0]].__len__(), kv[0]),
    )
    id_of: dict[bytes, int] = {}
    for idx, (packed, state) in enumerate(ordered, start=1):
        id_of[packed] = idx
        conn.execute(
            "INSERT INTO positions(position_id, ply, side_to_move, packed_state) VALUES(?,?,?,?)",
            (idx, len(paths[packed]), int(state.side_to_move), packed),
        )
        conn.execute(
            "INSERT INTO path_codes(position_id, path_u8) VALUES(?,?)",
            (idx, paths[packed]),
        )

    edge_rows = 0
    for (parent_packed, code), acc in edges.items():
        parent_id = id_of[parent_packed]
        child_packed = child_of[(parent_packed, code)]
        child_id = id_of[child_packed]
        conn.execute(
            """
            INSERT INTO edges(
                parent_position_id, move_code_u8, child_position_id,
                visit_count, wins_stm, losses_stm, draws, win_rate
            ) VALUES(?,?,?,?,?,?,?,?)
            """,
            (
                parent_id,
                code,
                child_id,
                acc.visit,
                acc.wins,
                acc.losses,
                acc.draws,
                round(acc.win_rate, 6),
            ),
        )
        edge_rows += 1

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "max_plies": str(max_plies),
        "game_count": str(len(games)),
        "position_count": str(len(positions)),
        "edge_count": str(edge_rows),
        "filter": "exclude Titanium / v15 / v16",
        "move_encoding": "u8 per ply (game_store PositionState encode_move)",
    }
    for key, value in meta.items():
        conn.execute("INSERT INTO metadata(key, value) VALUES(?,?)", (key, value))
    conn.commit()
    conn.close()

    return {
        "positions": len(positions),
        "edges": edge_rows,
        "max_path_bytes": max((len(p) for p in paths.values()), default=0),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, default=GAME_STORE_DB)
    ap.add_argument("--max-plies", type=int, default=MAX_PLIES)
    ap.add_argument("--min-games", type=int, default=3, help="Min games for book move pick")
    ap.add_argument("--out-json", type=Path, default=DEFAULT_JSON)
    ap.add_argument("--out-lines", type=Path, default=DEFAULT_LINES)
    ap.add_argument("--out-dag", type=Path, default=DEFAULT_DAG)
    args = ap.parse_args()

    games = load_games(args.db)
    dag_stats = build_opening_dag(games, args.max_plies, args.out_dag)
    nodes = build_tree(games, args.max_plies)
    main_line = build_book_lines(nodes, args.max_plies, args.min_games)

    source_counts: dict[str, int] = defaultdict(int)
    for _moves, _result, source in games:
        bucket = source.split(":", 1)[-1]
        if bucket.startswith("pool-"):
            bucket = bucket[len("pool-") :]
        if "-vs-" in bucket:
            bucket = bucket.rsplit("-", 1)[0] if bucket[-1].isdigit() else bucket
        source_counts[bucket] += 1

    by_ply: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for key, node in nodes.items():
        ply = 0 if not key else len(key.split())
        if ply >= args.max_plies:
            continue
        by_ply[ply].append(
            {
                "prefix": key.split() if key else [],
                "prefixKey": key,
                **node.to_dict(),
            }
        )

    payload = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "sourceDb": str(args.db),
        "filter": "exclude games involving Titanium / v15 / v16",
        "gameCount": len(games),
        "dagDb": str(args.out_dag),
        "dagStats": dag_stats,
        "sourceBreakdown": dict(sorted(source_counts.items(), key=lambda kv: -kv[1])),
        "maxPlies": args.max_plies,
        "minGamesForBookMove": args.min_games,
        "mainLine": main_line,
        "nodesByPly": {str(ply): sorted(entries, key=lambda e: (-e["totalGames"], e["prefixKey"])) for ply, entries in sorted(by_ply.items())},
        "depthReached": max((len(e["prefix"]) for entries in by_ply.values() for e in entries), default=0),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    line_text = [
        f"# Non-Titanium opening book (max {args.max_plies} plies)",
        f"# Games: {len(games)}  generated: {payload['generatedAt']}",
        "",
        "## Main line (greedy by win rate)",
    ]
    seq: list[str] = []
    for step in main_line:
        seq.append(step["move"])
        line_text.append(
            f"ply {step['ply']:2d} ({step['side']:5s}): {step['move']:4s}  "
            f"wr={step['winRate']:.1%}  n={step['games']}  line={' '.join(seq)}"
        )
    line_text.append("")
    line_text.append("## All root moves (ply 1)")
    root = nodes.get("", BookNode())
    for mv, st in sorted(root.moves.items(), key=lambda kv: (-kv[1].win_rate, -kv[1].games)):
        line_text.append(f"  {mv:4s}  wr={st.win_rate:.1%}  n={st.games}  (W{st.wins}/L{st.losses}/D{st.draws})")

    args.out_lines.write_text("\n".join(line_text) + "\n", encoding="utf-8")

    print(f"wrote {args.out_dag}  ({dag_stats['positions']} positions, {dag_stats['edges']} edges)")
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_lines}")
    print(f"main line ({len(main_line)} plies): {' '.join(seq)}")
    print(f"unique prefixes with data: {len(nodes)}")


if __name__ == "__main__":
    main()
