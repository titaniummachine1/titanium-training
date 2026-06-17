#!/usr/bin/env python3
"""Import a Quoridor game from clipboard HTML (site move-list panel).

Copy the move list from the website UI, then run:

    python training/import_clipboard_game.py
    python training/import_clipboard_game.py --winner W
    python training/import_clipboard_game.py --dry-run

Reads Windows clipboard via PowerShell. Parses amber/indigo move pairs,
normalizes wall notation (hd6 -> d6h), inserts into all_games.db, and
appends to training/data/clipboard_imports.games.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))

from datagen import DB_PATH, insert_single_game, validate_game  # noqa: E402
from move_codec import algebraic_to_ace, ace_to_algebraic  # noqa: E402

GAMES_FILE = ROOT / "training" / "data" / "clipboard_imports.games"
DEFAULT_TAG = "clipboard-import"


def read_clipboard() -> str:
    if sys.platform == "win32":
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "Get-Clipboard -Raw"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if r.returncode != 0:
            raise RuntimeError(r.stderr.strip() or "Get-Clipboard failed")
        return r.stdout
    for cmd in (["pbpaste"], ["xclip", "-o", "-selection", "clipboard"]):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if r.returncode == 0:
                return r.stdout
        except FileNotFoundError:
            continue
    raise RuntimeError("clipboard read unsupported on this platform")


def normalize_move(raw: str) -> str:
    s = raw.strip().lower()
    if not s or s.endswith("."):
        return ""
    if len(s) >= 3 and s[0] in "hv" and s[1].isalpha() and s[2].isdigit():
        return s[1:] + s[0]
    return s


def parse_html_moves(html: str) -> list[str]:
    moves: list[str] = []
    row_re = re.compile(
        r'flex items-center gap-1[^>]*>([\s\S]*?)</div>',
        re.IGNORECASE,
    )
    amber_re = re.compile(r"text-amber[^>]*>([^<]+)", re.IGNORECASE)
    indigo_re = re.compile(r"text-indigo[^>]*>([^<]+)", re.IGNORECASE)

    for row in row_re.finditer(html):
        inner = row.group(1)
        if "text-amber" not in inner:
            continue
        am = amber_re.search(inner)
        ind = indigo_re.search(inner)
        w = normalize_move(am.group(1)) if am else ""
        b = normalize_move(ind.group(1)) if ind else ""
        if w:
            moves.append(w)
        if b:
            moves.append(b)

    if not moves:
        # Plain text fallback: tokens on lines or space-separated
        tokens = re.findall(r"\b([a-i][1-9](?:[hv])?|[hv][a-i][1-9])\b", html, re.I)
        moves = [normalize_move(t) for t in tokens if normalize_move(t)]

    return moves


def validate_moves_replay(moves: list[str]) -> None:
    """Round-trip ACE encode/decode — catches bad wall/pawn tokens."""
    for i, m in enumerate(moves):
        try:
            ace = algebraic_to_ace(m)
            back = ace_to_algebraic(ace)
        except Exception as e:
            raise ValueError(f"illegal move #{i + 1} {m!r}: {e}") from e
        if back != m:
            raise ValueError(f"move #{i + 1} {m!r} normalizes to {back!r}")


def infer_outcome(moves: list[str]) -> int:
    if not moves:
        raise ValueError("no moves")
    last = moves[-1]
    if len(last) != 2 or not last[0].isalpha() or not last[1].isdigit():
        raise ValueError(
            f"last move {last!r} is not a pawn reach — pass --winner W or --winner B"
        )
    rank = last[1]
    white_to_move = len(moves) % 2 == 1
    if white_to_move:
        if rank == "9":
            return 1
        if rank == "1":
            return -1
    else:
        if rank == "1":
            return -1
        if rank == "9":
            return 1
    raise ValueError(f"cannot infer winner from last move {last!r} — pass --winner W|B")


def append_games_file(moves: list[str], outcome: int) -> None:
    GAMES_FILE.parent.mkdir(parents=True, exist_ok=True)
    winner = "W" if outcome == 1 else "B"
    with open(GAMES_FILE, "a", encoding="utf-8") as f:
        f.write(f"GAME {' '.join(moves)}\n")
        f.write(f"RESULT {winner}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--winner", choices=("W", "B"), help="Force winner if auto-detect fails")
    ap.add_argument("--tag", default=DEFAULT_TAG, help=f"DB source tag (default {DEFAULT_TAG})")
    ap.add_argument("--dry-run", action="store_true", help="Parse only, do not write DB")
    ap.add_argument("--text", help="Raw HTML/text instead of clipboard")
    ap.add_argument("--file", type=Path, help="Read HTML/text from file instead of clipboard")
    args = ap.parse_args()

    if args.file is not None:
        raw = args.file.read_text(encoding="utf-8", errors="replace")
    elif args.text is not None:
        raw = args.text
    else:
        raw = read_clipboard()
    if not raw.strip():
        print("ERROR: clipboard empty", file=sys.stderr)
        sys.exit(1)

    moves = parse_html_moves(raw)
    if not moves:
        print("ERROR: no moves found in clipboard HTML", file=sys.stderr)
        sys.exit(1)

    validate_moves_replay(moves)

    if args.winner:
        outcome = 1 if args.winner == "W" else -1
    else:
        outcome = infer_outcome(moves)

    err = validate_game(moves, outcome)
    if err:
        print(f"ERROR: {err}", file=sys.stderr)
        sys.exit(1)

    winner = "White (W)" if outcome == 1 else "Black (B)"
    print(f"Parsed {len(moves)} plies — {winner} wins")
    print(f"  first: {' '.join(moves[:6])} ...")
    print(f"  last:  {' '.join(moves[-4:])}")

    if args.dry_run:
        print("(dry-run — not written)")
        return

    gid = insert_single_game(moves, outcome, DB_PATH, args.tag)
    append_games_file(moves, outcome)
    print(f"Inserted game id={gid} tag={args.tag!r}")
    print(f"  DB:   {DB_PATH}")
    print(f"  file: {GAMES_FILE}")


if __name__ == "__main__":
    main()
