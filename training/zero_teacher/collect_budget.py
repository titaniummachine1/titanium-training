#!/usr/bin/env python3
"""Collect MCTS attention labels from quoridor-zero.ink (50–400 visit rollouts).

Trains search-budget / attention distillation — NOT per-node eval, NOT main WDL.

    python -m training.zero_teacher.collect_budget --from-db --limit 100 --visits 400
    python -m training.zero_teacher.collect_budget --visits 50 --bot-plies 40
"""

from __future__ import annotations

import argparse
import base64
import json
import hashlib
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

from tools.datagen.datagen import DB_PATH, load_games_from_db  # noqa: E402
from titanium_training.store.move_codec import pack_moves  # noqa: E402
from zero_teacher.client import (  # noqa: E402
    START_STATE,
    ZeroSettings,
    ZeroTeacherClient,
    ace_moves_to_zero_state,
    apply_zero_move,
    search_budget_features,
    paired_search_pressure,
    zero_move_text,
)
from zero_teacher.paths import DEFAULT_LABELS  # noqa: E402


def existing_keys(path: Path) -> set[str]:
    keys: set[str] = set()
    if not path.exists():
        return keys
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = row.get("moves_bin")
        if key:
            keys.add(str(key))
    return keys


def sample_db_prefixes(
    *,
    limit: int,
    min_ply: int,
    max_ply: int,
    seed: int,
    skip: set[str],
) -> list[tuple[list[str], int, str, str, str]]:
    rng = random.Random(seed)
    games = load_games_from_db(DB_PATH)
    candidates: list[tuple[list[str], int, str, str, str]] = []
    for moves, outcome, src in games:
        hi = min(max_ply, len(moves))
        for ply in range(min_ply, hi + 1):
            prefix = moves[:ply]
            key = base64.b64encode(pack_moves(prefix)).decode("ascii")
            if key not in skip:
                game_key = hashlib.sha256(pack_moves(moves)).hexdigest()[:20]
                candidates.append((prefix, outcome, src, key, game_key))
    rng.shuffle(candidates)
    return candidates[:limit]


def _row(
    *,
    key: str,
    moves: list[str],
    outcome: int | None,
    src: str,
    settings: ZeroSettings,
    feat: dict,
    chunks: list[dict],
    pressure: float,
    paired: dict,
    source_game_key: str,
    shallow_feat: dict,
    shallow_settings: ZeroSettings,
) -> dict:
    row = {
        "schema": "zero-search-budget-v2",
        "teacher": "quoridor-zero.ink",
        "moves_bin": key,
        "moves": moves,
        "src": src,
        "source_game_key": source_game_key,
        "ply": len(moves),
        "settings": {
            "shallow": shallow_settings.as_dict(),
            "deep": settings.as_dict(),
        },
        "search": {
            "shallow": shallow_feat,
            "deep": feat,
            "disagreement": {k: v for k, v in paired.items() if k != "search_pressure"},
        },
        "stream_last": chunks[-1] if chunks else None,
        "search_pressure": pressure,
    }
    if outcome in (-1, 1):
        row["outcome"] = outcome
    return row


def select_budget_chunks(chunks: list[dict], shallow_visits: int, deep_visits: int) -> tuple[dict, dict]:
    usable = [chunk for chunk in chunks if chunk.get("moves")]
    if not usable:
        raise RuntimeError("continuous search returned no move tables")

    def at_budget(target: int) -> dict:
        reached = [chunk for chunk in usable if int(chunk.get("totalVisits", 0)) >= target]
        return min(reached, key=lambda chunk: int(chunk.get("totalVisits", 0))) if reached else usable[-1]

    shallow = at_budget(shallow_visits)
    deep = at_budget(deep_visits)
    if int(deep.get("totalVisits", 0)) <= int(shallow.get("totalVisits", 0)):
        raise RuntimeError(
            f"continuous stream did not separate budgets: "
            f"{shallow.get('totalVisits')} vs {deep.get('totalVisits')}"
        )
    return shallow, deep


def consume_until_budget(stream, deep_visits: int, max_chunks: int | None) -> list[dict]:
    """Close the live stream as soon as a usable deep-budget snapshot arrives."""
    chunks: list[dict] = []
    try:
        for chunk in stream:
            chunks.append(chunk)
            if chunk.get("moves") and int(chunk.get("totalVisits", 0)) >= deep_visits:
                break
            if max_chunks is not None and len(chunks) >= max_chunks:
                break
    finally:
        close = getattr(stream, "close", None)
        if close is not None:
            close()
    return chunks


def collect_from_db(client: ZeroTeacherClient, args, out_path: Path) -> int:
    skip = existing_keys(out_path)
    prefixes = sample_db_prefixes(
        limit=args.limit,
        min_ply=args.min_ply,
        max_ply=args.max_ply,
        seed=args.seed,
        skip=skip,
    )
    settings = ZeroSettings(visits=args.deep_visits, threads=args.threads)
    shallow_settings = ZeroSettings(visits=args.shallow_visits, threads=args.threads)
    written = 0
    with out_path.open("a", encoding="utf-8") as f:
        for i, (moves, outcome, src, key, game_key) in enumerate(prefixes, 1):
            try:
                state = ace_moves_to_zero_state(moves)
                client.position(state)
                chunks = consume_until_budget(
                    client.continuous(state, settings),
                    args.deep_visits,
                    args.stream_chunks or None,
                )
                shallow, search = select_budget_chunks(
                    chunks, args.shallow_visits, args.deep_visits
                )
            except Exception as e:
                print(f"skip {i}/{len(prefixes)} ply={len(moves)}: {e}", file=sys.stderr)
                continue
            feat = search_budget_features(search, top_k=args.top_k)
            shallow_feat = search_budget_features(shallow, top_k=args.top_k)
            paired = paired_search_pressure(shallow, search)
            pressure = paired["search_pressure"]
            f.write(
                json.dumps(
                    _row(
                        key=key,
                        moves=moves,
                        outcome=outcome,
                        src=src,
                        settings=settings,
                        feat=feat,
                        chunks=chunks,
                        pressure=pressure,
                        paired=paired,
                        source_game_key=game_key,
                        shallow_feat=shallow_feat,
                        shallow_settings=shallow_settings,
                    ),
                    separators=(",", ":"),
                )
                + "\n"
            )
            f.flush()
            written += 1
            print(
                f"{written:4d} ply={len(moves):3d} pressure={pressure:+.3f} "
                f"topVF={feat['top_visit_fraction']:.1%} visits={feat['total_visits']}",
                flush=True,
            )
    return written


def collect_from_bot(client: ZeroTeacherClient, args, out_path: Path) -> int:
    settings = ZeroSettings(visits=args.deep_visits, threads=args.threads)
    shallow_settings = ZeroSettings(visits=args.shallow_visits, threads=args.threads)
    state = dict(START_STATE)
    moves: list[str] = []
    written = 0
    with out_path.open("a", encoding="utf-8") as f:
        for _ in range(1, args.bot_plies + 1):
            chunks = consume_until_budget(
                client.continuous(state, settings),
                args.deep_visits,
                args.stream_chunks or None,
            )
            shallow, search = select_budget_chunks(
                chunks, args.shallow_visits, args.deep_visits
            )
            feat = search_budget_features(search, top_k=args.top_k)
            shallow_feat = search_budget_features(shallow, top_k=args.top_k)
            paired = paired_search_pressure(shallow, search)
            pressure = paired["search_pressure"]
            key = base64.b64encode(pack_moves(moves)).decode("ascii")
            f.write(
                json.dumps(
                    _row(
                        key=key,
                        moves=list(moves),
                        outcome=None,
                        src="zero-bot",
                        settings=settings,
                        feat=feat,
                        chunks=chunks,
                        pressure=pressure,
                        paired=paired,
                        source_game_key=f"zero-bot-{args.seed}",
                        shallow_feat=shallow_feat,
                        shallow_settings=shallow_settings,
                    ),
                    separators=(",", ":"),
                )
                + "\n"
            )
            f.flush()
            written += 1
            print(
                f"{written:4d} ply={len(moves):3d} pressure={pressure:+.3f} "
                f"topVF={feat['top_visit_fraction']:.1%}",
                flush=True,
            )
            bot = client.bot_move(state, settings)
            state = apply_zero_move(state, bot["move"])
            moves.append(zero_move_text(bot["move"]))
            if state.get("winner") is not None:
                break
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(DEFAULT_LABELS))
    ap.add_argument("--base", default="https://quoridor-zero.ink")
    ap.add_argument("--model", default="resume-188/model_000159")
    ap.add_argument("--from-db", action="store_true")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--min-ply", type=int, default=4)
    ap.add_argument("--max-ply", type=int, default=80)
    ap.add_argument("--visits", type=int, default=None, help="deprecated alias for --deep-visits")
    ap.add_argument("--shallow-visits", type=int, default=50)
    ap.add_argument("--deep-visits", type=int, default=400)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--stream-chunks", type=int, default=32,
                    help="safety cap; collector selects ~50 and ~400 visit chunks from one stream")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--bot-plies", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    if args.visits is not None:
        args.deep_visits = args.visits

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    client = ZeroTeacherClient(base=args.base, model_id=args.model)
    n = collect_from_db(client, args, out_path) if args.from_db else collect_from_bot(
        client, args, out_path
    )
    print(f"wrote {n} rows -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
