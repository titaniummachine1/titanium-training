#!/usr/bin/env python3
"""Plan local/oracle shard spans from measured games-per-10min-per-worker."""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_CACHE = Path(__file__).with_name("match_throughput.json")

# Measured from jump_dist_ab local+oracle 60s/side (2026-07-17).
DEFAULT_LOCAL_GPT10 = 8.85
DEFAULT_ORACLE_GPT10 = 8.29


def count_shard_games(games: int, shard_count: int, offset: int, span: int) -> int:
    return sum(
        1
        for game_idx in range(games)
        if offset <= (game_idx % shard_count) < offset + span
    )


def estimate_finish_minutes(game_count: int, workers: int, gpt10: float) -> float:
    if game_count <= 0 or workers <= 0 or gpt10 <= 0:
        return 0.0
    return game_count / (gpt10 * workers / 10.0)


def plan_shards(
    games: int,
    *,
    local_workers: int,
    oracle_workers: int,
    local_gpt10: float,
    oracle_gpt10: float,
) -> dict[str, Any]:
    if games <= 0 or games % 2:
        raise ValueError("games must be a positive even number")
    if local_workers <= 0 or oracle_workers <= 0:
        raise ValueError("worker counts must be positive")
    if local_gpt10 <= 0 or oracle_gpt10 <= 0:
        raise ValueError("throughput must be positive")

    shard_count = local_workers + oracle_workers
    best: dict[str, Any] | None = None
    best_score = math.inf

    # Prefer spans near worker counts; allow ±2 slots to balance finish times.
    lo = max(1, local_workers - 2)
    hi = min(shard_count - 1, local_workers + 2)
    for local_span in range(lo, hi + 1):
        oracle_offset = local_span
        oracle_span = shard_count - local_span
        local_games = count_shard_games(games, shard_count, 0, local_span)
        oracle_games = count_shard_games(games, shard_count, oracle_offset, oracle_span)
        local_eta = estimate_finish_minutes(local_games, local_workers, local_gpt10)
        oracle_eta = estimate_finish_minutes(oracle_games, oracle_workers, oracle_gpt10)
        imbalance = abs(local_eta - oracle_eta)
        score = max(local_eta, oracle_eta) + 0.25 * imbalance
        if score < best_score:
            best_score = score
            best = {
                "games": games,
                "shard_count": shard_count,
                "local": {
                    "workers": local_workers,
                    "offset": 0,
                    "span": local_span,
                    "games": local_games,
                    "eta_minutes": round(local_eta, 2),
                    "games_per_10min_per_worker": local_gpt10,
                },
                "oracle": {
                    "workers": oracle_workers,
                    "offset": oracle_offset,
                    "span": oracle_span,
                    "games": oracle_games,
                    "eta_minutes": round(oracle_eta, 2),
                    "games_per_10min_per_worker": oracle_gpt10,
                },
                "finish_imbalance_minutes": round(imbalance, 2),
                "local_shard": f"0+{local_span}/{shard_count}",
                "oracle_shard": f"{oracle_offset}+{oracle_span}/{shard_count}",
                "local_results": f"results_shard_0_{local_span}.jsonl",
                "oracle_results": f"results_shard_{oracle_offset}_{oracle_span}.jsonl",
            }
    assert best is not None
    return best


def load_cache(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def save_cache(path: Path, doc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


def parse_iso(ts: str) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def ingest_status(status_path: Path, *, site: str, workers: int) -> dict[str, Any] | None:
    if not status_path.is_file():
        return None
    try:
        doc = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    completed = int(doc.get("completed_games", 0))
    if completed <= 0:
        return None
    started = parse_iso(str(doc.get("started_at", "")))
    finished = parse_iso(str(doc.get("finished_at", ""))) or parse_iso(
        str(doc.get("updated_at", ""))
    )
    if started is None or finished is None or finished <= started:
        return None
    elapsed_min = (finished - started).total_seconds() / 60.0
    if elapsed_min <= 0:
        return None
    gpt10 = completed / elapsed_min * 10.0 / max(1, workers)
    return {
        "site": site,
        "workers": workers,
        "completed_games": completed,
        "elapsed_minutes": round(elapsed_min, 3),
        "games_per_10min_per_worker": round(gpt10, 4),
        "source": str(status_path),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }


def resolve_rates(
    cache_path: Path,
    *,
    local_workers: int,
    oracle_workers: int,
    local_status: Path | None,
    oracle_status: Path | None,
    local_gpt10: float | None,
    oracle_gpt10: float | None,
) -> tuple[float, float, dict[str, Any]]:
    cache = load_cache(cache_path)
    meta: dict[str, Any] = {"cache_path": str(cache_path)}

    local_sample = None
    if local_status is not None:
        local_sample = ingest_status(local_status, site="local", workers=local_workers)
        if local_sample:
            cache["local"] = local_sample
            meta["local_sample"] = local_sample

    oracle_sample = None
    if oracle_status is not None:
        oracle_sample = ingest_status(oracle_status, site="oracle", workers=oracle_workers)
        if oracle_sample:
            cache["oracle"] = oracle_sample
            meta["oracle_sample"] = oracle_sample

    if local_gpt10 is None:
        local_gpt10 = float(
            (local_sample or {}).get("games_per_10min_per_worker")
            or (cache.get("local") or {}).get("games_per_10min_per_worker")
            or DEFAULT_LOCAL_GPT10
        )
    if oracle_gpt10 is None:
        oracle_gpt10 = float(
            (oracle_sample or {}).get("games_per_10min_per_worker")
            or (cache.get("oracle") or {}).get("games_per_10min_per_worker")
            or DEFAULT_ORACLE_GPT10
        )

    cache["defaults"] = {
        "local_gpt10": DEFAULT_LOCAL_GPT10,
        "oracle_gpt10": DEFAULT_ORACLE_GPT10,
    }
    cache["last_plan_inputs"] = {
        "local_gpt10": local_gpt10,
        "oracle_gpt10": oracle_gpt10,
        "local_workers": local_workers,
        "oracle_workers": oracle_workers,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    save_cache(cache_path, cache)
    meta["local_gpt10"] = local_gpt10
    meta["oracle_gpt10"] = oracle_gpt10
    return local_gpt10, oracle_gpt10, meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=None)
    ap.add_argument("--local-workers", type=int, default=4)
    ap.add_argument("--oracle-workers", type=int, default=13)
    ap.add_argument("--local-gpt10", type=float, default=None)
    ap.add_argument("--oracle-gpt10", type=float, default=None)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--local-status", type=Path, default=None)
    ap.add_argument("--oracle-status", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--ingest-only", action="store_true")
    args = ap.parse_args()

    local_gpt10, oracle_gpt10, rate_meta = resolve_rates(
        args.cache,
        local_workers=args.local_workers,
        oracle_workers=args.oracle_workers,
        local_status=args.local_status,
        oracle_status=args.oracle_status,
        local_gpt10=args.local_gpt10,
        oracle_gpt10=args.oracle_gpt10,
    )
    if args.ingest_only:
        print(json.dumps(rate_meta, indent=2))
        return 0
    if args.games is None:
        ap.error("--games is required unless --ingest-only")
    plan = plan_shards(
        args.games,
        local_workers=args.local_workers,
        oracle_workers=args.oracle_workers,
        local_gpt10=local_gpt10,
        oracle_gpt10=oracle_gpt10,
    )
    plan["rate_meta"] = rate_meta
    text = json.dumps(plan, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
