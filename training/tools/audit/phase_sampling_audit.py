#!/usr/bin/env python3
"""Read-only audit for streaming phase quota sampling."""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
TRAINING = ROOT / "training"
if str(TRAINING) not in sys.path:
    sys.path.insert(0, str(TRAINING))

from canonical_sampling import (  # noqa: E402
    ENDGAME_BUCKET_FRAC,
    MIDGAME_BUCKET_FRAC,
    OPENING_BUCKET_FRAC,
)
from db_import import LABELS_DB_PATH  # noqa: E402
from label_weights import game_phase_from_record  # noqa: E402
from streaming_db_loader import sample_epoch_keys  # noqa: E402
from streaming_val_split import split_streaming_epoch_keys  # noqa: E402


PHASES = ("opening", "midgame", "endgame")


def _json_load(raw: Any) -> dict[str, Any] | None:
    try:
        text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
        return json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
        return None


def _canonical_json_key(pos_key: str) -> str | None:
    if pos_key.startswith("teacher:"):
        return None
    return pos_key[5:] if pos_key.startswith("json:") else pos_key


def _phase_for_json_row(raw: Any) -> str:
    rec = _json_load(raw)
    if rec is None:
        return "midgame"
    return game_phase_from_record(rec)


def _phase_counts_for_keys(con: sqlite3.Connection, keys: list[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    cache: dict[str, str] = {}
    for key in keys:
        if key.startswith("teacher:"):
            counts["midgame"] += 1
            continue
        hex_key = _canonical_json_key(key)
        if not hex_key:
            counts["midgame"] += 1
            continue
        phase = cache.get(hex_key)
        if phase is None:
            row = con.execute(
                "SELECT position_data FROM positions WHERE pos_key=?",
                (hex_key,),
            ).fetchone()
            phase = _phase_for_json_row(row[0]) if row else "midgame"
            cache[hex_key] = phase
        counts[phase] += 1
    return counts


def _active_phase_map(
    con: sqlite3.Connection, *, max_rows: int | None = None
) -> tuple[dict[str, str], Counter[str], int, bool]:
    phase_by_key: dict[str, str] = {}
    counts: Counter[str] = Counter()
    scanned = 0
    query = """
        SELECT u.pos_key, p.position_data
        FROM position_usage u
        LEFT JOIN positions p ON p.pos_key =
            CASE WHEN substr(u.pos_key, 1, 5) = 'json:' THEN substr(u.pos_key, 6) ELSE u.pos_key END
        WHERE (u.retired = 0 OR COALESCE(u.protected_replay, 0) = 1)
        ORDER BY u.rowid ASC
    """
    for pos_key, raw in con.execute(query):
        scanned += 1
        key = str(pos_key)
        if key.startswith("teacher:"):
            phase = "midgame"
        elif raw is not None:
            phase = _phase_for_json_row(raw)
        else:
            phase = "midgame"
        phase_by_key[key] = phase
        counts[phase] += 1
        if max_rows is not None and scanned >= max_rows:
            break
    return phase_by_key, counts, scanned, max_rows is not None and scanned >= max_rows


def _phase_counts_from_map(keys: list[str], phase_by_key: dict[str, str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for key in keys:
        counts[phase_by_key.get(key, "midgame")] += 1
    return counts


def _quota_sample_from_phase_map(
    keys: list[str],
    phase_by_key: dict[str, str],
    *,
    seed: int,
) -> list[str]:
    import numpy as np

    n = len(keys)
    requested = _requested_counts(n)
    buckets: dict[str, list[str]] = defaultdict(list)
    for key in dict.fromkeys(keys):
        buckets[phase_by_key.get(key, "midgame")].append(key)

    rng = np.random.default_rng(seed)
    out: list[str] = []
    for phase in PHASES:
        count = int(requested.get(phase, 0))
        pool = buckets.get(phase, [])
        if not pool or count <= 0:
            continue
        chosen = rng.choice(pool, size=count, replace=len(pool) < count)
        out.extend(str(x) for x in chosen.tolist())
    if len(out) < n:
        pool = list(dict.fromkeys(keys))
        rng.shuffle(pool)
        out.extend(pool[: n - len(out)])
    rng.shuffle(out)
    return out[:n]


def _format_counts(counts: Counter[str]) -> dict[str, Any]:
    total = sum(counts.values())
    out: dict[str, Any] = {"total": total}
    for phase in PHASES:
        n = int(counts.get(phase, 0))
        out[phase] = {
            "count": n,
            "share_pct": round((100.0 * n / total) if total else 0.0, 3),
        }
    return out


def _requested_counts(n: int) -> Counter[str]:
    opening = max(0, int(round(n * OPENING_BUCKET_FRAC)))
    midgame = max(0, int(round(n * MIDGAME_BUCKET_FRAC)))
    endgame = max(0, n - opening - midgame)
    return Counter({"opening": opening, "midgame": midgame, "endgame": endgame})


def _shortfall(requested: Counter[str], available: Counter[str]) -> dict[str, int]:
    return {
        phase: max(0, int(requested.get(phase, 0)) - int(available.get(phase, 0)))
        for phase in PHASES
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels-db", type=Path, default=LABELS_DB_PATH)
    ap.add_argument("--epoch-size", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--retired-replay-fraction", type=float, default=0.05)
    ap.add_argument("--old-refresh-fraction", type=float, default=0.05)
    ap.add_argument("--full-active-epoch", action="store_true")
    ap.add_argument("--val-fraction", type=float, default=0.05)
    ap.add_argument("--eligible-max-rows", type=int, default=0)
    ap.add_argument(
        "--max-quota-keys",
        type=int,
        default=100000,
        help="Cap keys used for quota/split simulation after exact pre-quota counting; 0 means no cap.",
    )
    ap.add_argument("--out", type=Path, default=TRAINING / "runs" / "v16" / "phase_sampling_audit.json")
    args = ap.parse_args()

    labels_db = args.labels_db if args.labels_db.is_absolute() else (ROOT / args.labels_db).resolve()
    con = sqlite3.connect(str(labels_db), timeout=60)
    try:
        phase_by_key, eligible_counts, scanned, truncated = _active_phase_map(
            con,
            max_rows=args.eligible_max_rows or None,
        )
        if args.full_active_epoch and args.eligible_max_rows:
            pre_quota = list(phase_by_key.keys())
        else:
            pre_quota = sample_epoch_keys(
                con,
                epoch_size=args.epoch_size,
                seed=args.seed,
                retired_replay_fraction=args.retired_replay_fraction,
                old_refresh_fraction=args.old_refresh_fraction,
                full_active_epoch=args.full_active_epoch,
            )
        pre_counts = _phase_counts_from_map(pre_quota, phase_by_key)
        quota_input = pre_quota
        if args.max_quota_keys > 0 and len(pre_quota) > args.max_quota_keys:
            quota_input = pre_quota[: args.max_quota_keys]
        quota_keys = _quota_sample_from_phase_map(quota_input, phase_by_key, seed=args.seed)
        quota_counts = _phase_counts_from_map(quota_keys, phase_by_key)
        train_keys, val_keys = split_streaming_epoch_keys(
            quota_keys,
            labels_db=labels_db,
            val_fraction=args.val_fraction,
            seed=args.seed,
        )
        train_counts = _phase_counts_from_map(train_keys, phase_by_key)
        val_counts = _phase_counts_from_map(val_keys, phase_by_key)
    finally:
        con.close()

    requested = _requested_counts(len(pre_quota))
    report = {
        "title": "PHASE SAMPLING AUDIT",
        "labels_db": str(labels_db),
        "config": {
            "epoch_size": args.epoch_size,
            "seed": args.seed,
            "full_active_epoch": args.full_active_epoch,
            "retired_replay_fraction": args.retired_replay_fraction,
            "old_refresh_fraction": args.old_refresh_fraction,
            "val_fraction": args.val_fraction,
            "max_quota_keys": args.max_quota_keys,
            "quota_input_count": len(quota_input),
            "quota": {
                "opening": OPENING_BUCKET_FRAC,
                "midgame": MIDGAME_BUCKET_FRAC,
                "endgame": ENDGAME_BUCKET_FRAC,
            },
        },
        "eligible": {
            "counts": _format_counts(eligible_counts),
            "scanned": scanned,
            "truncated": truncated,
        },
        "requested": _format_counts(requested),
        "selected_before_fallback": _format_counts(pre_counts),
        "quota_simulation_input": _format_counts(_phase_counts_from_map(quota_input, phase_by_key)),
        "pre_quota_shortfall_if_no_replacement": _shortfall(requested, pre_counts),
        "fallback_selected": {
            "note": "canonical_sampling samples with replacement from non-empty phase buckets; this read-only audit uses the same phase targets but uniform within-phase selection for speed.",
            "counts": _format_counts(Counter()),
        },
        "actual_after_phase_quota": _format_counts(quota_counts),
        "actual_final_training_batches": _format_counts(train_counts),
        "validation_split": _format_counts(val_counts),
        "new_replay_breakdown": {
            "note": "sample_epoch_keys does not return source tags. In full_active_epoch mode, the pre-quota list is the active usage queue ordered by training visits before shuffling.",
        },
        "root_cause": "",
    }
    mid_req = requested.get("midgame", 0)
    mid_avail = pre_counts.get("midgame", 0)
    if mid_avail < mid_req:
        report["root_cause"] = (
            "The pre-quota candidate set does not contain enough midgame rows to satisfy "
            "the requested 47% midgame quota. The sampler uses replacement within non-empty "
            "phase buckets, so final quota output should still hit the target unless later "
            "train/validation splitting or featurization removes midgame rows."
        )
    elif train_counts.get("midgame", 0) < quota_counts.get("midgame", 0) * 0.5:
        report["root_cause"] = (
            "Phase quota output contains midgame rows, but the final training split loses a "
            "large share of them. Inspect streaming_val_split/build_val_position_keys."
        )
    else:
        report["root_cause"] = (
            "No quota bypass is visible in this dry audit. Compare this report with the "
            "epoch_weight_diagnostics JSON from the completed epoch for featurization skips "
            "or different runtime flags."
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
