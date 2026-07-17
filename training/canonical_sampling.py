"""Phase-aware epoch sampling with capped frequency emphasis."""
from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path

import numpy as np

from label_resolution import trainable_label_exists_sql
from label_weights import frequency_weight, game_phase_from_packed, game_phase_from_record

OPENING_BUCKET_FRAC = 0.28
MIDGAME_BUCKET_FRAC = 0.47
ENDGAME_BUCKET_FRAC = 0.25
OPENING_COMMON_FRAC = 0.65


def _phase_for_key(con: sqlite3.Connection, pos_key: str) -> str:
    if pos_key.startswith("teacher:"):
        try:
            packed_key = bytes.fromhex(pos_key[8:])
        except ValueError:
            return "midgame"
        row = con.execute(
            "SELECT packed_state FROM teacher_positions WHERE position_key=?",
            (packed_key,),
        ).fetchone()
        if not row:
            return "midgame"
        return game_phase_from_packed(bytes(row[0]))
    hex_key = pos_key[5:] if pos_key.startswith("json:") else pos_key
    row = con.execute(
        "SELECT position_data FROM positions WHERE pos_key=?",
        (hex_key,),
    ).fetchone()
    if not row:
        return "midgame"
    raw = row[0]
    try:
        rec = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return "midgame"
    return game_phase_from_record(rec)


def _is_trainable_key(con: sqlite3.Connection, pos_key: str) -> bool:
    if pos_key.startswith("teacher:"):
        return True
    hex_key = pos_key[5:] if pos_key.startswith("json:") else pos_key
    trainable = trainable_label_exists_sql(label_alias="l")
    row = con.execute(
        f"SELECT 1 FROM labels l WHERE l.pos_key=? AND {trainable} LIMIT 1",
        (hex_key,),
    ).fetchone()
    return row is not None


def _occurrence_for_key(con: sqlite3.Connection, pos_key: str) -> int:
    if pos_key.startswith("teacher:"):
        return 1
    hex_key = pos_key[5:] if pos_key.startswith("json:") else pos_key
    row = con.execute(
        "SELECT COALESCE(SUM(n_samples), 1) FROM labels WHERE pos_key=?",
        (hex_key,),
    ).fetchone()
    return max(1, int(row[0] or 1))


def apply_phase_sampling_quota(
    pos_keys: list[str],
    labels_db: Path,
    *,
    seed: int = 0,
    opening_frac: float = OPENING_BUCKET_FRAC,
    midgame_frac: float = MIDGAME_BUCKET_FRAC,
    endgame_frac: float = ENDGAME_BUCKET_FRAC,
) -> list[str]:
    """Rebalance an epoch toward explicit opening/mid/endgame quotas."""
    if not pos_keys:
        return []

    n = len(pos_keys)
    n_open = max(0, int(round(n * opening_frac)))
    n_mid = max(0, int(round(n * midgame_frac)))
    n_end = max(0, n - n_open - n_mid)

    con = sqlite3.connect(str(labels_db), timeout=60)
    try:
        buckets: dict[str, list[tuple[str, float]]] = {
            "opening": [],
            "midgame": [],
            "endgame": [],
        }
        for key in dict.fromkeys(pos_keys):
            if not _is_trainable_key(con, key):
                continue
            phase = _phase_for_key(con, key)
            occ = _occurrence_for_key(con, key)
            buckets[phase].append((key, frequency_weight(occ)))

        rng = np.random.default_rng(seed)

        def pick(bucket: str, count: int) -> list[str]:
            items = buckets[bucket]
            if not items or count <= 0:
                return []
            keys, weights = zip(*items)
            weights_arr = np.asarray(weights, dtype=np.float64)
            weights_arr /= weights_arr.sum()
            if bucket == "opening" and len(items) >= 2:
                sorted_items = sorted(items, key=lambda kv: kv[1], reverse=True)
                n_common = max(1, int(round(count * OPENING_COMMON_FRAC)))
                n_rare = count - n_common
                common_pool = [k for k, _ in sorted_items[: max(1, len(sorted_items) // 2)]]
                rare_pool = [k for k, _ in sorted_items[len(sorted_items) // 2 :]] or common_pool
                out: list[str] = []
                if common_pool:
                    out.extend(
                        rng.choice(
                            common_pool,
                            size=n_common,
                            replace=len(common_pool) < n_common,
                        ).tolist()
                    )
                if rare_pool and n_rare > 0:
                    out.extend(
                        rng.choice(
                            rare_pool,
                            size=n_rare,
                            replace=len(rare_pool) < n_rare,
                        ).tolist()
                    )
                rng.shuffle(out)
                return out[:count]
            replace = len(keys) < count
            chosen = rng.choice(list(keys), size=count, replace=replace, p=weights_arr)
            return list(chosen)

        merged = (
            pick("opening", n_open)
            + pick("midgame", n_mid)
            + pick("endgame", n_end)
        )
        if len(merged) < n:
            pool = [key for key in dict.fromkeys(pos_keys) if _is_trainable_key(con, key)]
            rng.shuffle(pool)
            merged.extend(pool[: n - len(merged)])
        rng.shuffle(merged)
        return merged[:n]
    finally:
        con.close()


def select_phase_balanced(
    pos_keys: list[str],
    labels_db: Path,
    *,
    count: int,
    seed: int = 0,
    opening_frac: float = OPENING_BUCKET_FRAC,
    midgame_frac: float = MIDGAME_BUCKET_FRAC,
    endgame_frac: float = ENDGAME_BUCKET_FRAC,
) -> list[str]:
    """Pick exactly ``count`` unique keys with opening/mid/end coverage.

    Preserves caller-controlled cohort sizes: useful when stratifying *inside*
    an already-sized fresh/recent/anchor pool without changing 80/10/10 ratios.
    Falls back to shuffled pool fill when a phase bucket is under-populated.
    """
    if count <= 0 or not pos_keys:
        return []

    unique = list(dict.fromkeys(pos_keys))
    n_open = max(0, int(round(count * opening_frac)))
    n_mid = max(0, int(round(count * midgame_frac)))
    n_end = max(0, count - n_open - n_mid)
    targets = {"opening": n_open, "midgame": n_mid, "endgame": n_end}

    con = sqlite3.connect(str(labels_db), timeout=60)
    try:
        buckets: dict[str, list[str]] = {"opening": [], "midgame": [], "endgame": []}
        for key in unique:
            if not _is_trainable_key(con, key):
                continue
            buckets[_phase_for_key(con, key)].append(key)
    finally:
        con.close()

    rng = np.random.default_rng(seed ^ 0xF11A5E)
    out: list[str] = []
    used: set[str] = set()
    for phase, need in targets.items():
        pool = [k for k in buckets[phase] if k not in used]
        rng.shuffle(pool)
        take = pool[:need]
        out.extend(take)
        used.update(take)

    if len(out) < count:
        leftovers = [k for k in unique if k not in used]
        rng.shuffle(leftovers)
        for key in leftovers:
            if len(out) >= count:
                break
            out.append(key)
            used.add(key)

    # If the unique pool itself is smaller than count, allow controlled repeats
    # only after uniqueness is exhausted (should be rare for production cohorts).
    while len(out) < count and unique:
        out.append(str(rng.choice(unique)))

    rng.shuffle(out)
    return [str(k) for k in out[:count]]
