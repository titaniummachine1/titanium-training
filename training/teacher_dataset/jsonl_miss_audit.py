"""JSONL miss classification for v7 quarantine labels — durable audit report."""
from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from position_store_config import REPORT_DIR, TEACHER_STORE_DB
from position_store_friend import discover_friend_shards
from position_store_lib import _alpha_action_to_move_u8, policy_semantic_hash

from .friend_state import parse_friend_state
from .promotion_gates import sha256_file


def _load_quarantine(path: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        e = json.loads(line)
        out[int(e["label_id"])] = e
    return out


def _build_jsonl_detail_index() -> tuple[
    dict[tuple[bytes, str], dict],
    dict[bytes, list[dict]],
    set[bytes],
]:
    by_packed_ph: dict[tuple[bytes, str], dict] = {}
    by_packed: dict[bytes, list[dict]] = defaultdict(list)
    dense_positions: set[bytes] = set()
    for shard_path in discover_friend_shards():
        shard_name = shard_path.parent.name
        with shard_path.open(encoding="utf-8") as fh:
            for line_no, raw in enumerate(fh, 1):
                raw = raw.strip()
                if not raw:
                    continue
                obj = json.loads(raw)
                uses_dense = "policy" in obj and "policyActions" not in obj and "policyValues" not in obj
                try:
                    state = parse_friend_state(obj)
                    packed = state.packed_state()
                except (KeyError, TypeError, ValueError):
                    continue
                if uses_dense:
                    dense_positions.add(packed)
                    continue
                actions = obj.get("policyActions") or obj.get("policy_actions") or []
                values = obj.get("policyValues") or obj.get("policy_values") or []
                if not actions or not values:
                    continue
                move_codes = [_alpha_action_to_move_u8(state, int(a)) for a in actions]
                ph = policy_semantic_hash(move_codes, [float(v) for v in values])
                detail = {
                    "shard": shard_name,
                    "line_no": line_no,
                    "policy_hash": ph,
                    "policy_len": len(move_codes),
                    "move_codes": move_codes,
                    "values": [float(v) for v in values],
                }
                by_packed_ph[(packed, ph)] = detail
                by_packed[packed].append(detail)
    return by_packed_ph, by_packed, dense_positions


def classify_jsonl_misses(
    quarantine_path: Path,
    *,
    teacher_db: Path = TEACHER_STORE_DB,
) -> dict[str, Any]:
    quarantine = _load_quarantine(quarantine_path)
    by_packed_ph, by_packed, dense_positions = _build_jsonl_detail_index()
    shard_names = {p.parent.name for p in discover_friend_shards()}

    conn = sqlite3.connect(f"file:{teacher_db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    top_counts: Counter = Counter()
    sub_counts: Counter = Counter()
    samples: dict[str, list] = defaultdict(list)
    unknown = 0

    label_ids = list(quarantine.keys())
    sqlite_info: dict[int, dict] = {}
    batch = 900
    for i in range(0, len(label_ids), batch):
        chunk = label_ids[i : i + batch]
        ph = ",".join("?" * len(chunk))
        for row in conn.execute(
            f"SELECT l.label_id, l.source, l.best_move_u8, l.payload_json, "
            f"p.canonical_hash, p.packed_state "
            f"FROM labels l JOIN positions p ON p.position_id = l.position_id "
            f"WHERE l.label_id IN ({ph})",
            chunk,
        ):
            payload = json.loads(row["payload_json"] or "{}")
            sqlite_info[int(row["label_id"])] = {
                "source": str(row["source"] or ""),
                "import_policy_hash": payload.get("policy_hash"),
                "policy_len": (payload.get("sidecar_ref") or {}).get("policy_len"),
                "best_move_u8": row["best_move_u8"],
                "canonical_hash": bytes(row["canonical_hash"]),
                "packed_state": bytes(row["packed_state"]),
            }
    conn.close()

    for label_id, q in quarantine.items():
        info = sqlite_info.get(label_id)
        sub = "unknown"
        if info is None:
            top_counts["STALE_INDEX"] += 1
            unknown += 1
            continue
        packed = info["packed_state"]
        import_ph = info["import_policy_hash"]
        source = info["source"]
        iter_name = source.split(":")[1] if ":" in source else None

        if packed not in by_packed and iter_name not in shard_names:
            top_counts["SOURCE_LINEAGE_MISSING"] += 1
            sub = "source_shard_absent"
        elif packed not in by_packed:
            top_counts["PACKED_STATE_MISMATCH"] += 1
            sub = "packed_not_in_jsonl"
        elif packed in dense_positions:
            top_counts["DENSE_POLICY_FORMAT"] += 1
            sub = "dense_policy_array"
        elif import_ph and (packed, import_ph) in by_packed_ph:
            top_counts["STALE_INDEX"] += 1
            sub = "should_have_matched"
            unknown += 1
        else:
            top_counts["POLICY_IDENTITY_MISMATCH"] += 1
            jsonl_entries = by_packed.get(packed, [])
            import_len = info.get("policy_len")
            jsonl_lens = {e["policy_len"] for e in jsonl_entries}
            if import_len is not None and jsonl_lens and int(import_len) not in jsonl_lens:
                sub = "policy_length_changed"
            elif jsonl_entries:
                # Compare move sets between import sidecar expectation and any JSONL row
                jsonl_moves = {tuple(e["move_codes"]) for e in jsonl_entries}
                if len(jsonl_moves) > 1:
                    sub = "duplicate_position_multiple_policies"
                else:
                    jm = next(iter(jsonl_moves))
                    # Without sidecar decode here, infer from hash mismatch + same length
                    if import_len and int(import_len) == len(jm):
                        sub = "policy_values_or_move_set_changed"
                    else:
                        sub = "move_set_changed"
            else:
                sub = "policy_absent_in_jsonl_sparse"
            if info["best_move_u8"] is not None and jsonl_entries:
                sub_counts[f"best_move_present:{sub}"] += 1

        sub_counts[sub] += 1
        if len(samples[sub]) < 5:
            samples[sub].append(
                {
                    "label_id": label_id,
                    "source": source,
                    "import_policy_hash": import_ph,
                    "import_policy_len": info.get("policy_len"),
                    "jsonl_policy_hashes_at_packed": [e["policy_hash"][:16] for e in by_packed.get(packed, [])[:3]],
                }
            )

    return {
        "total_quarantined": len(quarantine),
        "top_level_counts": dict(top_counts),
        "sub_classification_counts": dict(sub_counts),
        "unknown": unknown,
        "samples": {k: v for k, v in samples.items()},
        "inference_note": (
            "Import-time source JSONL checksums were not captured at Rust import. "
            "Sub-classifications infer data evolution from current JSONL vs import-time "
            "policy_hash/policy_len stored in SQLite; not a proof of which JSONL revision was imported."
        ),
        "passed": unknown == 0,
    }


def write_jsonl_miss_report(
    report: dict[str, Any],
    *,
    out_dir: Path = REPORT_DIR,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"jsonl_miss_classification_{stamp}.json"
    payload = {"generated_at": datetime.now(timezone.utc).isoformat(), **report}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
