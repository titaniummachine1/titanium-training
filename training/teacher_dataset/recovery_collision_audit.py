"""Recovery collision audit for (canonical_hash, policy_len) sidecar lookup key."""
from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from position_store_config import REPORT_DIR, TEACHER_SIDECARS, TEACHER_STORE_DB

from .policy_binary import EncodedPolicy
from .sidecar_reader import SidecarRecord, iter_sidecar_records


@dataclass
class RecoveryCollisionAudit:
    lookup_keys: int = 0
    unique_keys: int = 0
    colliding_keys: int = 0
    maximum_candidates_per_key: int = 0
    ambiguous_labels: int = 0
    resolved_labels: int = 0
    multi_candidate_tiebroken: int = 0
    passed: bool = False
    samples: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lookup_keys": self.lookup_keys,
            "unique_keys": self.unique_keys,
            "colliding_keys": self.colliding_keys,
            "maximum_candidates_per_key": self.maximum_candidates_per_key,
            "ambiguous_labels": self.ambiguous_labels,
            "resolved_labels": self.resolved_labels,
            "multi_candidate_tiebroken": self.multi_candidate_tiebroken,
            "passed": self.passed,
            "samples": self.samples[:20],
        }


def build_canon_len_index() -> dict[tuple[bytes, int], list[SidecarRecord]]:
    index: dict[tuple[bytes, int], list[SidecarRecord]] = defaultdict(list)
    friend_dir = TEACHER_SIDECARS / "friend_selfplay"
    for path in sorted(friend_dir.glob("iter_*.policy.bin.gz")):
        try:
            for _off, rec in iter_sidecar_records(path):
                key = (rec.canonical_hash, len(rec.move_codes))
                index[key].append(rec)
        except Exception:
            continue
    return index


def _pick_candidate(candidates: list[SidecarRecord], best_move_u8: int | None) -> SidecarRecord | None:
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    if best_move_u8 is not None:
        bm = int(best_move_u8)

        def score(rec: SidecarRecord) -> float:
            codes = list(rec.move_codes)
            vals = [v / 65535.0 for v in rec.policy_values_u16]
            if bm in codes:
                return vals[codes.index(bm)]
            return -1.0

        return sorted(candidates, key=score, reverse=True)[0]
    return candidates[0]


def audit_recovery_collisions(
    quarantine_path: Path,
    *,
    teacher_db: Path = TEACHER_STORE_DB,
) -> RecoveryCollisionAudit:
    quarantine: dict[int, dict] = {}
    for line in quarantine_path.read_text(encoding="utf-8").strip().splitlines():
        e = json.loads(line)
        quarantine[int(e["label_id"])] = e

    index = build_canon_len_index()
    audit = RecoveryCollisionAudit()
    audit.lookup_keys = len(index)
    audit.unique_keys = sum(1 for v in index.values() if len(v) == 1)
    audit.colliding_keys = sum(1 for v in index.values() if len(v) > 1)
    audit.maximum_candidates_per_key = max((len(v) for v in index.values()), default=0)

    conn = sqlite3.connect(f"file:{teacher_db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    ambiguous = 0
    resolved = 0
    tiebroken = 0

    for label_id in quarantine:
        row = conn.execute(
            "SELECT l.payload_json, l.best_move_u8, p.canonical_hash FROM labels l "
            "JOIN positions p ON p.position_id = l.position_id WHERE l.label_id = ?",
            (label_id,),
        ).fetchone()
        if row is None:
            ambiguous += 1
            continue
        payload = json.loads(row["payload_json"] or "{}")
        ref = payload.get("sidecar_ref") or {}
        policy_len = ref.get("policy_len")
        if policy_len is None:
            ambiguous += 1
            continue
        key = (bytes(row["canonical_hash"]), int(policy_len))
        cands = index.get(key, [])
        if len(cands) > 1:
            tiebroken += 1
        picked = _pick_candidate(cands, row["best_move_u8"])
        if picked is None:
            ambiguous += 1
            if len(audit.samples) < 10:
                audit.samples.append({"label_id": label_id, "reason": "no_candidate", "candidates": len(cands)})
        else:
            resolved += 1
    conn.close()

    audit.ambiguous_labels = ambiguous
    audit.resolved_labels = resolved
    audit.multi_candidate_tiebroken = tiebroken
    audit.passed = ambiguous == 0 and resolved == len(quarantine)
    return audit


def write_collision_audit_report(audit: RecoveryCollisionAudit, *, out_dir: Path = REPORT_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"recovery_collision_audit_{stamp}.json"
    path.write_text(
        json.dumps({"generated_at": datetime.now(timezone.utc).isoformat(), **audit.to_dict()}, indent=2),
        encoding="utf-8",
    )
    return path
