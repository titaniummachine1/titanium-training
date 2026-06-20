"""Build immutable Parquet teacher dataset — candidate output only until parity gates pass."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from position_store_config import ROOT, TEACHER_STORE_DB

from .audit_policies import audit_teacher_policies
from .canonical_identity import spec_document
from .config import TEACHER_DATASET_CANDIDATE_DIR
from .jsonl_policy_index import build_jsonl_policy_index
from .policy_binary import EncodedPolicy, PolicyChunkWriter
from .policy_lookup import PolicyLookupStats, lookup_teacher_policy
from .schema import LABEL_TYPE_TO_TARGET_KIND, TEACHER_DATASET_SCHEMA_VERSION, TARGET_OTHER
from .sidecar_policy_index import build_sidecar_policy_index


def _position_key(canonical_hash: bytes, packed_state: bytes) -> bytes:
    return hashlib.blake2b(canonical_hash + packed_state, digest_size=16).digest()


def _dataset_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "positions": output_dir / "positions",
        "labels": output_dir / "labels",
        "observations": output_dir / "observations",
        "policies": output_dir / "policies",
        "rejects": output_dir / "rejects",
        "reports": output_dir / "reports",
    }


def build_teacher_dataset(
    output_dir: Path = TEACHER_DATASET_CANDIDATE_DIR,
    *,
    sqlite_db: Path = TEACHER_STORE_DB,
    root: Path = ROOT,
    compression: str = "zstd",
    batch_size: int = 100_000,
    _sidecar_index: dict | None = None,
    _jsonl_by_packed: dict | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    paths = _dataset_paths(output_dir)
    for d in paths.values():
        d.mkdir(parents=True, exist_ok=True)

    policy_audit = audit_teacher_policies(sqlite_db, root=root, verify_payloads=False)
    if _sidecar_index is None:
        sidecar_index, skipped_sidecars = build_sidecar_policy_index()
    else:
        sidecar_index, skipped_sidecars = _sidecar_index, 0
    if _jsonl_by_packed is None:
        _jsonl_canonical, jsonl_by_packed = build_jsonl_policy_index()
    else:
        jsonl_by_packed = _jsonl_by_packed
    lookup_stats = PolicyLookupStats()

    conn = sqlite3.connect(sqlite_db)
    conn.row_factory = sqlite3.Row

    pos_rows: list[dict[str, Any]] = []
    pos_id_to_key: dict[int, bytes] = {}
    pos_id_to_packed: dict[int, bytes] = {}
    for row in conn.execute(
        "SELECT position_id, canonical_hash, packed_state, side_to_move, total_visits, source_flags "
        "FROM positions ORDER BY canonical_hash, packed_state"
    ):
        canonical = bytes(row["canonical_hash"])
        packed = bytes(row["packed_state"])
        pid = int(row["position_id"])
        key = _position_key(canonical, packed)
        pos_id_to_key[pid] = key
        pos_id_to_packed[pid] = packed
        pos_rows.append(
            {
                "position_key": key,
                "canonical_hash": canonical,
                "packed_state": packed,
                "side_to_move": int(row["side_to_move"]),
                "source_flags": int(row["source_flags"] or 0),
                "total_observations": int(row["total_visits"] or 0),
            }
        )

    positions_path = paths["positions"] / "part-00000.parquet"
    pq.write_table(pa.Table.from_pylist(pos_rows), positions_path, compression=compression)

    policy_dedup: dict[bytes, int] = {}
    policy_writer = PolicyChunkWriter(chunk_id=0)
    label_rows: list[dict[str, Any]] = []
    obs_rows: list[dict[str, Any]] = []
    quarantine_rows: list[dict[str, Any]] = []

    for row in conn.execute(
        "SELECT l.label_id, l.position_id, l.label_type, l.value, l.best_move_u8, l.source, l.payload_json, "
        "p.canonical_hash AS pos_canonical, p.packed_state AS pos_packed "
        "FROM labels l JOIN positions p ON p.position_id = l.position_id ORDER BY l.label_id"
    ):
        if len(label_rows) and len(label_rows) % 100_000 == 0:
            print(f"build progress: labels={len(label_rows)} policies={len(policy_dedup)} unresolved={lookup_stats.unresolved}", flush=True)
        payload = json.loads(row["payload_json"] or "{}")
        pid = int(row["position_id"])
        pos_key = pos_id_to_key[pid]
        canonical = bytes(row["pos_canonical"])
        packed = bytes(row["pos_packed"])
        target_kind = LABEL_TYPE_TO_TARGET_KIND.get(str(row["label_type"]), TARGET_OTHER)
        obs_count = int(payload.get("observation_count") or 1)
        policy_record_id = None
        ref = payload.get("sidecar_ref")
        policy_hash = payload.get("policy_hash")

        if policy_hash and str(row["source"] or "").startswith("friend_selfplay:"):
            record = lookup_teacher_policy(
                canonical_hash=canonical,
                packed_state=packed,
                policy_hash=str(policy_hash),
                sidecar_ref=ref if isinstance(ref, dict) else None,
                source=str(row["source"] or ""),
                label_id=int(row["label_id"]),
                sidecar_index=sidecar_index,
                jsonl_by_packed=jsonl_by_packed,
                root=root,
                stats=lookup_stats,
            )
            if record is None:
                quarantine_rows.append(
                    {
                        "label_id": int(row["label_id"]),
                        "source": str(row["source"]),
                        "policy_hash": str(policy_hash),
                        "reason": "unresolved_policy",
                    }
                )
            else:
                encoded = EncodedPolicy.from_sparse(list(record.move_codes), list(record.policy_values))
                if encoded.content_hash not in policy_dedup:
                    policy_dedup[encoded.content_hash] = policy_writer.add(encoded)
                policy_record_id = policy_dedup[encoded.content_hash]

        value = row["value"]
        value_i16 = int(round(float(value) * 100)) if value is not None else None
        label_rows.append(
            {
                "position_key": pos_key,
                "label_set_id": hashlib.blake2b(
                    f"{row['label_type']}:{row['source']}:{value}:{payload.get('policy_hash')}".encode(),
                    digest_size=8,
                ).digest(),
                "target_kind": target_kind,
                "value_i16": value_i16,
                "best_move_u8": int(row["best_move_u8"]) if row["best_move_u8"] is not None else None,
                "policy_record_id": policy_record_id,
                "has_policy": policy_record_id is not None,
                "observation_count": obs_count,
                "source_cohort": str(row["source"] or ""),
            }
        )

    for row in conn.execute(
        "SELECT position_id, source_cohort, visit_count, p0_wins, p1_wins, draws "
        "FROM observations ORDER BY position_id, source_cohort"
    ):
        pos_key = pos_id_to_key.get(int(row["position_id"]))
        if pos_key is None:
            continue
        obs_rows.append(
            {
                "position_key": pos_key,
                "source_cohort": str(row["source_cohort"] or ""),
                "observation_count": int(row["visit_count"] or 0),
                "p0_win_count": int(row["p0_wins"] or 0),
                "draw_count": int(row["draws"] or 0),
                "p1_win_count": int(row["p1_wins"] or 0),
            }
        )

    conn.close()

    labels_path = paths["labels"] / "part-00000.parquet"
    obs_path = paths["observations"] / "part-00000.parquet"
    pq.write_table(pa.Table.from_pylist(label_rows), labels_path, compression=compression)
    pq.write_table(pa.Table.from_pylist(obs_rows), obs_path, compression=compression)

    if quarantine_rows:
        (paths["rejects"] / "policy_quarantine.jsonl").write_text(
            "\n".join(json.dumps(r) for r in quarantine_rows) + "\n",
            encoding="utf-8",
        )

    bin_bytes, idx_bytes = policy_writer.finalize()
    bin_partial = paths["policies"] / "policy-00000.bin.partial"
    idx_partial = paths["policies"] / "policy-00000.idx.partial"
    bin_ready = paths["policies"] / "policy-00000.bin"
    idx_ready = paths["policies"] / "policy-00000.idx"
    bin_partial.write_bytes(bin_bytes)
    idx_partial.write_bytes(idx_bytes)
    bin_partial.replace(bin_ready)
    idx_partial.replace(idx_ready)

    elapsed = time.perf_counter() - t0
    manifest_path = output_dir / "manifest.json"
    schema_path = output_dir / "schema.json"
    # All promotion gates default to False at build time.
    # Each gate must be set to True by a separate post-build verification step.
    # promotion_allowed becomes True only when ALL required gates are True AND
    # policy_quarantined == 0.  The build itself never sets promotion_allowed=True.
    promotion_gates = {
        # Teacher dataset position codec parity (packed_state + canonical_hash).
        # Verified by: python training/train.py audit-position-parity
        "cross_language_position_parity": False,
        "canonical_hash_parity": False,
        # Policy semantic hash round-trip: policy_semantic_hash(Rust) == policy_semantic_hash(Python).
        "policy_hash_algorithm_parity": False,
        "dataset_semantic_parity": False,
        "policy_payload_audit": False,
        "duckdb_catalog_audit": False,
        "concurrent_reader_test": False,
        "value_loader_smoke": False,
        "policy_loader_smoke": False,
        "required_tests": False,
        "engine_move_gen_parity": False,
    }

    manifest = {
        "schema_version": TEACHER_DATASET_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_sqlite": str(sqlite_db),
        "teacher_dataset_status": "candidate",
        # promotion_allowed is a derived field; recomputed from gates on every write.
        "promotion_allowed": False,
        "immutable": True,
        "compression": compression,
        "promotion_gates": promotion_gates,
        "canonical_identity_spec": spec_document(),
        "counts": {
            "positions": len(pos_rows),
            "labels": len(label_rows),
            "observations": len(obs_rows),
            "unique_policies": len(policy_dedup),
            "policy_quarantined": len(quarantine_rows),
        },
        "policy_resolution": {
            "from_sidecar_index": lookup_stats.from_sidecar_index,
            "from_jsonl_packed": lookup_stats.from_jsonl_packed,
            "from_sidecar_recovery": lookup_stats.from_sidecar_recovery,
            "no_policy": lookup_stats.no_policy,
            "unresolved": lookup_stats.unresolved,
            "skipped_corrupt_sidecars": skipped_sidecars,
        },
        "policy_status": policy_audit.status_counts,
        "parts": {
            "positions": [str(positions_path.relative_to(root)).replace("\\", "/")],
            "labels": [str(labels_path.relative_to(root)).replace("\\", "/")],
            "observations": [str(obs_path.relative_to(root)).replace("\\", "/")],
            "policies": [
                str(bin_ready.relative_to(root)).replace("\\", "/"),
                str(idx_ready.relative_to(root)).replace("\\", "/"),
            ],
        },
        "bytes": {
            "positions": positions_path.stat().st_size,
            "labels": labels_path.stat().st_size,
            "observations": obs_path.stat().st_size,
            "policy_bin": bin_ready.stat().st_size,
            "policy_idx": idx_ready.stat().st_size,
        },
        "build_seconds": elapsed,
        "manifest_hash": "",
    }
    manifest["manifest_hash"] = hashlib.sha256(
        json.dumps({k: v for k, v in manifest.items() if k != "manifest_hash"}, sort_keys=True).encode()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    schema_path.write_text(
        json.dumps(
            {
                "TEACHER_DATASET_SCHEMA_VERSION": TEACHER_DATASET_SCHEMA_VERSION,
                "columns": {
                    "positions": list(pos_rows[0].keys()) if pos_rows else [],
                    "labels": list(label_rows[0].keys()) if label_rows else [],
                    "observations": list(obs_rows[0].keys()) if obs_rows else [],
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest
