"""Verify rebuilt policy chunk payloads in candidate dataset."""
from __future__ import annotations

import json
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from position_store_config import ROOT

from .schema import POLICY_CHUNK_MAGIC, POLICY_INDEX_MAGIC


@dataclass
class PolicyPayloadAudit:
    records_checked: int = 0
    unique_policy_records: int = 0
    has_policy_false: int = 0
    unresolved_policy: int = 0
    invalid_offset: int = 0
    invalid_length: int = 0
    decode_failure: int = 0
    checksum_mismatch: int = 0
    identity_mismatch: int = 0
    move_code_invalid: int = 0
    ambiguous_policy_matches: int = 0
    passed: bool = False
    samples: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "records_checked": self.records_checked,
            "unique_policy_records": self.unique_policy_records,
            "has_policy_false": self.has_policy_false,
            "unresolved_policy": self.unresolved_policy,
            "invalid_offset": self.invalid_offset,
            "invalid_length": self.invalid_length,
            "decode_failure": self.decode_failure,
            "checksum_mismatch": self.checksum_mismatch,
            "identity_mismatch": self.identity_mismatch,
            "move_code_invalid": self.move_code_invalid,
            "ambiguous_policy_matches": self.ambiguous_policy_matches,
            "passed": self.passed,
            "samples": self.samples[:20],
        }


def _validate_index_payloads(
    bin_blob: bytes,
    idx_data: bytes,
    audit: PolicyPayloadAudit,
    *,
    progress_every: int = 250_000,
) -> None:
    if not idx_data.startswith(POLICY_INDEX_MAGIC):
        raise ValueError("bad policy index magic")
    _version, count = struct.unpack_from("<HI", idx_data, 8)
    audit.unique_policy_records = int(count)
    header_size = len(POLICY_INDEX_MAGIC) + struct.calcsize("<HI")
    entry_size = struct.calcsize("<IQII32s")
    for record_id in range(count):
        audit.records_checked += 1
        if progress_every and record_id and record_id % progress_every == 0:
            print(f"[policy-payload-audit] {record_id}/{count}...", flush=True)
        entry_off = header_size + record_id * entry_size
        rid, payload_off, payload_len, crc, _content_hash = struct.unpack_from(
            "<IQII32s", idx_data, entry_off
        )
        if rid != record_id:
            audit.invalid_offset += 1
            if len(audit.samples) < 20:
                audit.samples.append({"policy_record_id": record_id, "reason": "rid_mismatch", "rid": rid})
            continue
        if payload_len <= 0 or payload_off + payload_len > len(bin_blob):
            audit.invalid_length += 1
            if len(audit.samples) < 20:
                audit.samples.append(
                    {"policy_record_id": record_id, "reason": "bad_bounds", "off": payload_off, "len": payload_len}
                )
            continue
        payload = bin_blob[payload_off : payload_off + payload_len]
        if (zlib.crc32(payload) & 0xFFFFFFFF) != crc:
            audit.checksum_mismatch += 1
            if len(audit.samples) < 20:
                audit.samples.append({"policy_record_id": record_id, "reason": "crc_mismatch"})
            continue
        try:
            n_moves, enc = struct.unpack_from("<BB", payload, 0)
        except struct.error as exc:
            audit.decode_failure += 1
            if len(audit.samples) < 20:
                audit.samples.append({"policy_record_id": record_id, "reason": "decode_failure", "error": str(exc)})
            continue
        if enc != 1:
            audit.decode_failure += 1
            continue
        pos = 2
        for _ in range(n_moves):
            if pos + 3 > len(payload):
                audit.decode_failure += 1
                break
            mv, _q = struct.unpack_from("<BH", payload, pos)
            if not 0 <= mv <= 135:
                audit.move_code_invalid += 1
                break
            pos += 3


def audit_built_policy_payloads(
    *,
    manifest_path: Path,
    root: Path = ROOT,
) -> PolicyPayloadAudit:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    counts = manifest.get("counts") or {}
    policy_parts = manifest.get("parts", {}).get("policies") or []
    if len(policy_parts) < 2:
        raise FileNotFoundError("manifest missing policy bin/idx parts")
    bin_path = root / policy_parts[0]
    idx_path = root / policy_parts[1]
    if not bin_path.is_file() or not idx_path.is_file():
        raise FileNotFoundError("policy chunk files missing")

    header = bin_path.read_bytes()[:8]
    if not header.startswith(POLICY_CHUNK_MAGIC):
        raise ValueError(f"unrecognized policy chunk magic: {header[:8]!r}")

    audit = PolicyPayloadAudit()
    labels_total = int(counts.get("labels", 0))
    has_policy_labels = int(counts.get("has_policy_labels", 0))
    audit.has_policy_false = labels_total - has_policy_labels

    print("[policy-payload-audit] loading policy bin (once)...", flush=True)
    bin_blob = bin_path.read_bytes()
    idx_data = idx_path.read_bytes()
    _validate_index_payloads(bin_blob, idx_data, audit)

    audit.passed = (
        audit.unresolved_policy == 0
        and audit.invalid_offset == 0
        and audit.invalid_length == 0
        and audit.decode_failure == 0
        and audit.checksum_mismatch == 0
        and audit.identity_mismatch == 0
        and audit.move_code_invalid == 0
        and audit.ambiguous_policy_matches == 0
    )
    return audit
