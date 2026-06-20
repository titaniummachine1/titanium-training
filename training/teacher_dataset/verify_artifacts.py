"""Verify every artifact referenced by a candidate manifest."""
from __future__ import annotations

import hashlib
import json
import struct
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from position_store_config import ROOT

from .policy_binary import PolicyChunkReader
from .promotion_gates import sha256_file
from .schema import POLICY_CHUNK_MAGIC, POLICY_INDEX_MAGIC


@dataclass
class ArtifactVerification:
    files_checked: int = 0
    bytes_checked: int = 0
    missing_files: int = 0
    partial_path_refs: int = 0
    size_mismatches: int = 0
    hash_mismatches: int = 0
    row_count_mismatches: int = 0
    invalid_policy_offsets: int = 0
    payload_crc_failures: int = 0
    invalid_move_codes: int = 0
    parquet_open_failures: int = 0
    passed: bool = False
    file_hashes: dict[str, str] = field(default_factory=dict)
    row_counts: dict[str, int] = field(default_factory=dict)
    samples: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "files_checked": self.files_checked,
            "bytes_checked": self.bytes_checked,
            "missing_files": self.missing_files,
            "partial_path_refs": self.partial_path_refs,
            "size_mismatches": self.size_mismatches,
            "hash_mismatches": self.hash_mismatches,
            "row_count_mismatches": self.row_count_mismatches,
            "invalid_policy_offsets": self.invalid_policy_offsets,
            "payload_crc_failures": self.payload_crc_failures,
            "invalid_move_codes": self.invalid_move_codes,
            "parquet_open_failures": self.parquet_open_failures,
            "passed": self.passed,
            "file_hashes": self.file_hashes,
            "row_counts": self.row_counts,
            "samples": self.samples[:20],
        }


def verify_candidate_artifacts(
    candidate_dir: Path,
    *,
    root: Path = ROOT,
    sample_policy_records: int = 500,
) -> ArtifactVerification:
    manifest_path = candidate_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    counts = manifest.get("counts") or {}
    bytes_expected = manifest.get("bytes") or {}
    report = ArtifactVerification()

    all_paths: list[str] = []
    for key, rels in (manifest.get("parts") or {}).items():
        for rel in rels:
            all_paths.append(rel)
            if ".partial" in rel.replace("\\", "/"):
                report.partial_path_refs += 1

    for rel in all_paths:
        path = root / rel
        report.files_checked += 1
        if not path.is_file():
            report.missing_files += 1
            continue
        size = path.stat().st_size
        report.bytes_checked += size
        report.file_hashes[rel] = sha256_file(path)

        # Size cross-check against manifest bytes block when keyed
        for bkey, bpath in [
            ("positions", manifest["parts"]["positions"][0]),
            ("labels", manifest["parts"]["labels"][0]),
            ("observations", manifest["parts"]["observations"][0]),
            ("policy_bin", manifest["parts"]["policies"][0]),
            ("policy_idx", manifest["parts"]["policies"][1]),
        ]:
            if rel == bpath and bkey in bytes_expected:
                if size != int(bytes_expected[bkey]):
                    report.size_mismatches += 1

    # Parquet row counts
    for part_key, count_key in [
        ("positions", "positions"),
        ("labels", "labels"),
        ("observations", "observations"),
    ]:
        rel = manifest["parts"][part_key][0]
        path = root / rel
        if not path.is_file():
            continue
        try:
            n = pq.read_metadata(path).num_rows
            report.row_counts[part_key] = int(n)
            if int(counts.get(count_key, -1)) != int(n):
                report.row_count_mismatches += 1
        except Exception as exc:
            report.parquet_open_failures += 1
            if len(report.samples) < 5:
                report.samples.append({"part": part_key, "error": str(exc)})

    # Policy binary header + sampled CRC/move-code audit
    bin_rel = manifest["parts"]["policies"][0]
    idx_rel = manifest["parts"]["policies"][1]
    bin_path = root / bin_rel
    idx_path = root / idx_rel
    if bin_path.is_file() and idx_path.is_file():
        header = bin_path.read_bytes()[:8]
        if not header.startswith(POLICY_CHUNK_MAGIC):
            report.invalid_policy_offsets += 1
        idx_data = idx_path.read_bytes()
        if not idx_data.startswith(POLICY_INDEX_MAGIC):
            report.invalid_policy_offsets += 1
        else:
            with PolicyChunkReader(bin_path, idx_path) as reader:
                n_records = reader.record_count
                step = max(1, n_records // sample_policy_records) if n_records else 1
                for rid in range(0, n_records, step):
                    try:
                        enc = reader.read(rid)
                    except (ValueError, struct.error, IndexError) as exc:
                        report.payload_crc_failures += 1
                        if len(report.samples) < 10:
                            report.samples.append({"policy_record_id": rid, "error": str(exc)})
                        continue
                    for code in enc.move_codes:
                        if not 0 <= code <= 135:
                            report.invalid_move_codes += 1
                            break

    report.passed = (
        report.missing_files == 0
        and report.partial_path_refs == 0
        and report.size_mismatches == 0
        and report.row_count_mismatches == 0
        and report.invalid_policy_offsets == 0
        and report.payload_crc_failures == 0
        and report.invalid_move_codes == 0
        and report.parquet_open_failures == 0
    )
    return report


def write_artifact_verification_report(
    report: ArtifactVerification,
    *,
    out_dir: Path,
    candidate_dir: Path,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"artifact_verification_{candidate_dir.name}_{stamp}.json"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "candidate_dir": str(candidate_dir),
        **report.to_dict(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
