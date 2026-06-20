"""Write immutable v10 provenance record linking audited v9 to finalized v10."""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from position_store_config import REPORT_DIR, ROOT

from .promotion_gates import compute_manifest_hash, sha256_file


def _artifact_hashes(candidate_dir: Path) -> dict[str, str]:
    rels = [
        "positions/part-00000.parquet",
        "labels/part-00000.parquet",
        "observations/part-00000.parquet",
        "policies/policy-00000.bin",
        "policies/policy-00000.idx",
    ]
    out: dict[str, str] = {}
    for rel in rels:
        p = candidate_dir / rel
        key = f"training/data/{candidate_dir.name}/{rel}".replace("\\", "/")
        out[key] = sha256_file(p)
    return out


def write_v10_provenance(
    *,
    source_candidate: str = "teacher_dataset_candidate_v9",
    target_candidate: str = "teacher_dataset_candidate_v10",
    audit_timestamp: str = "20260620T101843Z",
    gate_bundle_rel: str = (
        "training/data/position_store_reports/"
        "gate_evidence_bundle_teacher_dataset_candidate_v9_20260620T101843Z.json"
    ),
    tracked_out: Path | None = None,
) -> tuple[Path, Path]:
    """Return (reports_path, tracked_path)."""
    v9 = ROOT / "training" / "data" / source_candidate
    v10 = ROOT / "training" / "data" / target_candidate
    bundle_path = ROOT / gate_bundle_rel
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    bundle_sha_embedded = bundle.get("bundle_sha256")
    bundle_sha_on_disk = sha256_file(bundle_path)
    if bundle_sha_embedded and bundle_sha_embedded != bundle_sha_on_disk:
        bundle_sha_audit = bundle_sha_embedded
    else:
        bundle_sha_audit = bundle_sha_on_disk
    manifest_path = v10 / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    v9_hashes = _artifact_hashes(v9)
    v10_hashes = _artifact_hashes(v10)
    if sorted(v9_hashes.values()) != sorted(v10_hashes.values()):
        raise ValueError("v10 artifacts are not byte-identical to audited v9")

    now = datetime.now(timezone.utc).isoformat()
    payload = {
        "record_type": "teacher_dataset_candidate_provenance",
        "finalized_at": now,
        "source_candidate": source_candidate,
        "source_audit_timestamp": audit_timestamp,
        "source_evidence_bundle_path": gate_bundle_rel.replace("\\", "/"),
        "source_evidence_bundle_sha256": bundle_sha_audit,
        "source_evidence_bundle_sha256_on_disk": bundle_sha_on_disk,
        "target_candidate": target_candidate,
        "target_candidate_local_path": str(v10).replace("\\", "/"),
        "v10_manifest_path": f"training/data/{target_candidate}/manifest.json",
        "v10_manifest_sha256": manifest.get("manifest_hash"),
        "v10_manifest_sha256_recomputed": compute_manifest_hash(manifest),
        "promotion_allowed": False,
        "artifact_hashes_v9": v9_hashes,
        "artifact_hashes_v10": v10_hashes,
        "artifacts_byte_identical": True,
        "row_totals": manifest.get("counts"),
        "gate_evidence_bundle_reference": manifest.get("gate_evidence_bundle"),
        "finalization_notes": (
            "First finalize attempt briefly wrote promotion_allowed=true inside .partial via "
            "apply_promotion_allowed(); that directory was deleted before publication. "
            "Current v10 was atomically published from a fresh .partial with promotion_allowed=false "
            "and post-rename manifest path repair."
        ),
    }
    if payload["v10_manifest_sha256"] != payload["v10_manifest_sha256_recomputed"]:
        raise ValueError("v10 manifest hash mismatch")

    reports_dir = REPORT_DIR
    reports_dir.mkdir(parents=True, exist_ok=True)
    reports_path = reports_dir / f"v10_provenance_{audit_timestamp}.json"

    tracked = tracked_out or (
        Path(__file__).resolve().parent / "candidate_provenance" / "teacher_dataset_v10.json"
    )
    tracked.parent.mkdir(parents=True, exist_ok=True)

    for path in (reports_path, tracked):
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f"{path.stem}.", suffix=".tmp")
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp_path.replace(path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    return reports_path, tracked
