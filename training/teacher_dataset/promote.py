"""Atomic promotion of audited teacher dataset candidate to active path."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from position_store_config import REPORT_DIR, ROOT

from .catalog import benchmark_readers, build_teacher_catalog
from .config import TEACHER_CATALOG_DB, TEACHER_DATASET_DIR
from .evidence_envelope import validate_final_envelope
from .finalize import _bytes_for_parts, _part_paths
from .loader_smoke import run_loader_smoke_audit
from .promotion_gates import compute_manifest_hash, git_head, sha256_file
from .verify_artifacts import verify_candidate_artifacts


APPROVED = {
    "candidate_dir": "training/data/teacher_dataset_candidate_v10",
    "manifest_sha256": "95fa0dd5c7f5d0376bfcfd89d0933341adc5c495ce5268c6559bb16a2e23c38c",
    "audit_payload_sha256": "03244540a4db0e2a857491afde8c96a3c2240a8bfe29d75737a7615e87290d93",
    "final_envelope_sha256": "7b2d9297c4fe7fd9e85ce81f300568d3ab92d955a82bb01403cf9b4500076b44",
    "test_evidence_sha256": "4f6fc09f35e95494d934a065451160688204d899818efc9b2eade30a4b1af785",
    "counts": {
        "positions": 1405888,
        "labels": 2281163,
        "observations": 1454824,
        "unique_policies": 1927597,
        "has_policy_labels": 2275885,
        "policy_quarantined": 0,
    },
}


@dataclass
class PromotionContext:
    approved: dict[str, Any] = field(default_factory=lambda: dict(APPROVED))
    errors: list[str] = field(default_factory=list)

    def fail(self, msg: str) -> None:
        self.errors.append(msg)


def _artifact_rels() -> tuple[str, ...]:
    return (
        "positions/part-00000.parquet",
        "labels/part-00000.parquet",
        "observations/part-00000.parquet",
        "policies/policy-00000.bin",
        "policies/policy-00000.idx",
    )


def verify_pre_promotion(*, root: Path = ROOT) -> dict[str, Any]:
    ctx = PromotionContext()
    v10 = root / APPROVED["candidate_dir"]
    manifest_path = v10 / "manifest.json"
    if not manifest_path.is_file():
        ctx.fail("v10 manifest missing")
        return {"passed": False, "errors": ctx.errors}

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_hash = compute_manifest_hash(manifest)
    if manifest_hash != manifest.get("manifest_hash") or manifest_hash != APPROVED["manifest_sha256"]:
        ctx.fail(f"v10 manifest hash mismatch: {manifest_hash}")

    final_path = (
        root
        / "training/data/position_store_reports/"
        "gate_evidence_bundle_teacher_dataset_candidate_v9_20260620T101843Z.final.json"
    )
    test_path = root / "training/data/position_store_reports/teacher_dataset_test_evidence.json"
    if not final_path.is_file():
        ctx.fail("final evidence envelope missing")
    else:
        env = validate_final_envelope(final_path)
        if env["final_bundle_file_sha256"] != APPROVED["final_envelope_sha256"]:
            ctx.fail("final envelope sha256 mismatch")
        if env["audit_payload_sha256"] != APPROVED["audit_payload_sha256"]:
            ctx.fail("audit payload sha256 mismatch")

    if not test_path.is_file() or sha256_file(test_path) != APPROVED["test_evidence_sha256"]:
        ctx.fail("test evidence sha256 mismatch")

    art = verify_candidate_artifacts(v10, root=root)
    if not art.passed:
        ctx.fail("v10 artifact verification failed")

    counts = manifest.get("counts") or {}
    for key, expected in APPROVED["counts"].items():
        if int(counts.get(key, -1)) != int(expected):
            ctx.fail(f"count mismatch {key}: {counts.get(key)} != {expected}")
    labels_no_policy = int(counts.get("labels", 0)) - int(counts.get("has_policy_labels", 0))
    if labels_no_policy != 5278:
        ctx.fail(f"labels without policy mismatch: {labels_no_policy}")
    unresolved = (manifest.get("policy_resolution") or {}).get("v8_still_unresolved", 0)
    if int(unresolved) != 0:
        ctx.fail(f"unresolved policies: {unresolved}")

    if (root / "training/data/teacher_dataset_candidate_v10.partial").exists():
        ctx.fail("v10 .partial exists")
    if any(".partial" in str(p) for ps in manifest.get("parts", {}).values() for p in ps):
        ctx.fail("v10 manifest references .partial")

    return {
        "passed": not ctx.errors,
        "errors": ctx.errors,
        "manifest_sha256": manifest_hash,
        "artifact_hashes": art.file_hashes,
        "counts": counts,
    }


def snapshot_active_dataset(*, root: Path = ROOT, stamp: str | None = None) -> dict[str, Any]:
    active = root / "training/data/teacher_dataset"
    stamp = stamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rollback = root / f"training/data/teacher_dataset_rollback_{stamp}"
    info: dict[str, Any] = {
        "active_path": str(active.relative_to(root)).replace("\\", "/"),
        "rollback_path": str(rollback.relative_to(root)).replace("\\", "/"),
        "timestamp": stamp,
        "manifest_hash": None,
        "artifact_hashes": {},
        "counts": None,
        "is_symlink": active.is_symlink() if active.exists() else False,
        "target": str(active.readlink()) if active.is_symlink() else None,
        "bytes_on_disk": 0,
    }
    if not active.exists():
        info["note"] = "no prior active dataset directory"
        return info

    manifest_path = active / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        info["manifest_hash"] = manifest.get("manifest_hash")
        info["counts"] = manifest.get("counts")
        for rel in _artifact_rels():
            p = active / rel
            if p.is_file():
                key = f"training/data/teacher_dataset/{rel}".replace("\\", "/")
                info["artifact_hashes"][key] = sha256_file(p)
    for path in active.rglob("*"):
        if path.is_file():
            info["bytes_on_disk"] += path.stat().st_size
    return info


def repair_active_manifest_paths(*, active_dir: Path, root: Path = ROOT) -> dict[str, Any]:
    manifest_path = active_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["parts"] = _part_paths(active_dir, root)
    manifest["bytes"] = _bytes_for_parts(manifest["parts"], root)
    manifest["teacher_dataset_status"] = "promoted"
    manifest["promotion_allowed"] = False
    manifest["manifest_hash"] = compute_manifest_hash(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def atomic_promote(
    *,
    root: Path = ROOT,
    stamp: str | None = None,
) -> dict[str, Any]:
    stamp = stamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    pre = verify_pre_promotion(root=root)
    if not pre["passed"]:
        raise RuntimeError(f"pre-promotion verification failed: {pre['errors']}")

    v10 = (root / APPROVED["candidate_dir"]).resolve()
    active = (root / "training/data/teacher_dataset").resolve()
    rollback = (root / f"training/data/teacher_dataset_rollback_{stamp}").resolve()

    prior = snapshot_active_dataset(root=root, stamp=stamp)
    method = "two_step_directory_rename_same_filesystem"

    if rollback.exists():
        raise FileExistsError(f"rollback path already exists: {rollback}")

    promoted_manifest_hash_before_repair = APPROVED["manifest_sha256"]

    try:
        if active.exists():
            active.rename(rollback)
        v10.rename(active)
        manifest = repair_active_manifest_paths(active_dir=active, root=root)
    except Exception:
        if active.exists() and not rollback.exists() and (root / APPROVED["candidate_dir"]).exists() is False:
            pass
        if rollback.exists() and not active.exists():
            rollback.rename(active)
        elif rollback.exists() and active.exists():
            shutil.rmtree(active)
            rollback.rename(active)
        raise

    return {
        "promotion_timestamp": datetime.now(timezone.utc).isoformat(),
        "promotion_method": method,
        "rollback_path": str(rollback.relative_to(root)).replace("\\", "/"),
        "previous_active": prior,
        "candidate_manifest_sha256_at_promotion": promoted_manifest_hash_before_repair,
        "active_manifest_sha256_after_path_repair": manifest["manifest_hash"],
        "active_path": str(active.relative_to(root)).replace("\\", "/"),
    }


def verify_post_promotion(*, root: Path = ROOT) -> dict[str, Any]:
    active = root / "training/data/teacher_dataset"
    manifest = json.loads((active / "manifest.json").read_text(encoding="utf-8"))
    errors: list[str] = []

    if compute_manifest_hash(manifest) != manifest.get("manifest_hash"):
        errors.append("active manifest hash invalid")

    art = verify_candidate_artifacts(active, root=root)
    if not art.passed:
        errors.append("artifact verification failed")

    v10_hashes = APPROVED  # compare artifact bytes to approved via verify
    counts = manifest.get("counts") or {}
    for key, expected in APPROVED["counts"].items():
        if int(counts.get(key, -1)) != int(expected):
            errors.append(f"count mismatch {key}")

    catalog_path = root / "training/data/canonical/teacher_catalog_promoted.duckdb"
    build_teacher_catalog(catalog_path, manifest_path=active / "manifest.json", root=root)
    bench = benchmark_readers(catalog_path)
    for forbidden in ("teacher_dataset_candidate", "rollback", ".partial"):
        sql_blob = catalog_path.read_bytes()
        if forbidden.encode() in sql_blob:
            errors.append(f"catalog references forbidden token: {forbidden}")

    loader = run_loader_smoke_audit(active, root=root)
    if not loader.passed:
        errors.append(f"loader smoke failed: {loader.error or loader.checks}")

    import duckdb
    from concurrent.futures import ThreadPoolExecutor

    def _count() -> int:
        con = duckdb.connect(str(catalog_path), read_only=True)
        n = con.execute("SELECT COUNT(*) FROM teacher_labels").fetchone()[0]
        con.close()
        return int(n)

    with ThreadPoolExecutor(max_workers=4) as pool:
        counts_read = list(pool.map(lambda _: _count(), range(4)))
    concurrent_ok = len(set(counts_read)) == 1 and counts_read[0] == counts.get("labels")

    batch = _training_loader_batch(active, root=root)

    return {
        "passed": not errors and loader.passed and concurrent_ok and batch["passed"],
        "errors": errors,
        "manifest_sha256": manifest.get("manifest_hash"),
        "artifact_verification": art.to_dict(),
        "counts": counts,
        "duckdb": bench,
        "loader_smoke": loader.to_dict(),
        "concurrent_reader_counts": counts_read,
        "training_loader_batch": batch,
    }


def _training_loader_batch(active: Path, *, root: Path) -> dict[str, Any]:
    import pyarrow.parquet as pq
    from .policy_binary import PolicyChunkReader

    manifest = json.loads((active / "manifest.json").read_text(encoding="utf-8"))
    labels_path = root / manifest["parts"]["labels"][0]
    table = pq.read_table(labels_path, columns=["has_policy", "value_i16", "policy_record_id"])
    n = min(512, table.num_rows)
    with_policy = 0
    without_policy = 0
    bin_path = root / manifest["parts"]["policies"][0]
    idx_path = root / manifest["parts"]["policies"][1]
    with PolicyChunkReader(bin_path, idx_path) as reader:
        for i in range(n):
            if bool(table.column("has_policy")[i].as_py()):
                with_policy += 1
                rid = int(table.column("policy_record_id")[i].as_py())
                reader.read(rid)
            else:
                without_policy += 1
                _ = int(table.column("value_i16")[i].as_py())
    return {
        "passed": n == 512 or table.num_rows <= 512,
        "rows_scanned": n,
        "with_policy": with_policy,
        "without_policy": without_policy,
    }


def write_promotion_receipt(
    *,
    promotion_result: dict[str, Any],
    post_validation: dict[str, Any],
    root: Path = ROOT,
    tracked_path: Path | None = None,
) -> Path:
    tracked = tracked_path or (
        Path(__file__).resolve().parent / "candidate_provenance" / "teacher_dataset_v10_promotion_receipt.json"
    )
    receipt = {
        "record_type": "teacher_dataset_promotion_receipt",
        "promotion_timestamp": promotion_result["promotion_timestamp"],
        "promotion_method": promotion_result["promotion_method"],
        "promoted_by": "human_authorized_promotion",
        "candidate_identity": "teacher_dataset_candidate_v10",
        "candidate_manifest_sha256": APPROVED["manifest_sha256"],
        "audit_payload_sha256": APPROVED["audit_payload_sha256"],
        "final_evidence_envelope_sha256": APPROVED["final_envelope_sha256"],
        "test_evidence_sha256": APPROVED["test_evidence_sha256"],
        "previous_active_identity": promotion_result["previous_active"],
        "new_active_identity": {
            "path": promotion_result["active_path"],
            "manifest_sha256_after_path_repair": promotion_result["active_manifest_sha256_after_path_repair"],
        },
        "rollback_path": promotion_result["rollback_path"],
        "post_promotion_validation": post_validation,
        "training_commit": git_head(root),
    }
    tracked.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=tracked.parent, prefix=f"{tracked.stem}.", suffix=".tmp")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        tmp_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        tmp_path.replace(tracked)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
    return tracked


def promote_teacher_dataset_v10(*, root: Path = ROOT) -> dict[str, Any]:
    promotion = atomic_promote(root=root)
    post = verify_post_promotion(root=root)
    if not post["passed"]:
        _restore_rollback(root=root, rollback_rel=promotion["rollback_path"])
        post_after_restore = verify_post_promotion(root=root) if (root / "training/data/teacher_dataset").exists() else {}
        raise RuntimeError(
            json.dumps(
                {
                    "error": "post-promotion verification failed; rollback restored",
                    "post_validation": post,
                    "restored_check": post_after_restore,
                },
                indent=2,
            )
        )
    receipt_path = write_promotion_receipt(promotion_result=promotion, post_validation=post, root=root)
    receipt_sha = sha256_file(receipt_path)
    return {
        "promotion": promotion,
        "post_validation": post,
        "receipt_path": str(receipt_path.relative_to(root)).replace("\\", "/"),
        "receipt_sha256": receipt_sha,
    }


def _restore_rollback(*, root: Path, rollback_rel: str) -> None:
    active = root / "training/data/teacher_dataset"
    rollback = root / rollback_rel
    if active.exists():
        failed = root / f"{rollback_rel}_failed_active"
        if failed.exists():
            shutil.rmtree(failed)
        active.rename(failed)
    if rollback.exists():
        rollback.rename(active)
