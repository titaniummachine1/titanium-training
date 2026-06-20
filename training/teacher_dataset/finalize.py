"""Reproducible candidate finalization: .partial -> finalized directory with fresh manifest."""
from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from position_store_config import ROOT, TEACHER_STORE_DB

from .canonical_identity import spec_document
from .promotion_gates import apply_promotion_allowed, compute_manifest_hash, gate_evidence, git_head
from .schema import TEACHER_DATASET_SCHEMA_VERSION


def _dataset_subdirs() -> tuple[str, ...]:
    return ("positions", "labels", "observations", "policies", "rejects", "reports")


def _part_paths(output_dir: Path, root: Path) -> dict[str, list[str]]:
    output_dir = output_dir.resolve()
    root = root.resolve()
    positions = output_dir / "positions" / "part-00000.parquet"
    labels = output_dir / "labels" / "part-00000.parquet"
    obs = output_dir / "observations" / "part-00000.parquet"
    bin_path = output_dir / "policies" / "policy-00000.bin"
    idx_path = output_dir / "policies" / "policy-00000.idx"
    rel = lambda p: str(p.relative_to(root)).replace("\\", "/")
    return {
        "positions": [rel(positions)],
        "labels": [rel(labels)],
        "observations": [rel(obs)],
        "policies": [rel(bin_path), rel(idx_path)],
    }


def _bytes_for_parts(parts: dict[str, list[str]], root: Path) -> dict[str, int]:
    pos = root / parts["positions"][0]
    labels = root / parts["labels"][0]
    obs = root / parts["observations"][0]
    bin_p = root / parts["policies"][0]
    idx_p = root / parts["policies"][1]
    return {
        "positions": pos.stat().st_size,
        "labels": labels.stat().st_size,
        "observations": obs.stat().st_size,
        "policy_bin": bin_p.stat().st_size,
        "policy_idx": idx_p.stat().st_size,
    }


def finalize_teacher_candidate(
    *,
    source_dir: Path,
    target_dir: Path,
    root: Path = ROOT,
    parent_candidate: str | None = None,
    recovery_method: str | None = None,
    extra_manifest: dict[str, Any] | None = None,
    promotion_gates: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Copy immutable files from source into target.partial, write manifest, atomic rename."""
    t0 = time.perf_counter()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"source candidate missing: {source_dir}")
    if target_dir.exists():
        raise FileExistsError(f"target already exists: {target_dir}")

    src_manifest_path = source_dir / "manifest.json"
    if not src_manifest_path.is_file():
        raise FileNotFoundError(f"source manifest missing: {src_manifest_path}")
    src_manifest = json.loads(src_manifest_path.read_text(encoding="utf-8"))

    partial = Path(str(target_dir) + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    for sub in _dataset_subdirs():
        (partial / sub).mkdir(parents=True, exist_ok=True)

    # Copy parquet/binary/schema; never mutate source.
    for sub in ("positions", "labels", "observations", "policies"):
        src_sub = source_dir / sub
        if src_sub.is_dir():
            for f in src_sub.iterdir():
                if f.is_file() and not f.name.endswith(".partial"):
                    shutil.copy2(f, partial / sub / f.name)
    for sub in ("rejects", "reports"):
        src_sub = source_dir / sub
        if src_sub.is_dir():
            for f in src_sub.iterdir():
                if f.is_file():
                    shutil.copy2(f, partial / sub / f.name)
    if (source_dir / "schema.json").is_file():
        shutil.copy2(source_dir / "schema.json", partial / "schema.json")

    parts = _part_paths(partial, root)
    commit = git_head(root)
    if promotion_gates is not None:
        gates = promotion_gates
    else:
        gates = {
            "cross_language_position_parity": gate_evidence(
                passed=False,
                commit=commit,
                notes="Set by audit-position-parity report",
            ),
            "canonical_hash_parity": gate_evidence(
                passed=False,
                commit=commit,
                notes="Set together with position parity audit",
            ),
            "policy_hash_algorithm_parity": gate_evidence(
                passed=False,
                commit=commit,
                notes="Verify policy_semantic_hash round-trip Rust/Python",
            ),
            "dataset_semantic_parity": gate_evidence(passed=False, commit=commit),
            "policy_payload_audit": gate_evidence(passed=False, commit=commit),
            "duckdb_catalog_audit": gate_evidence(passed=False, commit=commit),
            "concurrent_reader_test": gate_evidence(passed=False, commit=commit),
            "value_loader_smoke": gate_evidence(passed=False, commit=commit),
            "policy_loader_smoke": gate_evidence(passed=False, commit=commit),
            "required_tests": gate_evidence(passed=False, commit=commit),
            "engine_move_gen_parity": gate_evidence(
                passed=False,
                commit=commit,
                notes="Engine deployment gate only",
            ),
        }

    manifest: dict[str, Any] = {
        "schema_version": TEACHER_DATASET_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_sqlite": str(src_manifest.get("source_sqlite", TEACHER_STORE_DB)),
        "teacher_dataset_status": "candidate",
        "promotion_allowed": False,
        "immutable": True,
        "compression": src_manifest.get("compression", "zstd"),
        "parent_candidate": parent_candidate or str(source_dir.name),
        "recovery_method": recovery_method or src_manifest.get("recovery_method"),
        "canonical_identity_spec": spec_document(),
        "promotion_gates": gates,
        "counts": dict(src_manifest.get("counts") or {}),
        "policy_resolution": dict(src_manifest.get("policy_resolution") or {}),
        "parts": parts,
        "bytes": _bytes_for_parts(parts, root),
        "build_seconds": time.perf_counter() - t0,
        "manifest_hash": "",
    }
    if extra_manifest:
        for k, v in extra_manifest.items():
            if k not in ("manifest_hash", "promotion_allowed", "parts", "bytes"):
                manifest[k] = v

    # Gate evidence documents readiness; human promotion remains a separate decision.
    manifest["promotion_allowed"] = False
    manifest["manifest_hash"] = compute_manifest_hash(manifest)
    (partial / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    loaded = json.loads((partial / "manifest.json").read_text(encoding="utf-8"))
    assert compute_manifest_hash(loaded) == loaded["manifest_hash"]

    partial.rename(target_dir)

    manifest_path = target_dir / "manifest.json"
    final_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    final_parts = _part_paths(target_dir, root)
    final_manifest["parts"] = final_parts
    final_manifest["bytes"] = _bytes_for_parts(final_parts, root)
    final_manifest["promotion_allowed"] = False
    final_manifest["manifest_hash"] = compute_manifest_hash(final_manifest)
    manifest_path.write_text(json.dumps(final_manifest, indent=2), encoding="utf-8")
    return final_manifest


def finalize_with_gate_evidence(
    *,
    source_dir: Path,
    target_dir: Path,
    gate_bundle_path: Path,
    root: Path = ROOT,
) -> dict[str, Any]:
    """Copy source candidate to target.partial; attach gate evidence bundle; atomic rename."""
    if target_dir.exists():
        raise FileExistsError(f"target already exists: {target_dir}")
    bundle = json.loads(gate_bundle_path.resolve().read_text(encoding="utf-8"))
    gates = bundle.get("promotion_gates") or {}
    src_manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
    bundle_ref = gate_bundle_path.resolve()
    extra = {
        k: v
        for k, v in src_manifest.items()
        if k
        not in (
            "manifest_hash",
            "promotion_allowed",
            "promotion_gates",
            "teacher_dataset_status",
            "created_at",
            "parts",
            "bytes",
        )
    }
    extra["parity_audit"] = src_manifest.get("parity_audit")
    extra["jsonl_miss_classification"] = src_manifest.get("jsonl_miss_classification")
    extra["gate_evidence_bundle"] = {
        "path": str(bundle_ref.relative_to(root.resolve())).replace("\\", "/"),
        "sha256": bundle.get("bundle_sha256"),
        "generated_at": bundle.get("generated_at"),
    }
    return finalize_teacher_candidate(
        source_dir=source_dir,
        target_dir=target_dir,
        root=root,
        parent_candidate=str(source_dir.name),
        recovery_method=src_manifest.get("recovery_method"),
        extra_manifest=extra,
        promotion_gates=gates,
    )


def repair_manifest_paths(
    candidate_dir: Path,
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    """Rewrite manifest parts/bytes to match on-disk files under candidate_dir (no data copy)."""
    candidate_dir = candidate_dir.resolve()
    root = root.resolve()
    manifest_path = candidate_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    parts = _part_paths(candidate_dir, root)
    manifest["parts"] = parts
    manifest["bytes"] = _bytes_for_parts(parts, root)
    manifest = apply_promotion_allowed(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
