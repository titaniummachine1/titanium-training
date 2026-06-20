"""Oracle upload bundle construction and verification helpers."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from repo_constants import (
    ACTIVE_MANIFEST_PATH,
    ACTIVE_MANIFEST_SHA256,
    ACTIVE_TEACHER_DATASET,
    APPROVED_DATASET_COUNTS,
    AUDIT_PAYLOAD_SHA256,
    FINAL_EVIDENCE_ENVELOPE_SHA256,
    FORBIDDEN_BUNDLE_PREFIXES,
    ORACLE_CODE_PATHS,
    PROMOTION_RECEIPT,
    PROMOTION_RECEIPT_SHA256,
    PROVENANCE_V10,
    REPO_ROOT,
    TEST_EVIDENCE_SHA256,
)

_WINDOWS_ABS = re.compile(r"^[A-Za-z]:\\|^\\\\")
_UNIX_ABS = re.compile(r"^/(?:home|Users|mnt|opt)/")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel_posix(path: Path, *, root: Path = REPO_ROOT) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def git_head(root: Path = REPO_ROOT) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def is_forbidden_bundle_path(rel: str) -> bool:
    rel = rel.replace("\\", "/")
    if rel.endswith("/"):
        rel = rel[:-1]
    for prefix in FORBIDDEN_BUNDLE_PREFIXES:
        if rel == prefix.rstrip("/") or rel.startswith(prefix):
            return True
    if ".partial" in rel:
        return True
    if "/__pycache__/" in f"/{rel}/":
        return True
    if rel.startswith("training/data/"):
        return True
    if rel.startswith("training/checkpoints_smoke/"):
        return True
    if rel == "docs/maintenance/repository_inventory.json":
        return True
    return False


def verify_active_manifest(*, root: Path = REPO_ROOT) -> list[str]:
    errors: list[str] = []
    manifest_path = root / rel_posix(ACTIVE_MANIFEST_PATH, root=root)
    if not manifest_path.is_file():
        return [f"missing active manifest: {manifest_path}"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("manifest_hash") != ACTIVE_MANIFEST_SHA256:
        errors.append(
            f"active manifest hash mismatch: got {manifest.get('manifest_hash')!r}, "
            f"expected {ACTIVE_MANIFEST_SHA256}"
        )
    counts = manifest.get("counts") or {}
    for key, expected in APPROVED_DATASET_COUNTS.items():
        if key == "unresolved":
            continue
        if key == "labels_without_policy":
            actual = counts.get("labels_without_policy", counts.get("no_policy_labels"))
            if actual is None and "labels" in counts and "has_policy_labels" in counts:
                actual = int(counts["labels"]) - int(counts["has_policy_labels"])
            actual = int(actual if actual is not None else -1)
        else:
            actual = int(counts.get(key, -1))
        if actual != expected:
            errors.append(f"count mismatch {key}: got {actual}, expected {expected}")
    return errors


def verify_provenance(*, root: Path = REPO_ROOT) -> list[str]:
    errors: list[str] = []
    for path, label in (
        (PROVENANCE_V10, "provenance v10"),
        (PROMOTION_RECEIPT, "promotion receipt"),
    ):
        p = root / rel_posix(path, root=root)
        if not p.is_file():
            errors.append(f"missing {label}: {p}")
            continue
        digest = sha256_file(p)
        if label == "promotion receipt" and digest != PROMOTION_RECEIPT_SHA256:
            errors.append(f"promotion receipt sha256 mismatch: {digest}")
    return errors


def iter_source_paths(*, include_dataset: bool, root: Path = REPO_ROOT) -> Iterable[Path]:
    for rel in ORACLE_CODE_PATHS:
        path = root / rel
        if not path.exists():
            continue
        if path.is_file():
            yield path
        else:
            for child in path.rglob("*"):
                if child.is_file() and not is_forbidden_bundle_path(rel_posix(child, root=root)):
                    yield child
    if include_dataset:
        dataset = root / "training" / "data" / "teacher_dataset"
        if dataset.is_dir():
            for child in dataset.rglob("*"):
                if child.is_file():
                    yield child


def categorize(rel: str) -> str:
    rel = rel.replace("\\", "/")
    if rel.startswith("training/data/teacher_dataset/"):
        return "teacher_dataset"
    if rel.startswith("training/teacher_dataset/"):
        return "teacher_dataset_code"
    if rel.startswith("scripts/oracle/"):
        return "oracle_scripts"
    if rel.startswith("scripts/maintenance/"):
        return "maintenance"
    if rel.startswith("docs/"):
        return "documentation"
    if rel.startswith("training/configs/"):
        return "training_config"
    if rel.startswith("training/"):
        return "training_code"
    if rel.startswith("tools/"):
        return "tools"
    return "other"


@dataclass
class BundleResult:
    output_dir: Path
    manifest_path: Path
    manifest_sha256: str
    file_count: int
    total_bytes: int
    include_dataset: bool
    errors: list[str] = field(default_factory=list)


def build_bundle(
    output_dir: Path,
    *,
    include_dataset: bool = False,
    code_only: bool = False,
    root: Path = REPO_ROOT,
    dry_run: bool = False,
) -> BundleResult:
    if code_only:
        include_dataset = False
    errors = verify_active_manifest(root=root) if include_dataset else []
    errors.extend(verify_provenance(root=root))
    if include_dataset:
        try:
            import sys

            sys_path = str(root / "training")
            if sys_path not in sys.path:
                sys.path.insert(0, sys_path)
            from teacher_dataset.verify_artifacts import verify_candidate_artifacts

            report = verify_candidate_artifacts(
                root / "training" / "data" / "teacher_dataset",
                root=root,
                sample_policy_records=100,
            )
            if not report.passed:
                errors.append(f"artifact verification failed: {report.to_dict()}")
        except Exception as exc:
            errors.append(f"artifact verification error: {exc}")

    if errors and not dry_run:
        return BundleResult(
            output_dir=output_dir,
            manifest_path=output_dir / "transfer-manifest.json",
            manifest_sha256="",
            file_count=0,
            total_bytes=0,
            include_dataset=include_dataset,
            errors=errors,
        )

    if output_dir.exists() and not dry_run:
        import stat

        def _on_rm_error(func, path, exc_info):
            import os
            os.chmod(path, stat.S_IWRITE)
            func(path)

        shutil.rmtree(output_dir, onexc=_on_rm_error)
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []
    total_bytes = 0
    for src in iter_source_paths(include_dataset=include_dataset, root=root):
        rel = rel_posix(src, root=root)
        if is_forbidden_bundle_path(rel):
            if not (include_dataset and rel.startswith("training/data/teacher_dataset/")):
                continue
        size = src.stat().st_size
        digest = sha256_file(src) if not dry_run else ""
        if not dry_run:
            dest = output_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
        entries.append(
            {
                "path": rel,
                "size_bytes": size,
                "sha256": digest,
                "category": categorize(rel),
            }
        )
        total_bytes += size

    entries.sort(key=lambda e: e["path"])
    manifest_doc = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repository_head": git_head(root),
        "include_dataset": include_dataset,
        "code_only": code_only,
        "active_manifest_sha256": ACTIVE_MANIFEST_SHA256 if include_dataset else None,
        "audit_identities": {
            "audit_payload_sha256": AUDIT_PAYLOAD_SHA256,
            "final_evidence_envelope_sha256": FINAL_EVIDENCE_ENVELOPE_SHA256,
            "test_evidence_sha256": TEST_EVIDENCE_SHA256,
            "promotion_receipt_sha256": PROMOTION_RECEIPT_SHA256,
        },
        "files": entries,
    }
    manifest_bytes = json.dumps(manifest_doc, indent=2, sort_keys=True).encode()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    manifest_doc["manifest_sha256"] = manifest_sha256

    readme = _readme_first(include_dataset=include_dataset, file_count=len(entries), total_bytes=total_bytes)
    if not dry_run:
        (output_dir / "transfer-manifest.json").write_bytes(
            json.dumps(manifest_doc, indent=2, sort_keys=True).encode()
        )
        (output_dir / "README_FIRST.md").write_text(readme, encoding="utf-8")

    return BundleResult(
        output_dir=output_dir,
        manifest_path=output_dir / "transfer-manifest.json",
        manifest_sha256=manifest_sha256,
        file_count=len(entries),
        total_bytes=total_bytes,
        include_dataset=include_dataset,
        errors=errors,
    )


def _readme_first(*, include_dataset: bool, file_count: int, total_bytes: int) -> str:
    dataset_line = (
        "This bundle **includes** `training/data/teacher_dataset/` (promoted v10)."
        if include_dataset
        else "This bundle is **code-only** — copy `training/data/teacher_dataset/` separately or rebuild with `--include-active-dataset`."
    )
    return f"""# Titanium Oracle upload bundle

{dataset_line}

## Verify before transfer

```bash
python scripts/oracle/verify_upload_bundle.py .
```

## Bootstrap on Oracle (Linux ARM)

```bash
bash scripts/oracle/bootstrap.sh
bash scripts/oracle/doctor.sh
bash scripts/oracle/smoke_train.sh
```

## Contents

- Files: {file_count:,}
- Total bytes: {total_bytes:,}

Active manifest SHA256 (when dataset included): `{ACTIVE_MANIFEST_SHA256}`

Do **not** modify files under `training/data/teacher_dataset/` after packaging.
"""


def verify_bundle(bundle_dir: Path, *, root: Path = REPO_ROOT) -> tuple[bool, list[str]]:
    errors: list[str] = []
    manifest_path = bundle_dir / "transfer-manifest.json"
    if not manifest_path.is_file():
        return False, ["missing transfer-manifest.json"]

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw = json.dumps({k: v for k, v in manifest.items() if k != "manifest_sha256"}, indent=2, sort_keys=True).encode()
    expected = hashlib.sha256(raw).hexdigest()
    if manifest.get("manifest_sha256") != expected:
        errors.append("transfer-manifest self-hash mismatch")

    include_dataset = bool(manifest.get("include_dataset"))
    for entry in manifest.get("files") or []:
        rel = entry["path"].replace("\\", "/")
        if is_forbidden_bundle_path(rel):
            if not (include_dataset and rel.startswith("training/data/teacher_dataset/")):
                errors.append(f"forbidden path in bundle: {rel}")
                continue
        dest = bundle_dir / rel
        if not dest.is_file():
            errors.append(f"missing bundled file: {rel}")
            continue
        digest = sha256_file(dest)
        if digest != entry.get("sha256"):
            errors.append(f"hash mismatch: {rel}")
        text_ext = {".yaml", ".yml", ".json", ".md", ".txt", ".sh", ".py", ".toml", ".ini"}
        if dest.suffix.lower() in text_ext and _contains_local_abs_path(dest.read_text(encoding="utf-8", errors="replace")):
            errors.append(f"absolute local path in bundled file: {rel}")

    if include_dataset:
        errors.extend(verify_active_manifest(root=bundle_dir))
        ds_manifest = bundle_dir / "training" / "data" / "teacher_dataset" / "manifest.json"
        if not ds_manifest.is_file():
            errors.append("bundle missing active dataset manifest")

    required = [
        "README_FIRST.md",
        "scripts/oracle/build_upload_bundle.py",
        "scripts/oracle/verify_upload_bundle.py",
        "scripts/maintenance/repository_doctor.py",
        "training/nnue_cli.py",
        "docs/ORACLE_DEPLOYMENT.md",
    ]
    for rel in required:
        if not (bundle_dir / rel).is_file():
            errors.append(f"missing required bundle file: {rel}")

    return len(errors) == 0, errors


def _contains_local_abs_path(text: str) -> bool:
    for line in text.splitlines():
        stripped = line.strip()
        if _WINDOWS_ABS.search(stripped) or _UNIX_ABS.search(stripped):
            if "REPO_ROOT" in stripped or "Path(__file__)" in stripped:
                continue
            if "example" in stripped.lower():
                continue
            return True
    return False
