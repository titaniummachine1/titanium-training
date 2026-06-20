#!/usr/bin/env python3
"""Repository health checks for Titanium operators."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "lib"))

from repo_constants import (  # noqa: E402
    ACTIVE_MANIFEST_PATH,
    ACTIVE_MANIFEST_SHA256,
    APPROVED_DATASET_COUNTS,
    CANONICAL_DOCS,
    PROMOTION_RECEIPT,
    PROVENANCE_V10,
    REPO_ROOT,
)
from bundle_lib import sha256_file, verify_active_manifest, verify_provenance  # noqa: E402

BROKEN_LINK_RE = re.compile(r"\]\(([^)#]+)\)")


def check(name: str, ok: bool, detail: str = "") -> tuple[bool, str]:
    status = "PASS" if ok else "FAIL"
    line = f"[{status}] {name}"
    if detail:
        line += f" — {detail}"
    print(line)
    return ok, line


def check_structure() -> list[str]:
    failures: list[str] = []
    required_dirs = ["docs", "scripts/oracle", "scripts/maintenance", "training", "engine"]
    for rel in required_dirs:
        path = ROOT / rel
        ok, _ = check(f"structure/{rel}", path.exists(), "missing" if not path.exists() else "")
        if not ok:
            failures.append(rel)
    return failures


def check_canonical_entrypoints() -> list[str]:
    failures: list[str] = []
    entrypoints = [
        "scripts/oracle/build_upload_bundle.py",
        "scripts/oracle/verify_upload_bundle.py",
        "training/nnue_cli.py",
        "training/titanium_training/validation/smoke.py",
        "training/configs/smoke.yaml",
    ]
    for rel in entrypoints:
        ok, _ = check(f"entrypoint/{rel}", (ROOT / rel).is_file())
        if not ok:
            failures.append(rel)
    return failures


def check_dataset() -> list[str]:
    failures: list[str] = []
    manifest_path = ROOT / ACTIVE_MANIFEST_PATH.relative_to(REPO_ROOT)
    ok, _ = check("dataset/active_manifest_exists", manifest_path.is_file())
    if not ok:
        return ["active manifest missing"]

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ok, _ = check(
        "dataset/active_manifest_hash",
        manifest.get("manifest_hash") == ACTIVE_MANIFEST_SHA256,
        manifest.get("manifest_hash", ""),
    )
    if not ok:
        failures.append("manifest hash")

    for err in verify_active_manifest(root=ROOT):
        ok, _ = check(f"dataset/{err[:40]}", False, err)
        failures.append(err)

    rollback = sorted((ROOT / "training" / "data").glob("teacher_dataset_rollback_*"))
    ok, _ = check("dataset/rollback_present", len(rollback) >= 1, str(rollback[-1].name) if rollback else "none")
    if not ok:
        failures.append("rollback missing")

    partial_dirs = [
        p
        for p in (ROOT / "training" / "data").rglob("*")
        if p.is_dir() and (p.name.endswith(".partial") or p.name.endswith("_partial"))
    ]
    ok, _ = check("dataset/no_partial_trees", len(partial_dirs) == 0, f"found {len(partial_dirs)}")
    if not ok:
        failures.append("partial trees")

    return failures


def check_provenance() -> list[str]:
    failures: list[str] = []
    for path, label in ((PROVENANCE_V10, "provenance_v10"), (PROMOTION_RECEIPT, "promotion_receipt")):
        ok, _ = check(f"provenance/{label}", path.is_file())
        if not ok:
            failures.append(label)
    for err in verify_provenance(root=ROOT):
        ok, _ = check("provenance/verify", False, err)
        failures.append(err)
    return failures


def check_engine_submodule() -> list[str]:
    failures: list[str] = []
    ok, _ = check("engine/present", (ROOT / "engine").is_dir())
    if not ok:
        failures.append("engine")
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(ROOT / "engine"),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        check("engine/head", True, head[:12])
    except (OSError, subprocess.CalledProcessError):
        check("engine/head", False)
        failures.append("engine head")
    return failures


def check_docs() -> list[str]:
    failures: list[str] = []
    for rel in CANONICAL_DOCS:
        ok, _ = check(f"docs/{Path(rel).name}", (ROOT / rel).is_file())
        if not ok:
            failures.append(rel)
    return failures


def check_doc_links() -> list[str]:
    failures: list[str] = []
    for doc in (ROOT / "docs").rglob("*.md"):
        text = doc.read_text(encoding="utf-8", errors="replace")
        for match in BROKEN_LINK_RE.finditer(text):
            target = match.group(1).strip()
            if target.startswith("http") or target.startswith("#"):
                continue
            resolved = (doc.parent / target).resolve()
            if not resolved.is_file():
                rel = doc.relative_to(ROOT)
                failures.append(f"{rel}: {target}")
                check(f"link/{rel}", False, target)
    if not failures:
        check("links/internal", True, "0 broken")
    return failures


def check_imports() -> list[str]:
    failures: list[str] = []
    sys.path.insert(0, str(ROOT / "training"))
    try:
        import teacher_dataset  # noqa: F401

        check("import/teacher_dataset", True)
    except Exception as exc:
        check("import/teacher_dataset", False, str(exc))
        failures.append("teacher_dataset")
    return failures


def check_quarantine() -> list[str]:
    q = ROOT / ".cleanup_quarantine"
    ok, _ = check("cleanup/no_quarantine", not q.exists())
    return [] if ok else ["quarantine present"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="Emit JSON summary")
    args = ap.parse_args()

    print(f"Titanium repository doctor — {ROOT}\n")
    all_failures: list[str] = []
    for fn in (
        check_structure,
        check_canonical_entrypoints,
        check_dataset,
        check_provenance,
        check_engine_submodule,
        check_docs,
        check_doc_links,
        check_imports,
        check_quarantine,
    ):
        all_failures.extend(fn())

    print()
    if all_failures:
        print(f"RESULT: FAIL ({len(all_failures)} issue(s))")
        if args.json:
            print(json.dumps({"passed": False, "failures": all_failures}, indent=2))
        return 1

    print("RESULT: PASS")
    if args.json:
        print(json.dumps({"passed": True, "active_manifest_sha256": ACTIVE_MANIFEST_SHA256}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
