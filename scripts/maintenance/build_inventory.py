#!/usr/bin/env python3
"""Generate repository inventory JSON and cleanup plan markdown."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CLASSIFICATIONS = (
    "KEEP_CANONICAL",
    "KEEP_SUPPORTING",
    "MOVE",
    "MERGE",
    "RENAME",
    "DELETE_PROVEN_DEAD",
    "GENERATED_IGNORE",
    "LOCAL_DATA",
    "HISTORICAL_DECISION",
    "UNKNOWN_REQUIRES_REVIEW",
)

DOC_PATHS = {
    "docs/",
    "training/README.md",
    "training/ARCHITECTURE_HANDOFF.md",
    "training/POSITION_STORE_RUNBOOK.md",
    "training/CANONICAL_DATASTORE.md",
}


def git_tracked(root: Path) -> set[str]:
    try:
        out = subprocess.check_output(
            ["git", "ls-files"],
            cwd=str(root),
            text=True,
        )
        return {line.strip().replace("\\", "/") for line in out.splitlines() if line.strip()}
    except (OSError, subprocess.CalledProcessError):
        return set()


def classify(rel: str, tracked: bool) -> tuple[str, str, str]:
    rel = rel.replace("\\", "/")
    if rel.startswith("training/data/teacher_dataset/"):
        return "LOCAL_DATA", "KEEP", "Active promoted teacher dataset — do not modify"
    if "rollback" in rel and "teacher_dataset" in rel:
        return "LOCAL_DATA", "KEEP", "Rollback dataset — preserve locally"
    if ".partial" in rel:
        return "GENERATED_IGNORE", "DELETE_PROVEN_DEAD", "Stale partial candidate tree"
    if rel.startswith("engine/target") or rel.endswith(".pyc") or "__pycache__" in rel:
        return "GENERATED_IGNORE", "DELETE_PROVEN_DEAD", "Build/cache artifact"
    if rel.startswith("docs/"):
        return "KEEP_CANONICAL", "KEEP", "Canonical documentation"
    if rel.startswith("scripts/"):
        return "KEEP_CANONICAL", "KEEP", "Oracle/maintenance tooling"
    if rel.startswith("training/teacher_dataset/"):
        return "KEEP_CANONICAL", "KEEP", "Teacher dataset package"
    if rel in {"training/plan.md", "training/AUDIT_REPORT.md", "training/data/handoff.txt"}:
        return "HISTORICAL_DECISION", "MERGE", "Superseded by docs/ — merge or remove"
    if not tracked and rel.startswith("training/data/"):
        return "LOCAL_DATA", "KEEP", "Runtime training data"
    if tracked:
        return "KEEP_SUPPORTING", "KEEP", "Tracked source"
    return "UNKNOWN_REQUIRES_REVIEW", "KEEP", "Review before deletion"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-json", default=str(ROOT / "docs" / "maintenance" / "repository_inventory.json"))
    ap.add_argument("--output-plan", default=str(ROOT / "docs" / "maintenance" / "repository_cleanup_plan.md"))
    args = ap.parse_args()

    tracked = git_tracked(ROOT)
    entries: list[dict] = []
    skip_dirs = {".git", "KaAiData", "engine/target", "engine/target-profile", "site/web/node_modules"}
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        if any(part in skip_dirs for part in path.parts):
            continue
        rel = path.relative_to(ROOT).as_posix()
        if rel.startswith("training/data/teacher_dataset/") and path.name not in {"manifest.json", "schema.json"}:
            continue
        if rel.startswith("training/data/teacher_dataset_rollback_"):
            continue
        is_tracked = rel in tracked
        classification, action, reason = classify(rel, is_tracked)
        git_date = ""
        if is_tracked:
            try:
                git_date = subprocess.check_output(
                    ["git", "log", "-1", "--format=%cI", "--", rel],
                    cwd=str(ROOT),
                    stderr=subprocess.DEVNULL,
                    text=True,
                ).strip()
            except (OSError, subprocess.CalledProcessError):
                git_date = ""
        entries.append(
            {
                "path": rel,
                "file_type": path.suffix.lower() or "none",
                "tracked": is_tracked,
                "size_bytes": path.stat().st_size,
                "last_git_modification": git_date,
                "classification": classification,
                "proposed_action": action,
                "reason": reason,
                "risk": "high" if "teacher_dataset" in rel and "rollback" not in rel else "low",
            }
        )

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"generated_at": datetime.now(timezone.utc).isoformat(), "entries": entries}, indent=2), encoding="utf-8")

    delete_candidates = [e for e in entries if e["proposed_action"] == "DELETE_PROVEN_DEAD"]
    merge_candidates = [e for e in entries if e["proposed_action"] == "MERGE"]
    plan_lines = [
        "# Repository cleanup plan",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Summary",
        "",
        f"- Files inventoried: {len(entries):,}",
        f"- Tracked: {sum(1 for e in entries if e['tracked']):,}",
        f"- Delete candidates: {len(delete_candidates):,}",
        f"- Merge candidates: {len(merge_candidates):,}",
        "",
        "## Delete candidates (proven dead / generated)",
        "",
    ]
    for e in delete_candidates[:100]:
        plan_lines.append(f"- `{e['path']}` — {e['reason']}")
    if len(delete_candidates) > 100:
        plan_lines.append(f"- … and {len(delete_candidates) - 100} more")

    plan_lines.extend(["", "## Merge / consolidate", ""])
    for e in merge_candidates:
        plan_lines.append(f"- `{e['path']}` — {e['reason']}")

    out_plan = Path(args.output_plan)
    out_plan.write_text("\n".join(plan_lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_plan}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
