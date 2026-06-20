#!/usr/bin/env python3
"""Validate internal Markdown links and documented script paths."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

LINK_RE = re.compile(r"\]\(([^)#]+)\)")
BACKTICK_PATH_RE = re.compile(r"`((?:training|scripts|docs)/[^`]+)`")


def resolve_link(source: Path, target: str) -> Path:
    target = target.split("#", 1)[0].strip()
    if not target:
        return source
    return (source.parent / target).resolve()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(ROOT / "docs"))
    args = ap.parse_args()

    root = Path(args.root)
    errors: list[str] = []

    for doc in root.rglob("*.md"):
        text = doc.read_text(encoding="utf-8", errors="replace")
        for match in LINK_RE.finditer(text):
            target = match.group(1).strip()
            if target.startswith("http") or target.startswith("#"):
                continue
            resolved = resolve_link(doc, target)
            if not resolved.is_file():
                errors.append(f"{doc.relative_to(ROOT)}: broken link -> {target}")

        for match in BACKTICK_PATH_RE.finditer(text):
            rel = match.group(1).replace("\\", "/")
            if any(ch in rel for ch in "*<>"):
                continue
            path = ROOT / rel
            if not path.exists():
                errors.append(f"{doc.relative_to(ROOT)}: missing path `{rel}`")

    readme = ROOT / "README.md"
    if readme.is_file():
        text = readme.read_text(encoding="utf-8", errors="replace")
        for match in LINK_RE.finditer(text):
            target = match.group(1).strip()
            if target.startswith("http") or target.startswith("#"):
                continue
            resolved = resolve_link(readme, target)
            if not resolved.is_file():
                errors.append(f"README.md: broken link -> {target}")

    if errors:
        print(f"FAIL: {len(errors)} documentation issue(s)", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print("PASS: documentation links and referenced paths OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
