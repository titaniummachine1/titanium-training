#!/usr/bin/env python3
"""Build an Oracle upload bundle from the Titanium repository."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "lib"))

from bundle_lib import build_bundle  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--output",
        default=str(ROOT / "dist" / "oracle_upload"),
        help="Output directory for the bundle",
    )
    ap.add_argument(
        "--include-active-dataset",
        action="store_true",
        help="Copy training/data/teacher_dataset/ into the bundle",
    )
    ap.add_argument(
        "--code-only",
        action="store_true",
        help="Package training code and docs only (no dataset)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print manifest stats without copying files",
    )
    ap.add_argument(
        "--archive",
        action="store_true",
        help="Create a .tar.gz next to the output directory when tar is available",
    )
    args = ap.parse_args()

    output = Path(args.output)
    result = build_bundle(
        output,
        include_dataset=args.include_active_dataset,
        code_only=args.code_only,
        root=ROOT,
        dry_run=args.dry_run,
    )
    if result.errors:
        print("Bundle build FAILED:", file=sys.stderr)
        for err in result.errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print(f"Bundle {'plan' if args.dry_run else 'built'}: {output}")
    print(f"  files: {result.file_count:,}")
    print(f"  bytes: {result.total_bytes:,}")
    print(f"  include_dataset: {result.include_dataset}")
    if not args.dry_run:
        print(f"  manifest_sha256: {result.manifest_sha256}")

    if args.archive and not args.dry_run:
        import shutil
        import tarfile

        archive = output.with_suffix(".tar.gz")
        if archive.exists():
            archive.unlink()
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(output, arcname=output.name)
        print(f"  archive: {archive}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
