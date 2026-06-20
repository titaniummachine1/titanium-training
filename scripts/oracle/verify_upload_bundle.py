#!/usr/bin/env python3
"""Verify an Oracle upload bundle before transfer or after extraction."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "lib"))

from bundle_lib import verify_bundle  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bundle_dir", help="Path to extracted bundle directory")
    args = ap.parse_args()

    bundle_dir = Path(args.bundle_dir).resolve()
    ok, errors = verify_bundle(bundle_dir)
    if ok:
        print(f"PASS: bundle verified at {bundle_dir}")
        return 0

    print(f"FAIL: bundle verification at {bundle_dir}", file=sys.stderr)
    for err in errors:
        print(f"  - {err}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
