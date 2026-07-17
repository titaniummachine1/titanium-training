"""Validated titanium binary identity for training and pool runs.

All NNUE training inputs must come from one checked release binary.  The stamp is
path + sha256 + size; if it exists, a different binary blocks training/data gen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from titanium_training.paths import ENGINE_BIN, REPO_ROOT, TRAINING_ROOT

ROOT = REPO_ROOT
BIN = ENGINE_BIN
STAMP = Path(
    os.environ.get(
        "TITANIUM_ENGINE_STAMP",
        str(TRAINING_ROOT / "data" / "engine_stamp.json"),
    )
)


def binary_stamp(path: Path = BIN) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"missing titanium binary: {path}")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "sha256": h.hexdigest(),
        "size": stat.st_size,
    }


def load_expected_stamp(path: Path = STAMP) -> dict | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def write_expected_stamp(stamp: dict, path: Path = STAMP) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stamp, indent=2) + "\n", encoding="utf-8")


def assert_binary_identity(*, write_if_missing: bool = False) -> dict:
    current = binary_stamp(BIN)
    expected = load_expected_stamp()
    if expected is None:
        if write_if_missing:
            return current
        raise RuntimeError(
            f"engine stamp missing: {STAMP}. Run training/titanium_training/validation/engine_identity.py --write "
            "after a native release build and parity check."
        )

    keys = ("path", "sha256", "size")
    mismatch = [k for k in keys if expected.get(k) != current.get(k)]
    if mismatch:
        if write_if_missing:
            return current
        details = ", ".join(f"{k}: expected {expected.get(k)!r}, got {current.get(k)!r}" for k in mismatch)
        raise RuntimeError(f"titanium binary identity mismatch ({details})")
    return current


def assert_legal_wall_schema() -> None:
    result = subprocess.run(
        [str(BIN), "eval-batch"],
        input=b"\n",
        capture_output=True,
        check=True,
        timeout=30,
    )
    line = result.stdout.decode("utf-8", errors="replace").splitlines()[0]
    rec = json.loads(line)
    if rec.get("legal_wall_count") != 0:
        raise RuntimeError(f"legal_wall_count schema mismatch on startpos: {rec.get('legal_wall_count')!r}")
    if "legal_path_cross_p0" not in rec or "legal_path_cross_p1" not in rec:
        raise RuntimeError("eval-batch record missing legal_path_cross_p0/p1 (rebuild engine)")


def assert_parity_6_of_6() -> None:
    result = subprocess.run(
        [sys.executable, str(TRAINING_ROOT / "titanium_training" / "validation" / "parity_check.py")],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        tail = (result.stdout + "\n" + result.stderr).strip().splitlines()[-20:]
        raise RuntimeError("parity_check failed; training blocked:\n" + "\n".join(tail))


def assert_engine_ready(*, write_if_missing: bool = False, parity: bool = True) -> dict:
    stamp = assert_binary_identity(write_if_missing=write_if_missing)
    assert_legal_wall_schema()
    if parity:
        assert_parity_6_of_6()
    if write_if_missing:
        write_expected_stamp(stamp)
    return stamp


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true", help="write/refresh stamp after validation")
    ap.add_argument("--no-parity", action="store_true", help="skip parity_check.py")
    args = ap.parse_args()
    stamp = assert_engine_ready(write_if_missing=args.write, parity=not args.no_parity)
    if args.write:
        write_expected_stamp(stamp)  # idempotent after assert_engine_ready
    print(json.dumps(stamp, indent=2))


if __name__ == "__main__":
    main()
