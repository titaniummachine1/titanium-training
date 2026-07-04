#!/usr/bin/env python3
"""Deploy latest accepted streaming weights to engine live blob + website WASM."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_TRAINING = Path(__file__).resolve().parent
_REPO = _TRAINING.parent
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

from streaming_checkpoint_chain import (
    ENGINE_WEIGHTS,
    FROZEN_WEIGHTS,
    atomic_copy2,
    latest_accepted,
    resolve_accepted_weights,
    resolve_latest_accepted_weights,
    sha256_file,
)
from titanium_training.validation.opening_sanity import assert_opening_sanity

SITE_WEB = _REPO / "site" / "web"
BUILD_META = SITE_WEB / "src" / "wasm" / "titanium" / "build-meta.json"


def deploy_weights(src: Path) -> Path:
    assert_opening_sanity(src)
    atomic_copy2(src, ENGINE_WEIGHTS)
    return ENGINE_WEIGHTS


def rebuild_native_engine() -> None:
    """net.rs bakes weights in via include_bytes!("net_weights.bin") at
    COMPILE time -- copying a new file there does nothing to the native
    titanium.exe binary until it's rebuilt. Only TITANIUM_NET_WEIGHTS_PATH
    overrides read the file live; anything that calls the native binary
    without that env var (parity_check.py, `titanium eval`, etc.) silently
    keeps evaluating with whatever was baked in at the last build. Confirmed
    live: deploying epoch 8 without this step made parity_check.py compare
    epoch 8's real weights against a stale baked-in binary -- 49-87cp
    "mismatches" that were actually just a missing rebuild, not a real bug."""
    env = __import__("os").environ.copy()
    env["RUSTFLAGS"] = "-C target-cpu=native"
    proc = subprocess.run(
        ["cargo", "build", "--release", "--bin", "titanium"],
        cwd=str(_REPO / "engine"),
        env=env,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"native engine rebuild failed (rc={proc.returncode})")


def build_wasm() -> None:
    npm = "npm.cmd" if sys.platform == "win32" else "npm"
    proc = subprocess.run(
        [npm, "run", "build:wasm"],
        cwd=str(SITE_WEB),
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"npm run build:wasm failed (rc={proc.returncode})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Deploy a specific accepted epoch (default: latest)",
    )
    ap.add_argument(
        "--skip-wasm",
        action="store_true",
        help="Only copy weights to engine/src/titanium/net_weights.bin",
    )
    args = ap.parse_args()

    if args.epoch is not None:
        from streaming_checkpoint_chain import load_chain

        matches = [e for e in load_chain().get("epochs") or [] if int(e["epoch"]) == args.epoch]
        if not matches:
            raise SystemExit(f"no accepted epoch {args.epoch} in chain")
        src = resolve_accepted_weights(matches[-1])
        entry = matches[-1]
    else:
        entry = latest_accepted()
        if entry is None:
            raise SystemExit("no accepted checkpoints in chain")
        src = resolve_latest_accepted_weights()

    frozen_before = sha256_file(FROZEN_WEIGHTS) if FROZEN_WEIGHTS.is_file() else None
    dest = deploy_weights(src)
    if frozen_before and sha256_file(FROZEN_WEIGHTS) != frozen_before:
        raise SystemExit(f"REFUSING: frozen weights were modified at {FROZEN_WEIGHTS}")

    print(f"Deployed epoch {entry['epoch']} -> {dest}")
    print(f"  source: {src}")
    print(f"  sha256: {entry['sha256'][:16]}…")

    print("Rebuilding native engine (bakes new weights into titanium.exe)...")
    rebuild_native_engine()

    if not args.skip_wasm:
        print("Building website WASM (embeds live weights)...")
        build_wasm()
        if BUILD_META.is_file():
            meta = json.loads(BUILD_META.read_text(encoding="utf-8"))
            print(
                f"WASM ready: weights_live_sha256={meta.get('weights_live_sha256', '?')[:16]}… "
                f"built {meta.get('build_timestamp')}"
            )
        print("Local preview: cd site/web && npm run preview:pages")
        print("GitHub Pages: commit site/web/src/wasm and push to main")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
