"""Hash-addressed promoted weight storage for Oracle VM."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from .protocol import atomic_write_json, read_json, sha256_file, utc_now

WEIGHTS_DIR_NAME = "weights"
MANIFEST_NAME = "manifest.json"


class WeightStore:
    """Store weight blobs by SHA256 under ``<root>/weights/<hash>.bin``."""

    def __init__(self, root: Path):
        self.root = root
        self.weights_dir = root / WEIGHTS_DIR_NAME
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.weights_dir / MANIFEST_NAME

    def read_manifest(self) -> dict[str, Any]:
        return read_json(self.manifest_path) or {}

    def weight_path(self, sha256_hex: str) -> Path:
        return self.weights_dir / f"{sha256_hex.lower()}.bin"

    def sha256_exists(self, sha256_hex: str) -> bool:
        path = self.weight_path(sha256_hex)
        if not path.is_file():
            return False
        return sha256_file(path) == sha256_hex.lower()

    def install_weight(self, local_path: Path, expected_sha: str) -> bool:
        """Upload-style install with tmp + verify. Returns True if newly installed."""
        expected = expected_sha.lower()
        final = self.weight_path(expected)
        if self.sha256_exists(expected):
            return False
        tmp = Path(str(final) + ".tmp")
        if tmp.exists():
            tmp.unlink()
        shutil.copy2(local_path, tmp)
        actual = sha256_file(tmp)
        if actual != expected:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"Weight checksum mismatch: expected={expected}, actual={actual}")
        os.replace(tmp, final)
        return True

    def update_manifest(
        self,
        *,
        generation: int,
        current_sha: str,
        previous_sha: str | None,
        engine_sha: str,
    ) -> dict[str, Any]:
        prev = previous_sha.lower() if previous_sha else None
        cur = current_sha.lower()
        manifest = {
            "generation": generation,
            "current": {"sha256": cur, "filename": f"{cur}.bin"},
            "previous": (
                {"sha256": prev, "filename": f"{prev}.bin"}
                if prev and prev != cur
                else None
            ),
            "engine_sha256": engine_sha,
            "updated_at": utc_now(),
        }
        atomic_write_json(self.manifest_path, manifest)
        return manifest

    def materialize_generation_links(self, gen_dir: Path, manifest: dict[str, Any]) -> None:
        """Symlink generation current.bin / prior.bin to hash-addressed weights."""
        cur = manifest["current"]["sha256"]
        cur_src = self.weight_path(cur)
        _link_or_copy(cur_src, gen_dir / "current.bin")

        prev = manifest.get("previous")
        prior_dst = gen_dir / "prior.bin"
        if prev and prev.get("sha256"):
            prev_src = self.weight_path(prev["sha256"])
            _link_or_copy(prev_src, prior_dst)
        elif prior_dst.exists() or prior_dst.is_symlink():
            prior_dst.unlink()
            _link_or_copy(cur_src, prior_dst)

    def prune_unreferenced(self, *, keep_hashes: set[str]) -> list[str]:
        removed: list[str] = []
        for path in self.weights_dir.glob("*.bin"):
            if path.name.endswith(".tmp"):
                continue
            h = path.stem.lower()
            if h not in keep_hashes:
                path.unlink(missing_ok=True)
                removed.append(h)
        return removed


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)
