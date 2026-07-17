"""Atomic deployed-generation staging and activation."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from .protocol import atomic_write_json, read_json, sha256_file
from .weight_store import WeightStore


class GenerationStore:
    def __init__(self, root: Path):
        self.root = root
        self.staging = root / "staging"
        self.active_link = root / "active"
        install_root = Path("/opt/titanium-game-factory")
        if install_root.is_dir():
            self.weights = WeightStore(install_root)
        else:
            self.weights = WeightStore(root.parent)
        self.root.mkdir(parents=True, exist_ok=True)
        self.staging.mkdir(parents=True, exist_ok=True)

    def active_dir(self) -> Path | None:
        if not self.active_link.exists():
            return None
        return self.active_link.resolve()

    def active_manifest(self) -> dict[str, Any]:
        active = self.active_dir()
        if not active:
            return {}
        return read_json(active / "generation.json")

    def stage(self, source: Path) -> dict[str, Any]:
        source = source.resolve()
        manifest = read_json(source / "generation.json")
        if not manifest:
            raise ValueError("generation.json missing")
        for name, key in (("current.bin", "current_deployed_hash"), ("prior.bin", "prior_deployed_hash")):
            path = source / name
            if not path.is_file():
                raise ValueError(f"{name} missing")
            if sha256_file(path) != manifest.get(key):
                raise ValueError(f"{name} hash mismatch")

        gen_id = str(manifest["generation_id"])
        dest = self.staging / gen_id
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(source, dest)

        cur_sha = str(manifest["current_deployed_hash"]).lower()
        prev_sha = str(manifest.get("prior_deployed_hash", cur_sha)).lower()
        self.weights.install_weight(source / "current.bin", cur_sha)
        if prev_sha != cur_sha:
            self.weights.install_weight(source / "prior.bin", prev_sha)
        weights_manifest = manifest.get("weights_manifest")
        if weights_manifest:
            atomic_write_json(self.weights.manifest_path, weights_manifest)
        return {"staged": gen_id, "path": str(dest)}

    def activate(self, generation_id: str) -> dict[str, Any]:
        src = self.staging / generation_id
        if not src.is_dir():
            raise FileNotFoundError(generation_id)
        manifest = read_json(src / "generation.json")
        for name, key in (("current.bin", "current_deployed_hash"), ("prior.bin", "prior_deployed_hash")):
            if sha256_file(src / name) != manifest.get(key):
                raise ValueError(f"{name} hash mismatch")

        live = self.root / "generations" / generation_id
        live.parent.mkdir(parents=True, exist_ok=True)
        if live.exists():
            shutil.rmtree(live)
        shutil.move(str(src), str(live))

        cur_sha = str(manifest["current_deployed_hash"]).lower()
        prev_sha = str(manifest.get("prior_deployed_hash", cur_sha)).lower()
        keep = {cur_sha}
        if prev_sha != cur_sha:
            keep.add(prev_sha)
        self.weights.materialize_generation_links(live, {
            "current": {"sha256": cur_sha},
            "previous": {"sha256": prev_sha} if prev_sha != cur_sha else None,
        })
        self.weights.prune_unreferenced(keep_hashes=keep)

        tmp_link = self.root / ".active.tmp"
        if tmp_link.exists() or tmp_link.is_symlink():
            tmp_link.unlink()
        try:
            os.symlink(live, tmp_link, target_is_directory=True)
            os.replace(tmp_link, self.active_link)
        except OSError:
            # Windows/non-symlink fallback for tests; Linux install uses symlink.
            atomic_write_json(self.root / "active.json", {"path": str(live), "generation_id": generation_id})
            if self.active_link.exists() and self.active_link.is_dir():
                shutil.rmtree(self.active_link)
            shutil.copytree(live, self.active_link, dirs_exist_ok=True)
        return {"activated": generation_id, "manifest": manifest}

