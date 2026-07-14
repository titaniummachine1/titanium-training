"""Corpus semantic versioning manifest — required for any finalized corpus."""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from diversity.lanes import DIVERSITY_SPEC_VERSION

REQUIRED_FIELDS = (
    "engine_semantic_version",
    "engine_binary_hash",
    "nnue_feature_schema_version",
    "move_encoding_version",
    "evaluation_semantics_version",
    "solver_oracle_version",
    "search_configuration_version",
    "diversity_spec_version",
    "generation_configuration_hash",
    "label_configuration_hash",
    "source_commit_hash",
    "generation_timestamp",
    "corpus_generation_id",
)


@dataclass(frozen=True)
class CorpusSemanticManifest:
    engine_semantic_version: str
    engine_binary_hash: str
    nnue_feature_schema_version: str
    move_encoding_version: str
    evaluation_semantics_version: str
    solver_oracle_version: str
    search_configuration_version: str
    diversity_spec_version: str
    generation_configuration_hash: str
    label_configuration_hash: str
    source_commit_hash: str
    generation_timestamp: str
    corpus_generation_id: str
    dirty_tree_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out = {f.name: getattr(self, f.name) for f in fields(self)}
        return {k: v for k, v in out.items() if v is not None}

    def validate(self) -> list[str]:
        errors: list[str] = []
        for name in REQUIRED_FIELDS:
            val = getattr(self, name, None)
            if val is None or str(val).strip() == "":
                errors.append(f"missing required field: {name}")
        if self.diversity_spec_version != DIVERSITY_SPEC_VERSION:
            errors.append(
                f"diversity_spec_version {self.diversity_spec_version!r} != {DIVERSITY_SPEC_VERSION}"
            )
        return errors


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit_hash(repo: Path) -> tuple[str, str | None]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty_hash = hashlib.sha256(dirty.encode()).hexdigest()[:16] if dirty else None
        return commit, dirty_hash
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown", None


def build_prep_manifest(
    *,
    engine_bin: Path,
    corpus_generation_id: str,
    generation_config: dict[str, Any],
    label_config: dict[str, Any],
    repo: Path | None = None,
) -> CorpusSemanticManifest:
    repo = repo or Path(__file__).resolve().parents[1]
    commit, dirty = git_commit_hash(repo)
    gen_hash = hashlib.sha256(
        json.dumps(generation_config, sort_keys=True).encode()
    ).hexdigest()[:16]
    label_hash = hashlib.sha256(
        json.dumps(label_config, sort_keys=True).encode()
    ).hexdigest()[:16]
    return CorpusSemanticManifest(
        engine_semantic_version="engine-semantics-prep",
        engine_binary_hash=sha256_file(engine_bin) if engine_bin.is_file() else "missing",
        nnue_feature_schema_version="nnue-fv-v1",
        move_encoding_version="move-alg-v1",
        evaluation_semantics_version="eval-semantics-prep",
        solver_oracle_version="solver-oracle-prep",
        search_configuration_version="search-cfg-prep",
        diversity_spec_version=DIVERSITY_SPEC_VERSION,
        generation_configuration_hash=gen_hash,
        label_configuration_hash=label_hash,
        source_commit_hash=commit,
        generation_timestamp=datetime.now(timezone.utc).isoformat(),
        corpus_generation_id=corpus_generation_id,
        dirty_tree_hash=dirty,
    )


def reject_incompatible_manifest(
    manifest: CorpusSemanticManifest,
    expected: CorpusSemanticManifest,
) -> None:
    errors: list[str] = []
    for name in REQUIRED_FIELDS:
        if getattr(manifest, name) != getattr(expected, name):
            errors.append(
                f"semantic mismatch on {name}: {getattr(manifest, name)!r} != {getattr(expected, name)!r}"
            )
    if errors:
        raise ValueError("; ".join(errors))


def load_manifest(path: Path) -> CorpusSemanticManifest:
    data = json.loads(path.read_text(encoding="utf-8"))
    return CorpusSemanticManifest(**{k: data[k] for k in REQUIRED_FIELDS if k in data})
