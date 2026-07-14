"""Launch gate for future real DIVERSITY_SPEC_V1 corpus generation."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from corpus_semantic_manifest import CorpusSemanticManifest
from diversity.eval_denylist import default_evaluation_registry
from diversity.lanes import DIVERSITY_SPEC_VERSION
from engine_semantic_contract import EngineSemanticsContract
from prep_guard import prep_only_enabled

_REPO = Path(__file__).resolve().parents[1]
APPROVAL_FILE = _REPO / "training" / "APPROVE_GENERATION.json"

APPROVAL_HASH_FIELDS = (
    "corpus_generation_id",
    "engine_semantics_hash",
    "generation_config_hash",
    "label_config_hash",
    "diversity_spec_version",
)


@dataclass(frozen=True)
class LaunchGateResult:
    allowed: bool
    blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"allowed": self.allowed, "blockers": list(self.blockers)}


def _approval_hashes(approval: dict[str, Any]) -> dict[str, str]:
    return {k: str(approval.get(k, "")) for k in APPROVAL_HASH_FIELDS}


def validate_launch_gate(
    *,
    corpus_generation_id: str | None,
    manifest: CorpusSemanticManifest | None,
    engine_contract: EngineSemanticsContract | None = None,
    generation_config_hash: str | None = None,
    label_config_hash: str | None = None,
    approval_path: Path = APPROVAL_FILE,
) -> LaunchGateResult:
    blockers: list[str] = []
    if prep_only_enabled():
        blockers.append("TRAINING_PREP_ONLY=1")
    if not corpus_generation_id:
        blockers.append("missing explicit corpus_generation_id")
    if manifest is None:
        blockers.append("missing semantic manifest")
    else:
        blockers.extend(manifest.validate())
        if manifest.engine_semantic_version.endswith("-prep"):
            blockers.append("engine semantic version still in prep mode")
    if engine_contract is not None:
        blockers.extend(engine_contract.validate())
    if not default_evaluation_registry():
        blockers.append("evaluation leakage registry not loaded")
    import os

    frozen = os.environ.get("ENGINE_SEMANTIC_VERSION_FROZEN", "").strip()
    if not frozen:
        blockers.append("ENGINE_SEMANTIC_VERSION_FROZEN not set")
    elif engine_contract and frozen != engine_contract.engine_semantic_version:
        blockers.append("ENGINE_SEMANTIC_VERSION_FROZEN mismatch")

    if not approval_path.is_file():
        blockers.append(f"missing approval file: {approval_path}")
    else:
        try:
            approval = json.loads(approval_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            blockers.append("approval file is not valid JSON")
            approval = {}
        expected = {
            "corpus_generation_id": corpus_generation_id or "",
            "engine_semantics_hash": engine_contract.semantics_hash() if engine_contract else "",
            "generation_config_hash": generation_config_hash or "",
            "label_config_hash": label_config_hash or "",
            "diversity_spec_version": DIVERSITY_SPEC_VERSION,
        }
        actual = _approval_hashes(approval)
        for field, exp in expected.items():
            if not actual.get(field):
                blockers.append(f"approval missing hash field: {field}")
            elif actual[field] != exp:
                blockers.append(f"approval stale/mismatch on {field}: {actual[field]!r} != {exp!r}")

    return LaunchGateResult(allowed=not blockers, blockers=tuple(blockers))


def assert_launch_allowed(**kwargs: Any) -> None:
    result = validate_launch_gate(**kwargs)
    if not result.allowed:
        raise SystemExit(f"launch gate blocked: {'; '.join(result.blockers)}")
