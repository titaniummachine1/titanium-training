"""Promotion metadata for accepted checkpoint chain entries."""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROMOTION_RECORD_VERSION = "promotion-record-v1"


@dataclass(frozen=True)
class PromotionRecord:
    """Reproducibility metadata stored with each accept/quarantine decision."""

    promotion_record_version: str
    epoch_id: int
    parent_accepted_epoch: int | None
    grandparent_validation_epoch: int | None
    candidate_weights_sha256: str
    parent_weights_sha256: str | None
    engine_semantic_hash: str
    training_corpus_id: str | None
    corpus_manifest_hash: str | None
    diversity_certificate_status: str | None
    diversity_certificate_hash: str | None
    training_config_hash: str | None
    validation_config_hash: str | None
    match_results: dict[str, Any]
    score_thresholds: dict[str, Any]
    sign_test_result: dict[str, Any] | None
    decision: str
    accepted_at: str
    source_commit: str
    dirty_tree_hash: str | None
    code_promotion_separate: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def record_hash(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.promotion_record_version:
            errors.append("missing promotion_record_version")
        if len(self.candidate_weights_sha256 or "") != 64:
            errors.append("candidate_weights_sha256 must be 64 hex chars")
        if not self.engine_semantic_hash:
            errors.append("missing engine_semantic_hash")
        if not self.decision:
            errors.append("missing decision")
        return errors


def _git_head() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _git_dirty_hash() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        if not out.strip():
            return None
        return hashlib.sha256(out.encode()).hexdigest()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_promotion_record(
    *,
    epoch_id: int,
    candidate_weights_sha256: str,
    validation: dict[str, Any],
    parent_accepted_epoch: int | None = None,
    grandparent_validation_epoch: int | None = None,
    parent_weights_sha256: str | None = None,
    engine_semantic_hash: str = "prep",
    training_corpus_id: str | None = None,
    corpus_manifest_hash: str | None = None,
    diversity_certificate_status: str | None = None,
    diversity_certificate_hash: str | None = None,
    training_config_hash: str | None = None,
    validation_config_hash: str | None = None,
    decision: str = "accepted",
) -> PromotionRecord:
    match_prev = validation.get("match_vs_previous") or {}
    evidence = match_prev.get("promotion_evidence") or {}
    return PromotionRecord(
        promotion_record_version=PROMOTION_RECORD_VERSION,
        epoch_id=epoch_id,
        parent_accepted_epoch=parent_accepted_epoch,
        grandparent_validation_epoch=grandparent_validation_epoch,
        candidate_weights_sha256=candidate_weights_sha256,
        parent_weights_sha256=parent_weights_sha256,
        engine_semantic_hash=engine_semantic_hash,
        training_corpus_id=training_corpus_id,
        corpus_manifest_hash=corpus_manifest_hash,
        diversity_certificate_status=diversity_certificate_status,
        diversity_certificate_hash=diversity_certificate_hash,
        training_config_hash=training_config_hash,
        validation_config_hash=validation_config_hash,
        match_results={
            "match_vs_previous": match_prev,
            "match_vs_grandparent": validation.get("match_vs_grandparent"),
        },
        score_thresholds={
            "prior_epoch_min_score": match_prev.get("min_score"),
            "prior_epoch_min_games": match_prev.get("min_games"),
        },
        sign_test_result=evidence if evidence else None,
        decision=decision,
        accepted_at=datetime.now(timezone.utc).isoformat(),
        source_commit=_git_head(),
        dirty_tree_hash=_git_dirty_hash(),
    )


def build_synthetic_validation_report(
    *,
    candidate_sha256: str = "d" * 64,
    parent_sha256: str | None = "c" * 64,
    passed: bool = True,
) -> dict[str, Any]:
    """Fixture validation dict for prep-mode promotion record rehearsal."""
    evidence = {
        "passed": passed,
        "sign_test_alpha": 0.05,
        "sign_test_p_value": 0.01 if passed else 0.5,
        "min_decisive_pairs": 20,
        "decisive_pairs": 100,
        "wins": 55 if passed else 45,
    }
    return {
        "passed": passed,
        "candidate_sha256": candidate_sha256,
        "synthetic_only": True,
        "match_vs_previous": {
            "games": 200,
            "score": 0.55 if passed else 0.48,
            "passed": passed,
            "promotion_evidence": evidence,
            "min_score": 0.50,
            "min_games": 200,
        },
        "reject_reason": None if passed else "synthetic_validation_failed",
    }


def fixture_promotion_record(*, passed: bool = True) -> PromotionRecord:
    validation = build_synthetic_validation_report(passed=passed)
    return build_promotion_record(
        epoch_id=1,
        candidate_weights_sha256="d" * 64,
        parent_weights_sha256="c" * 64,
        parent_accepted_epoch=0,
        validation=validation,
        decision="accepted" if passed else "quarantined",
    )
