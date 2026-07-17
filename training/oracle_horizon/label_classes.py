"""Strict provenance classes for oracle-horizon labels."""
from __future__ import annotations

from enum import Enum
from typing import Any


class LabelClass(str, Enum):
    EXACT_ORACLE = "EXACT_ORACLE"
    ORACLE_BACKED_MINIMAX = "ORACLE_BACKED_MINIMAX"
    ORACLE_SUPPORTED_PARTIAL = "ORACLE_SUPPORTED_PARTIAL"
    SEARCH_ONLY = "SEARCH_ONLY"


def _coerce(label_class: LabelClass | str) -> LabelClass:
    try:
        return label_class if isinstance(label_class, LabelClass) else LabelClass(str(label_class))
    except ValueError as exc:
        raise ValueError(f"unknown label class: {label_class!r}") from exc


def can_train_primary(label_class: LabelClass | str) -> bool:
    return _coerce(label_class) in {LabelClass.EXACT_ORACLE, LabelClass.ORACLE_BACKED_MINIMAX}


def sample_weight(label_class: LabelClass | str) -> float:
    return {
        LabelClass.EXACT_ORACLE: 1.0,
        LabelClass.ORACLE_BACKED_MINIMAX: 1.0,
        LabelClass.ORACLE_SUPPORTED_PARTIAL: 0.25,
        LabelClass.SEARCH_ONLY: 0.0,
    }[_coerce(label_class)]


def assert_not_fake_exact(label: Any) -> None:
    """Reject claims that search scores or partial labels are exact proof.

    A large NNUE/evaluation number is still a prediction, never a solved W/L.
    """
    if isinstance(label, dict):
        klass = label.get("label_class", label.get("proof_completeness_class"))
        exact_claim = bool(label.get("exact", label.get("is_exact", False))) or str(klass) in {
            LabelClass.EXACT_ORACLE.value,
            "EXACT",
        }
        nnue_only = bool(label.get("nnue_score_only", False)) or (
            label.get("nnue_score") is not None and not label.get("oracle_proof")
        )
    else:
        klass = label
        exact_claim = _coerce(label) in {LabelClass.SEARCH_ONLY, LabelClass.ORACLE_SUPPORTED_PARTIAL}
        nnue_only = False
    if nnue_only and exact_claim:
        raise ValueError("NNUE prediction cannot certify a solved state")
    if exact_claim and _coerce(klass) in {LabelClass.SEARCH_ONLY, LabelClass.ORACLE_SUPPORTED_PARTIAL}:
        raise ValueError(f"{_coerce(klass).value} cannot be claimed as exact")
    if nnue_only:
        raise ValueError("NNUE score alone is not exact oracle proof")
