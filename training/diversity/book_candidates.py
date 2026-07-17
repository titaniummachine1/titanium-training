"""Safe, review-only book candidates derived from Claustrophobia seeds.

This module deliberately has no database writer.  A candidate is not training
data and cannot be imported into a live book by accident.
"""
from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = "claustrophobia-book-candidate-v1"


class CandidateStatus(str, Enum):
    PROPOSED = "PROPOSED"
    REJECTED_UNSTABLE = "REJECTED_UNSTABLE"
    REJECTED_EVAL_LEAKAGE = "REJECTED_EVAL_LEAKAGE"
    REJECTED_ILLEGAL = "REJECTED_ILLEGAL"
    BOOK_CANDIDATE_VERIFIED = "BOOK_CANDIDATE_VERIFIED"
    LIVE_BOOK_MERGE_PENDING = "LIVE_BOOK_MERGE_PENDING"


REQUIRED_PROVENANCE = (
    "claustro_checkpoint_sha256",
    "repository_commit",
    "source_opening_id",
    "titanium_weights_sha256",
)


def _clean_root() -> Path:
    return Path(__file__).resolve().parents[1] / "external_sources" / "claustrophobia" / "eval_games" / "clean_v1"


@lru_cache(maxsize=1)
def _denylist_values() -> tuple[set[str], set[str]]:
    """Return denylisted identifiers and move/canonical representations."""
    root = _clean_root()
    ids: set[str] = {"clean_v1", "claustrophobia_clean_v1"}
    keys: set[str] = set()
    deny = root / "EVAL_DENYLIST_KEYS.json"
    if deny.is_file():
        data = json.loads(deny.read_text(encoding="utf-8"))
        ids.update(str(x) for x in data.get("opening_ids", []) + data.get("lineage_ids", []))
        keys.update(str(x) for x in data.get("canonical_keys", []) + data.get("opening_hashes", []))
    for path in root.rglob("openings_used.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for opening in data.get("openings", data if isinstance(data, list) else []):
            if isinstance(opening, dict):
                if opening.get("opening_id") is not None:
                    ids.add(str(opening["opening_id"]))
                moves = tuple(str(x) for x in opening.get("moves", []))
                if moves:
                    keys.update(_move_keys(moves))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        keys.update({digest, f"openings_used_sha256:{digest}", f"clean_v1:openings_used_sha256:{digest}"})
    return ids, keys


def _move_keys(moves: Iterable[str]) -> set[str]:
    values = tuple(str(x) for x in moves)
    encoded = json.dumps(list(values), separators=(",", ":"))
    joined = " ".join(values)
    return {encoded, joined, "|".join(values), hashlib.sha256(encoded.encode()).hexdigest()}


def is_clean_v1_excluded(
    opening_id: str | None = None,
    moves: Iterable[str] | None = None,
    canonical_key: str | None = None,
) -> bool:
    """Return true if an opening, move prefix, or canonical key is eval-only."""
    if moves is None and opening_id is not None and not isinstance(opening_id, str):
        moves, opening_id = opening_id, None
    ids, keys = _denylist_values()
    if opening_id and str(opening_id) in ids:
        return True
    if canonical_key and str(canonical_key) in keys:
        return True
    if moves is not None and bool(_move_keys(moves) & keys):
        return True
    return False


def validate_candidate_row(row: dict[str, Any], *, raise_on_invalid: bool = False) -> dict[str, Any]:
    """Validate the contract, returning INVALID details rather than guessing."""
    provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
    aliases = {
        "claustro_checkpoint_sha256": ("claustro_checkpoint_hash",),
        "repository_commit": ("repo_commit",),
        "titanium_weights_sha256": ("titanium_weight_hash",),
    }
    placeholder_values = {"", "not-supplied", "unknown_unavailable"}
    def present(value: Any) -> bool:
        return value is not None and str(value).strip() not in placeholder_values

    missing = [
        field for field in REQUIRED_PROVENANCE
        if not (present(provenance.get(field)) or any(present(provenance.get(alias)) for alias in aliases.get(field, ())))
    ]
    required = ("prefix_moves", "proposed_move", "titanium_best", "budgets", "stability",
                "exact_label_kind", "provenance", "status")
    missing += [field for field in required if field not in row]
    if missing:
        result = {"valid": False, "status": "INVALID", "missing": sorted(set(missing))}
        if raise_on_invalid:
            raise ValueError("INVALID book candidate: missing " + ", ".join(result["missing"]))
        return result
    if row.get("training_eligible") is not False or row.get("live_book_eligible") is not False:
        result = {"valid": False, "status": "INVALID",
                  "reason": "eligibility_flags_must_remain_false_until_explicit_accept"}
        if raise_on_invalid:
            raise ValueError(result["reason"])
        return result
    if row["status"] not in {status.value for status in CandidateStatus}:
        result = {"valid": False, "status": "INVALID", "reason": "unknown_status"}
        if raise_on_invalid:
            raise ValueError(result["reason"])
        return result
    return {"valid": True, "status": row["status"], "missing": []}


def admit_to_live_book_allowed(row: dict[str, Any]) -> bool:
    """Always false: live-book admission requires a separate explicit process."""
    return False


def candidate_row(
    *,
    prefix_moves: Iterable[str],
    proposed_move: str,
    opening_id: str,
    claustro_checkpoint_sha256: str,
    repository_commit: str,
    titanium_weights_sha256: str,
    status: str = CandidateStatus.PROPOSED.value,
    titanium_best: str | None = None,
    top_alternatives: list[Any] | None = None,
    budgets: dict[str, Any] | None = None,
    stability: dict[str, Any] | None = None,
    exact_label_kind: str = "unavailable",
    canonical_key: str | None = None,
) -> dict[str, Any]:
    """Create a review row.

    ``exact_label_kind`` identifies the exact/oracle label source (or
    ``unavailable``); it is never inferred from Titanium search.  Provenance
    records the Claustrophobia checkpoint, repository commit, source opening,
    and Titanium weights. Placeholder provenance is intentionally invalid for
    verification, while fast PROPOSED manifests may still carry it.
    """
    row = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "prefix_moves": list(prefix_moves),
        "proposed_move": proposed_move,
        "titanium_best": titanium_best,
        "top_alternatives": top_alternatives or [],
        "budgets": budgets or {},
        "stability": stability or {"stable": False, "reason": "not_verified"},
        "exact_label_kind": exact_label_kind,
        "provenance": {
            "claustro_checkpoint_sha256": claustro_checkpoint_sha256,
            "repository_commit": repository_commit,
            "source_opening_id": opening_id,
            "titanium_weights_sha256": titanium_weights_sha256,
        },
        "canonical_key": canonical_key,
        "training_eligible": False,
        "live_book_eligible": False,
    }
    return row
