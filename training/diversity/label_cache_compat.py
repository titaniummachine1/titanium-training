"""Label-cache compatibility keys — never mark a position labeled forever."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

LABEL_CACHE_COMPAT_VERSION = "label-cache-compat-v1"


class LabelCacheLookup(str, Enum):
    HIT_COMPATIBLE = "HIT_COMPATIBLE"
    MISS_NEW_STATE = "MISS_NEW_STATE"
    MISS_SEMANTICS_CHANGED = "MISS_SEMANTICS_CHANGED"
    MISS_SEARCH_CONFIG_CHANGED = "MISS_SEARCH_CONFIG_CHANGED"
    MISS_ORACLE_CHANGED = "MISS_ORACLE_CHANGED"
    INVALID_METADATA = "INVALID_METADATA"


REQUIRED_COMPAT_FIELDS = (
    "canonical_state_key",
    "engine_semantic_hash",
    "search_configuration_hash",
    "evaluation_semantics_version",
    "score_band_version",
    "oracle_semantics_version",
    "move_encoding_version",
    "label_configuration_hash",
    "exact_label_kind",
    "side_to_move",
)


@dataclass(frozen=True)
class LabelCompatKey:
    canonical_state_key: str
    engine_semantic_hash: str
    search_configuration_hash: str
    evaluation_semantics_version: str
    score_band_version: str
    oracle_semantics_version: str
    move_encoding_version: str
    label_configuration_hash: str
    exact_label_kind: str
    side_to_move: int

    def validate(self) -> list[str]:
        errors: list[str] = []
        d = asdict(self)
        for field in REQUIRED_COMPAT_FIELDS:
            if d.get(field) in (None, ""):
                errors.append(f"missing:{field}")
        if self.side_to_move not in (0, 1):
            errors.append("invalid:side_to_move")
        return errors

    def fingerprint(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()


def lookup_label_cache(
    query: LabelCompatKey,
    stored: LabelCompatKey | None,
) -> LabelCacheLookup:
    q_err = query.validate()
    if q_err:
        return LabelCacheLookup.INVALID_METADATA
    if stored is None:
        return LabelCacheLookup.MISS_NEW_STATE
    s_err = stored.validate()
    if s_err:
        return LabelCacheLookup.INVALID_METADATA
    if query.canonical_state_key != stored.canonical_state_key:
        return LabelCacheLookup.MISS_NEW_STATE
    if query.engine_semantic_hash != stored.engine_semantic_hash:
        return LabelCacheLookup.MISS_SEMANTICS_CHANGED
    if query.evaluation_semantics_version != stored.evaluation_semantics_version:
        return LabelCacheLookup.MISS_SEMANTICS_CHANGED
    if query.search_configuration_hash != stored.search_configuration_hash:
        return LabelCacheLookup.MISS_SEARCH_CONFIG_CHANGED
    if query.oracle_semantics_version != stored.oracle_semantics_version:
        return LabelCacheLookup.MISS_ORACLE_CHANGED
    if query.fingerprint() == stored.fingerprint():
        return LabelCacheLookup.HIT_COMPATIBLE
    # Remaining mismatches (score band / move encoding / label config / exact kind / STM)
    return LabelCacheLookup.MISS_SEMANTICS_CHANGED
