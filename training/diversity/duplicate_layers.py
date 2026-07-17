"""Three distinct duplicate layers — do not collapse into one vague 'seen' flag."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class DuplicateLayer(str, Enum):
    GENERATION = "generation"
    LABEL = "label"
    FINAL_CORPUS = "final_corpus"


@dataclass
class GenerationDedup:
    """Same canonical state was already produced recently."""

    recent_keys: set[str] = field(default_factory=set)
    allow_controlled_revisit: bool = False

    def observe(self, canonical_key: str, *, force_revisit: bool = False) -> bool:
        """Return True if generation should proceed; False if duplicate blocked."""
        if canonical_key in self.recent_keys:
            if force_revisit and self.allow_controlled_revisit:
                return True
            return False
        self.recent_keys.add(canonical_key)
        return True


@dataclass
class LabelDedup:
    """Reuse label only when compatibility fields match (see label_cache_compat)."""

    labeled_fingerprints: set[str] = field(default_factory=set)

    def already_labeled_compatible(self, compat_fingerprint: str) -> bool:
        return compat_fingerprint in self.labeled_fingerprints

    def remember(self, compat_fingerprint: str) -> None:
        self.labeled_fingerprints.add(compat_fingerprint)


@dataclass
class FinalCorpusDedup:
    """Zero duplicate canonical states in the finalized sampled corpus."""

    keys: set[str] = field(default_factory=set)

    def accept(self, canonical_key: str) -> bool:
        if canonical_key in self.keys:
            return False
        self.keys.add(canonical_key)
        return True

    @property
    def duplicate_count_if_closed(self) -> int:
        return 0  # by construction accept() rejects dups

    def assert_zero_duplicates(self, keys: list[str]) -> None:
        if len(keys) != len(set(keys)):
            raise AssertionError("final corpus contains duplicate canonical states")
