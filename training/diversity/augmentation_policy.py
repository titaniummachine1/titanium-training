"""Augmentation policy: dynamic reflection during batching only (preferred).

Do not materialize reflected rows into the stored corpus.
"""
from __future__ import annotations

from dataclasses import dataclass

AUGMENTATION_POLICY_VERSION = "augmentation-policy-v1"
AUGMENTATION_KIND = "reflection_lr_dynamic"


@dataclass(frozen=True)
class AugmentationPolicy:
    """Preferred model: store canonical base states only; mirror at batch time."""

    kind: str = AUGMENTATION_KIND
    version: str = AUGMENTATION_POLICY_VERSION
    materialize_reflected_rows: bool = False
    dynamic_during_batching: bool = True
    track_base_state_id: bool = True

    def __post_init__(self) -> None:
        if self.materialize_reflected_rows and self.dynamic_during_batching:
            raise ValueError("must choose exactly one augmentation model, not both")
        if not self.dynamic_during_batching:
            raise ValueError("preferred policy requires dynamic_during_batching=True")


PREFERRED_AUGMENTATION = AugmentationPolicy()
