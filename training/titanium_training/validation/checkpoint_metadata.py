"""Infer HalfPW architecture from checkpoint artifacts — never default to legacy h."""
from __future__ import annotations

import struct
from pathlib import Path

import torch

from titanium_training.training.trainer import H_HEADER_LEN, read_net_h

ARCHITECTURE_INFERENCE_ERROR = (
    "Cannot infer checkpoint architecture; refusing to default to h48"
)


class CheckpointArchitectureError(RuntimeError):
    """Raised when checkpoint width cannot be inferred safely."""


def hidden_size_from_state_dict(state_dict: dict) -> int | None:
    b1 = state_dict.get("b1")
    if b1 is None:
        return None
    return int(b1.shape[0])


def hidden_size_from_weights_bin(path: Path) -> int:
    if not path.is_file():
        raise CheckpointArchitectureError(f"weights file missing: {path}")
    return int(read_net_h(path))


def infer_hidden_size(
    *,
    weights_bin: Path | None = None,
    checkpoint: Path | None = None,
    state_dict: dict | None = None,
) -> int:
    """Resolve NET_H from header, checkpoint tensors, or explicit weights."""
    candidates: list[tuple[str, int]] = []

    if weights_bin is not None and weights_bin.is_file():
        candidates.append(("weights_header", hidden_size_from_weights_bin(weights_bin)))

    if state_dict is None and checkpoint is not None and checkpoint.is_file():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if isinstance(payload, dict) and "model" in payload:
            state_dict = payload["model"]

    if state_dict is not None:
        h = hidden_size_from_state_dict(state_dict)
        if h is not None:
            candidates.append(("state_dict", h))

    if not candidates:
        raise CheckpointArchitectureError(ARCHITECTURE_INFERENCE_ERROR)

    values = {h for _, h in candidates}
    if len(values) != 1:
        raise CheckpointArchitectureError(
            f"Conflicting architecture inference: {dict(candidates)}"
        )
    return candidates[0][1]


def architecture_bin_for_checkpoint(
    *,
    candidate_bin: Path,
    checkpoint: Path | None = None,
    parent_bin: Path | None = None,
) -> Path:
    """Pick a weights.bin whose NET_H header matches the checkpoint tensors."""
    for path in (candidate_bin, parent_bin):
        if path is not None and path.is_file():
            try:
                infer_hidden_size(weights_bin=path, checkpoint=checkpoint)
                return path
            except CheckpointArchitectureError:
                continue
    raise CheckpointArchitectureError(
        ARCHITECTURE_INFERENCE_ERROR
    )
