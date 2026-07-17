"""Training epoch diagnostics helpers."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch


def assert_finite_tensor(name: str, tensor: torch.Tensor) -> None:
    """Raise RuntimeError if tensor contains NaN or Inf."""
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"non-finite values in {name}")


def grad_norm_stats(model: torch.nn.Module) -> dict[str, float]:
    """Return mean and max L2 gradient norms across parameters."""
    norms = [
        p.grad.detach().norm().item()
        for p in model.parameters()
        if p.grad is not None
    ]
    if not norms:
        return {"mean": 0.0, "max": 0.0}
    return {"mean": sum(norms) / len(norms), "max": max(norms)}


def param_norm(model: torch.nn.Module) -> float:
    """Return total L2 norm of all model parameters."""
    total = sum(p.data.norm().item() ** 2 for p in model.parameters())
    return math.sqrt(total)


def update_norm(model: torch.nn.Module, prev_params: dict[str, torch.Tensor]) -> float:
    """Return L2 norm of parameter update since prev_params snapshot."""
    total = 0.0
    for name, p in model.named_parameters():
        if name in prev_params:
            total += (p.data - prev_params[name]).norm().item() ** 2
    return math.sqrt(total)


def prediction_label_stats(
    preds: list[float], labels: list[float]
) -> dict[str, float]:
    """Return basic statistics about model predictions vs labels."""
    if not preds or not labels:
        return {}
    n = len(preds)
    pred_mean = sum(preds) / n
    label_mean = sum(labels) / n
    mae = sum(abs(p - l) for p, l in zip(preds, labels)) / n
    return {
        "pred_mean": pred_mean,
        "label_mean": label_mean,
        "mae": mae,
        "n_samples": n,
    }


def write_epoch_diagnostics(path: Path | str, diag: dict[str, Any]) -> None:
    """Write epoch diagnostics dict as JSON to path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(diag, indent=2, default=str) + "\n", encoding="utf-8")


__all__ = [
    "assert_finite_tensor",
    "grad_norm_stats",
    "param_norm",
    "prediction_label_stats",
    "update_norm",
    "write_epoch_diagnostics",
]
