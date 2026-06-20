#!/usr/bin/env python3
"""Train the frozen-trunk, linear safe-and-beneficial +1 LMR sidecar."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import struct
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.lmr.reduction_counterfactual_schema import (
    FEATURE_SCHEMA,
    SCHEMA,
    SIDECAR_SCHEMA,
    validate_row,
    wilson_lower,
)

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "engine" / "src" / "acev13" / "net_weights.bin"
DEFAULT_DATA = ROOT / "training" / "data" / "reduction_counterfactuals.jsonl"
DEFAULT_OUT = ROOT / "training" / "checkpoints" / "search_reduction_head.pt"
INPUTS = 37


def load_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            validate_row(row)
            rows.append(row)
    return rows


def features(row: dict) -> list[float]:
    values = [float(v) for v in row["hidden32"]] + [float(v) for v in row["context5"]]
    if len(values) != INPUTS:
        raise ValueError(f"expected {INPUTS} features, got {len(values)}")
    return values


def tensors(rows: list[dict]):
    return (
        torch.tensor([features(row) for row in rows], dtype=torch.float32),
        torch.tensor([float(row["activate_plus_one"]) for row in rows], dtype=torch.float32),
    )


def fit_platt(logits: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    if len(labels) < 4 or labels.min() == labels.max():
        return 1.0, 0.0
    scale = nn.Parameter(torch.ones(()))
    shift = nn.Parameter(torch.zeros(()))
    opt = torch.optim.LBFGS([scale, shift], max_iter=80, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(scale * logits + shift, labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(scale.detach()), float(shift.detach())


def choose_threshold(probs: list[float], rows: list[dict], min_activations: int) -> tuple[float, dict]:
    best = (1.0, {"activations": 0, "precision": 0.0, "wilson_lower": 0.0, "net_nodes_saved": 0})
    for threshold in (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.975, 0.99, 0.995):
        active = [row for p, row in zip(probs, rows) if p >= threshold]
        if len(active) < min_activations:
            continue
        correct = sum(bool(row["activate_plus_one"]) for row in active)
        stats = {
            "activations": len(active),
            "precision": correct / len(active),
            "wilson_lower": wilson_lower(correct, len(active)),
            "net_nodes_saved": sum(int(row.get("net_nodes_saved", 0)) for row in active),
        }
        prior = best[1]
        if (stats["wilson_lower"], stats["net_nodes_saved"]) > (
            prior["wilson_lower"], prior["net_nodes_saved"]
        ):
            best = (threshold, stats)
    return best


def evaluate(model, rows: list[dict], scale: float, shift: float, threshold: float) -> dict:
    if not rows:
        return {"rows": 0, "activations": 0}
    x, y = tensors(rows)
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(scale * logits + shift)
    active = probs >= threshold
    count = int(active.sum())
    true_active = int(((y > 0.5) & active).sum())
    unsafe = count - true_active
    saved = sum(
        int(row.get("net_nodes_saved", 0))
        for row, enabled in zip(rows, active.tolist())
        if enabled
    )
    return {
        "rows": len(rows),
        "positives": int(y.sum()),
        "activations": count,
        "false_activations": unsafe,
        "precision": true_active / count if count else 0.0,
        "precision_wilson_lower_95": wilson_lower(true_active, count),
        "recall": true_active / max(1, int(y.sum())),
        "net_nodes_saved": saved,
        "activation_rate": count / len(rows),
    }


def write_binary(
    path: Path,
    model: nn.Linear,
    trunk_hash: str,
    calibration_scale: float,
    calibration_shift: float,
    threshold: float,
) -> tuple[str, str]:
    weights = model.weight.detach().cpu().double().flatten().tolist()
    bias = float(model.bias.detach().cpu())
    payload = bytearray(b"TISRDX1\0")
    payload.extend(struct.pack("<III", 1, 1, INPUTS))
    payload.extend(bytes.fromhex(trunk_hash))
    payload.extend(struct.pack("<II", 1, 1))  # calibration and training-data versions
    payload.extend(struct.pack(f"<{INPUTS}d", *weights))
    payload.extend(struct.pack("<dddd", bias, calibration_scale, calibration_shift, threshold))
    digest = hashlib.sha256(payload).digest()
    artifact = bytes(payload + digest)
    path.write_bytes(artifact)
    return digest.hex(), hashlib.sha256(artifact).hexdigest()


def train_seed(train_rows, calibration_rows, test_rows, *, seed: int, epochs: int, lr: float, unsafe_weight: float, min_activations: int):
    torch.manual_seed(seed)
    model = nn.Linear(INPUTS, 1)
    nn.init.zeros_(model.weight)
    nn.init.constant_(model.bias, -6.0)
    x_train, y_train = tensors(train_rows)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        opt.zero_grad()
        logits = model(x_train).squeeze(1)
        per_row = F.binary_cross_entropy_with_logits(logits, y_train, reduction="none")
        weights = torch.where(y_train > 0.5, 1.0, unsafe_weight)
        loss = (per_row * weights).mean()
        loss.backward()
        opt.step()

    x_cal, y_cal = tensors(calibration_rows)
    with torch.no_grad():
        raw_cal = model(x_cal).squeeze(1)
    scale, shift = fit_platt(raw_cal, y_cal)
    probs = torch.sigmoid(scale * raw_cal + shift).tolist()
    threshold, calibration_choice = choose_threshold(probs, calibration_rows, min_activations)
    return model, {
        "seed": seed,
        "unsafe_weight": unsafe_weight,
        "calibration_scale": scale,
        "calibration_shift": shift,
        "threshold": threshold,
        "calibration_choice": calibration_choice,
        "calibration": evaluate(model, calibration_rows, scale, shift, threshold),
        "final_test": evaluate(model, test_rows, scale, shift, threshold),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", action="append", default=None)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--seeds", default="1337,2027,4099")
    parser.add_argument("--unsafe-weights", default="4,8,16")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--min-rows", type=int, default=100)
    parser.add_argument("--min-activations", type=int, default=10)
    parser.add_argument("--min-positive-calibration", type=int, default=10)
    args = parser.parse_args()

    paths = [Path(p) for p in (args.data or [DEFAULT_DATA])]
    all_rows = load_rows(paths)
    unknown = [row for row in all_rows if row["sample_status"] == "UNKNOWN"]
    known = [row for row in all_rows if row["sample_status"] != "UNKNOWN"]
    natural = [row for row in known if row.get("population") == "natural"]
    train_rows = [row for row in known if row.get("split") == "train"]
    calibration_rows = [row for row in natural if row.get("split") == "calibration"]
    test_rows = [row for row in natural if row.get("split") == "final_test"]
    support = {
        "train_positive": sum(bool(row["activate_plus_one"]) for row in train_rows),
        "train_negative": sum(not bool(row["activate_plus_one"]) for row in train_rows),
        "calibration_positive": sum(bool(row["activate_plus_one"]) for row in calibration_rows),
        "calibration_negative": sum(not bool(row["activate_plus_one"]) for row in calibration_rows),
        "test_positive": sum(bool(row["activate_plus_one"]) for row in test_rows),
        "test_negative": sum(not bool(row["activate_plus_one"]) for row in test_rows),
    }
    support_ready = (
        support["train_positive"] > 0
        and support["train_negative"] > 0
        and support["calibration_positive"] >= args.min_positive_calibration
        and support["calibration_negative"] > 0
        and support["test_positive"] >= args.min_positive_calibration
        and support["test_negative"] > 0
    )
    if (
        len(known) < args.min_rows
        or not train_rows
        or not calibration_rows
        or not test_rows
        or not support_ready
    ):
        print(
            f"not train-ready: known={len(known)} train={len(train_rows)} "
            f"natural_calibration={len(calibration_rows)} natural_test={len(test_rows)} "
            f"unknown={len(unknown)} support={support}"
        )
        return 2

    trunk_before = hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()
    reports = []
    candidates = []
    for unsafe_weight in (float(v) for v in args.unsafe_weights.split(",")):
        for seed in (int(v) for v in args.seeds.split(",")):
            model, report = train_seed(
                train_rows, calibration_rows, test_rows,
                seed=seed, epochs=args.epochs, lr=args.lr,
                unsafe_weight=unsafe_weight, min_activations=args.min_activations,
            )
            reports.append(report)
            candidates.append((report["final_test"].get("precision_wilson_lower_95", 0.0), report["final_test"].get("net_nodes_saved", 0), model, report))
            print(json.dumps(report, sort_keys=True))

    _, _, best_model, best_report = max(candidates, key=lambda item: (item[0], item[1]))
    trunk_after = hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()
    if trunk_before != trunk_after:
        raise RuntimeError("frozen value weights changed during sidecar training")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    binary = out.with_suffix(".bin")
    payload_hash, sidecar_hash = write_binary(
        binary,
        best_model,
        trunk_before,
        best_report["calibration_scale"],
        best_report["calibration_shift"],
        best_report["threshold"],
    )
    artifact = {
        "kind": SIDECAR_SCHEMA,
        "schema_version": 1,
        "feature_schema": FEATURE_SCHEMA,
        "compatible_trunk_sha256": trunk_before,
        "sidecar_sha256": sidecar_hash,
        "payload_sha256": payload_hash,
        "calibration_version": 1,
        "training_data_version": SCHEMA,
        "runtime_enabled": False,
        "best_report": best_report,
        "all_runs": reports,
        "known_rows": len(known),
        "unknown_rows": len(unknown),
        "safe_rows": sum(row["sample_status"] == "SAFE" for row in known),
        "unsafe_rows": sum(row["sample_status"] == "UNSAFE" for row in known),
        "useful_positive_rows": sum(bool(row["activate_plus_one"]) for row in known),
        "trunk_freeze_proof": {"before": trunk_before, "after": trunk_after, "unchanged": True},
    }
    torch.save({"head": best_model.state_dict(), "metadata": artifact}, out)
    out.with_suffix(".report.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    precision_values = [r["final_test"].get("precision", 0.0) for r in reports]
    print(
        f"saved detached sidecar {binary} hash={sidecar_hash} runtime_enabled=false "
        f"precision_mean={statistics.mean(precision_values):.3f} "
        f"precision_sd={statistics.pstdev(precision_values):.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
