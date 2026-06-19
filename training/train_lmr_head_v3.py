#!/usr/bin/env python3
"""LMR-head trainer — Phase 3 (separate from value-network training).

Architecture
------------
Three primary models, all using DETACHED hidden32 from a FROZEN value trunk:

  P       Linear(32, 1)                 — position only
  PL      Linear(37, 1)                 — position + validated LMR context5
  PL-NL8  Linear(37, 8) -> ReLU -> Linear(8, 1)  — tiny interaction model

The LMR loss NEVER backpropagates into net_weights.bin.
The trunk SHA-256 is bound to every artifact and to every output row.

Execution order
---------------
  A. Load data, verify trunk hash matches every row, verify group leakage
  B. Check sealed holdout was never opened before manifest freeze
  C. Three-seed narrowing sweep  (P / PL / PL-NL8 × neg_w × unsafe_w)
  D. Freeze promising families
  E. Ten-seed stability comparison on frozen families
  F. Feature-ablation leave-one-out from best context-bearing model
  G. Compare against handcrafted baselines (calibration only, then holdout)
  H. Freeze experiment manifest
  I. Open fresh final holdout ONCE
  J. Shadow validation (parity checks only; runtime activation OFF)
  K. Write GO / NO-GO report

Usage
-----
python training/train_lmr_head_v3.py \\
    --natural training/data/lmr_phase3/natural.jsonl \\
    --hard-negatives training/data/lmr_phase3/hard_negatives.jsonl \\
    --out-dir training/checkpoints/lmr_v3 \\
    --phase narrowing   # or stability | manifest | holdout | shadow

On first run use --phase narrowing; subsequent phases use their own flags.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import math
import random
import statistics
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any, NamedTuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))

import torch
import torch.nn as nn
import torch.nn.functional as F

from move_codec import unpack_moves
from reduction_counterfactual_schema import (
    FEATURE_SCHEMA_V2,
    SCHEMA,
    rank_percentile,
    validate_row,
    wilson_lower,
)

WEIGHTS = ROOT / "engine" / "src" / "acev13" / "net_weights.bin"

# Cost of one LMR-head inference in node-equivalents (583 ns / 833 ns-per-node)
INFERENCE_COST_NODES = 0.7
# Asymmetric loss weights — initial sweep values
NEG_WEIGHTS  = [1.0, 2.0, 5.0]
UNSAFE_WEIGHTS = [20.0, 50.0, 100.0]
SEEDS_NARROW = [42, 137, 271]
SEEDS_STABILITY = [42, 137, 271, 512, 1337, 2027, 4099, 8191, 16381, 65537]

THRESHOLD_GRID = [
    0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35,
    0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 0.95, 0.99,
]

# Binary artifact magic (extended from sidecar_v1 to support variable-dim models)
MAGIC_V3 = b"TILMR3\0\0"


# ═══════════════════════════════════════════════════════════════════════════════
# Feature provenance table
# ═══════════════════════════════════════════════════════════════════════════════

FEATURE_PROVENANCE = [
    # name, rust_location, pre_search, already_computed, runtime_cost,
    # normalisation, used_by_native_lmr, group
    {
        "feature": "hidden32[0..32]",
        "rust_location": "engine/src/acev13/search.rs :: halfpw_acc.output() after make_move()",
        "pre_search": True,
        "already_computed": True,
        "runtime_cost": "incremental accumulator O(changed_planes)",
        "normalisation": "tanh-activated net output, range ~[-1,1]",
        "used_by_native_lmr": False,
        "hard_gate_or_soft": "soft (model input)",
        "leakage_risk": "none — computed before ab() is called",
        "group": "P",
        "nonzero_rate_note": "always nonzero (weight-initialized net)",
    },
    {
        "feature": "remaining_depth",
        "rust_location": "engine/src/acev13/search.rs :: depth arg to ab(), normalised (depth-1)/30",
        "pre_search": True,
        "already_computed": True,
        "runtime_cost": "free",
        "normalisation": "clamp(0, (depth-1)/30, 1)",
        "used_by_native_lmr": True,
        "hard_gate_or_soft": "hard gate (LMR only fires at depth>=3)",
        "leakage_risk": "none",
        "group": "L",
        "nonzero_rate_note": "always nonzero at LMR-eligible depths",
    },
    {
        "feature": "move_index",
        "rust_location": "engine/src/acev13/search.rs :: loop counter i, normalised i/128",
        "pre_search": True,
        "already_computed": True,
        "runtime_cost": "free",
        "normalisation": "clamp(0, i/128, 1)",
        "used_by_native_lmr": True,
        "hard_gate_or_soft": "hard gate (LMR only fires at i>=4)",
        "leakage_risk": "none",
        "group": "L",
        "nonzero_rate_note": "i>=4 always; always nonzero",
    },
    {
        "feature": "base_reduction",
        "rust_location": "engine/src/acev13/search.rs :: ace_graduated_lmr_reduction(i, depth)",
        "pre_search": True,
        "already_computed": True,
        "runtime_cost": "one comparison, free",
        "normalisation": "red/4, range [0, 0.75] for red in {0,1,2,3}",
        "used_by_native_lmr": True,
        "hard_gate_or_soft": "soft input (reduction tier)",
        "leakage_risk": "none",
        "group": "L",
        "nonzero_rate_note": "red>=1 at LMR threshold; always nonzero",
    },
    {
        "feature": "is_horizontal",
        "rust_location": "engine/src/acev13/search.rs :: low bit of move encoding",
        "pre_search": True,
        "already_computed": True,
        "runtime_cost": "free",
        "normalisation": "binary 0/1",
        "used_by_native_lmr": False,
        "hard_gate_or_soft": "soft input",
        "leakage_risk": "none",
        "group": "L",
        "nonzero_rate_note": "~50% of wall moves",
    },
    {
        "feature": "is_vertical",
        "rust_location": "engine/src/acev13/search.rs :: complement of is_horizontal",
        "pre_search": True,
        "already_computed": True,
        "runtime_cost": "free",
        "normalisation": "binary 0/1",
        "used_by_native_lmr": False,
        "hard_gate_or_soft": "soft input",
        "leakage_risk": "none",
        "group": "L",
        "nonzero_rate_note": "~50% of wall moves",
    },
    {
        "feature": "history_score",
        "rust_location": "engine/src/acev13/search.rs :: history_tbl[m] (already read in order_moves)",
        "pre_search": True,
        "already_computed": True,
        "runtime_cost": "already paid by move ordering",
        "normalisation": "clamp(0, (raw+10000)/20000, 1)",
        "used_by_native_lmr": False,
        "hard_gate_or_soft": "soft input",
        "leakage_risk": "none",
        "group": "O",
        "nonzero_rate_note": "~6% nonzero in Stage-2 data (early positions); grows with game depth",
    },
    {
        "feature": "rank_percentile",
        "rust_location": "engine/src/acev13/search.rs :: i / max(n-1,1) where n=total_legal_moves",
        "pre_search": True,
        "already_computed": True,
        "runtime_cost": "one division, free (n already computed by gen_moves)",
        "normalisation": "range [0, 1]",
        "used_by_native_lmr": False,
        "hard_gate_or_soft": "soft input",
        "leakage_risk": "none",
        "group": "B",
        "nonzero_rate_note": "nonzero whenever i>0",
    },
    # Forbidden inputs — listed for completeness
    {
        "feature": "FORBIDDEN: returned score / bound",
        "leakage_risk": "post-search leakage — never use",
        "group": "FORBIDDEN",
    },
    {
        "feature": "FORBIDDEN: verification_triggered",
        "leakage_risk": "post-search leakage — never use",
        "group": "FORBIDDEN",
    },
    {
        "feature": "FORBIDDEN: baseline_nodes / counterfactual_nodes",
        "leakage_risk": "post-search leakage — never use",
        "group": "FORBIDDEN",
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading and validation
# ═══════════════════════════════════════════════════════════════════════════════

def load_rows(path: Path) -> list[dict]:
    rows = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        row = json.loads(ln)
        if row.get("schema") == SCHEMA:
            validate_row(row)
        rows.append(row)
    return rows


def verify_trunk_binding(rows: list[dict], trunk_sha: str, label: str) -> list[str]:
    errors: list[str] = []
    for i, row in enumerate(rows):
        row_sha = row.get("trunk_sha256")
        if row_sha and row_sha != trunk_sha:
            errors.append(f"[{label}] trunk hash mismatch row {i}: "
                          f"{row_sha[:16]}... != {trunk_sha[:16]}...")
    if errors:
        errors.insert(0, f"[{label}] {len(errors)} trunk mismatches (first 3 shown)")
        return errors[:4]
    return []


def check_group_leakage(nat_rows: list[dict], hn_rows: list[dict]) -> list[str]:
    """Verify that hard-negative families do not cross into held-out natural splits."""
    errors: list[str] = []
    nat_test_keys  = {r["source_game_key"] for r in nat_rows if r.get("split") == "final_test"}
    nat_cal_keys   = {r["source_game_key"] for r in nat_rows if r.get("split") == "calibration"}
    hn_keys = {r["source_game_key"] for r in hn_rows}
    leaked_test = hn_keys & nat_test_keys
    leaked_cal  = hn_keys & nat_cal_keys
    if leaked_test:
        errors.append(f"Group leakage DETECTED: {len(leaked_test)} HN families "
                      f"overlap with natural final_test — remove them from training")
    if leaked_cal:
        errors.append(f"Group leakage WARNING: {len(leaked_cal)} HN families "
                      f"overlap with natural calibration")
    return errors


def filter_hn_for_training(nat_rows: list[dict], hn_rows: list[dict]) -> list[dict]:
    """Remove HN rows whose game_key appears in any natural held-out split."""
    held_out = {r["source_game_key"] for r in nat_rows
                if r.get("split") in ("final_test", "calibration")}
    return [r for r in hn_rows if r["source_game_key"] not in held_out]


def split_natural(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    train = [r for r in rows if r.get("split") == "train" and r.get("sample_status") != "UNKNOWN"]
    cal   = [r for r in rows if r.get("split") == "calibration" and r.get("sample_status") != "UNKNOWN"]
    test  = [r for r in rows if r.get("split") == "final_test" and r.get("sample_status") != "UNKNOWN"]
    return train, cal, test


# ═══════════════════════════════════════════════════════════════════════════════
# Feature extraction
# ═══════════════════════════════════════════════════════════════════════════════

CONTEXT5_NAMES = [
    "remaining_depth", "move_index", "base_reduction", "is_horizontal", "is_vertical"
]

def get_hidden32(row: dict) -> list[float]:
    return [float(v) for v in row["hidden32"]]

def get_context5(row: dict) -> list[float]:
    return [float(v) for v in row["context5"]]

def features_P(row: dict) -> list[float]:
    return get_hidden32(row)

def features_L(row: dict) -> list[float]:
    return get_context5(row)

def features_PL(row: dict) -> list[float]:
    return get_hidden32(row) + get_context5(row)

def features_PL_ablation(row: dict, zero_indices: set[int]) -> list[float]:
    """PL with specific context5 positions zeroed for leave-one-out."""
    h = get_hidden32(row)
    c = get_context5(row)
    c = [0.0 if i in zero_indices else v for i, v in enumerate(c)]
    return h + c

def get_features(row: dict, model_name: str, ablation_zeros: set[int] | None = None) -> list[float]:
    if model_name == "P":
        return features_P(row)
    if model_name == "L":
        return features_L(row)
    if model_name in ("PL", "PL-NL8"):
        if ablation_zeros:
            return features_PL_ablation(row, ablation_zeros)
        return features_PL(row)
    raise ValueError(f"Unknown model: {model_name!r}")

def to_tensors(rows: list[dict], model_name: str, ablation_zeros: set[int] | None = None):
    x = torch.tensor([get_features(r, model_name, ablation_zeros) for r in rows],
                     dtype=torch.float32)
    y = torch.tensor([float(r.get("activate_plus_one", False)) for r in rows],
                     dtype=torch.float32)
    is_unsafe = torch.tensor(
        [float(r.get("sample_status") == "UNSAFE") for r in rows],
        dtype=torch.float32
    )
    return x, y, is_unsafe


# ═══════════════════════════════════════════════════════════════════════════════
# Models
# ═══════════════════════════════════════════════════════════════════════════════

class ModelP(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(32, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.constant_(self.linear.bias, -6.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(1)

    def input_dim(self) -> int:
        return 32


class ModelL(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.constant_(self.linear.bias, -6.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(1)

    def input_dim(self) -> int:
        return 5


class ModelPL(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(37, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.constant_(self.linear.bias, -6.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(1)

    def input_dim(self) -> int:
        return 37


class ModelPLNL8(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(37, 8)
        self.fc2 = nn.Linear(8, 1)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, -6.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x))).squeeze(1)

    def input_dim(self) -> int:
        return 37


def make_model(name: str) -> nn.Module:
    if name == "P":
        return ModelP()
    if name == "L":
        return ModelL()
    if name == "PL":
        return ModelPL()
    if name == "PL-NL8":
        return ModelPLNL8()
    raise ValueError(f"Unknown model: {name!r}")


def make_optimizer(model: nn.Module, lr: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)


# ═══════════════════════════════════════════════════════════════════════════════
# Platt scaling
# ═══════════════════════════════════════════════════════════════════════════════

def fit_platt(logits: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    if len(labels) < 4 or labels.min() == labels.max():
        return 1.0, 0.0
    scale = nn.Parameter(torch.ones(()))
    shift = nn.Parameter(torch.zeros(()))
    opt = torch.optim.LBFGS([scale, shift], max_iter=200, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(scale * logits + shift, labels)
        loss.backward()
        return loss
    opt.step(closure)
    return float(scale.detach()), float(shift.detach())


# ═══════════════════════════════════════════════════════════════════════════════
# Threshold selection — calibration only
# ═══════════════════════════════════════════════════════════════════════════════

def enumerate_thresholds(probs: list[float]) -> list[float]:
    candidates = {1.0}
    for p in probs:
        if math.isfinite(p):
            candidates.add(round(min(max(float(p), 0.0), 1.0), 12))
    return sorted(candidates)


def select_threshold(probs: list[float], rows: list[dict]) -> tuple[float, dict]:
    """Select threshold maximising net signed savings, subject to unsafe==0."""
    best_t = 1.0
    best_stats: dict = {
        "threshold": 1.0, "activations": 0, "true_positives": 0,
        "unsafe_activations": 0, "gross_nodes_saved": 0,
        "net_nodes_saved": 0.0, "activation_rate": 0.0,
        "feasible": False, "precision": 0.0, "recall": 0.0,
        "wilson_lower_95": 0.0,
        "tp_delta": 0, "safe_fp_delta": 0, "unsafe_delta": 0,
    }
    total_pos = sum(1 for r in rows if r.get("activate_plus_one"))
    for t in enumerate_thresholds(probs):
        activated = [(p, r) for p, r in zip(probs, rows) if p >= t]
        n_act = len(activated)
        n_unsafe = sum(1 for _, r in activated if r.get("sample_status") == "UNSAFE")
        if n_unsafe > 0:
            continue
        n_tp = sum(1 for _, r in activated if r.get("activate_plus_one"))
        tp_d  = sum(int(r.get("net_nodes_saved", 0)) for _, r in activated if r.get("activate_plus_one"))
        sfp_d = sum(int(r.get("net_nodes_saved", 0)) for _, r in activated
                    if r.get("sample_status") == "SAFE" and not r.get("activate_plus_one"))
        u_d   = sum(int(r.get("net_nodes_saved", 0)) for _, r in activated
                    if r.get("sample_status") == "UNSAFE")
        gross = tp_d + sfp_d + u_d
        net = gross - INFERENCE_COST_NODES * n_act
        prec = n_tp / n_act if n_act else 0.0
        rec  = n_tp / max(1, total_pos)
        stats = {
            "threshold": t, "activations": n_act, "true_positives": n_tp,
            "unsafe_activations": 0, "tp_delta": tp_d, "safe_fp_delta": sfp_d,
            "unsafe_delta": u_d, "gross_nodes_saved": gross,
            "inference_cost_nodes": round(INFERENCE_COST_NODES * n_act, 2),
            "net_nodes_saved": round(net, 2),
            "activation_rate": round(n_act / max(1, len(rows)), 4),
            "precision": round(prec, 4),
            "wilson_lower_95": round(wilson_lower(n_tp, n_act), 4),
            "recall": round(rec, 4),
            "feasible": True,
        }
        if net > 0 and net > best_stats.get("net_nodes_saved", -999):
            best_t = t
            best_stats = stats
    return best_t, best_stats


def evaluate_at_threshold(
    model: nn.Module,
    rows: list[dict],
    scale: float,
    shift: float,
    threshold: float,
    model_name: str,
    ablation_zeros: set[int] | None = None,
) -> dict:
    if not rows:
        return {"rows": 0}
    x, y, is_unsafe = to_tensors(rows, model_name, ablation_zeros)
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(scale * logits + shift)
    active = probs >= threshold
    n_act = int(active.sum())
    n_tp  = int(((y > 0.5) & active).sum())
    n_uns = int((is_unsafe.bool() & active).sum())
    active_list = active.tolist()
    tp_d  = sum(int(r.get("net_nodes_saved", 0)) for r, a in zip(rows, active_list)
                if a and r.get("activate_plus_one"))
    sfp_d = sum(int(r.get("net_nodes_saved", 0)) for r, a in zip(rows, active_list)
                if a and r.get("sample_status") == "SAFE" and not r.get("activate_plus_one"))
    u_d   = sum(int(r.get("net_nodes_saved", 0)) for r, a in zip(rows, active_list)
                if a and r.get("sample_status") == "UNSAFE")
    gross = tp_d + sfp_d + u_d
    net   = gross - INFERENCE_COST_NODES * n_act
    total_pos = int(y.sum())
    return {
        "rows": len(rows),
        "positives": total_pos,
        "activations": n_act,
        "true_positives": n_tp,
        "unsafe_activations": n_uns,
        "false_activations_safe": n_act - n_tp - n_uns,
        "precision": round(n_tp / n_act, 4) if n_act else 0.0,
        "precision_wilson_lower_95": round(wilson_lower(n_tp, n_act), 4),
        "recall": round(n_tp / max(1, total_pos), 4),
        "tp_delta": tp_d, "safe_fp_delta": sfp_d, "unsafe_delta": u_d,
        "gross_nodes_saved": gross,
        "inference_cost_nodes": round(INFERENCE_COST_NODES * n_act, 2),
        "net_nodes_saved": round(net, 2),
        "activation_rate": round(n_act / max(1, len(rows)), 4),
        "unsafe_rate": round(n_uns / max(1, n_act), 4),
    }


def full_threshold_sweep(
    model: nn.Module,
    rows: list[dict],
    scale: float,
    shift: float,
    model_name: str,
    ablation_zeros: set[int] | None = None,
) -> list[dict]:
    if not rows:
        return []
    x, y, is_unsafe = to_tensors(rows, model_name, ablation_zeros)
    with torch.no_grad():
        logits = model(x)
        probs  = torch.sigmoid(scale * logits + shift).tolist()
    total_pos = int(y.sum())
    results = []
    for t in enumerate_thresholds(probs):
        active = [p >= t for p in probs]
        n_act  = sum(active)
        n_tp   = sum(a and r.get("activate_plus_one") for a, r in zip(active, rows))
        n_uns  = sum(a and r.get("sample_status") == "UNSAFE" for a, r in zip(active, rows))
        tp_d   = sum(int(r.get("net_nodes_saved", 0)) for a, r in zip(active, rows)
                     if a and r.get("activate_plus_one"))
        sfp_d  = sum(int(r.get("net_nodes_saved", 0)) for a, r in zip(active, rows)
                     if a and r.get("sample_status") == "SAFE" and not r.get("activate_plus_one"))
        u_d    = sum(int(r.get("net_nodes_saved", 0)) for a, r in zip(active, rows)
                     if a and r.get("sample_status") == "UNSAFE")
        gross = tp_d + sfp_d + u_d
        net   = gross - INFERENCE_COST_NODES * n_act
        results.append({
            "threshold": t, "activations": n_act, "true_positives": n_tp,
            "unsafe_activations": n_uns,
            "precision": round(n_tp / n_act, 4) if n_act else 0.0,
            "wilson_lower_95": round(wilson_lower(n_tp, n_act), 4),
            "recall": round(n_tp / max(1, total_pos), 4),
            "tp_delta": tp_d, "safe_fp_delta": sfp_d, "unsafe_delta": u_d,
            "gross_nodes_saved": gross, "net_nodes_saved": round(net, 2),
            "activation_rate": round(n_act / max(1, len(rows)), 4),
        })
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════

def train_one(
    train_rows: list[dict],
    cal_rows: list[dict],
    *,
    seed: int,
    model_name: str,
    neg_weight: float,
    unsafe_weight: float,
    epochs: int,
    lr: float,
    ablation_zeros: set[int] | None = None,
) -> tuple[nn.Module, float, float, float, dict]:
    torch.manual_seed(seed)
    model = make_model(model_name)
    x_train, y_train, is_unsafe_train = to_tensors(train_rows, model_name, ablation_zeros)
    opt = make_optimizer(model, lr)
    for _ in range(epochs):
        opt.zero_grad()
        logits = model(x_train)
        per_row = F.binary_cross_entropy_with_logits(logits, y_train, reduction="none")
        w = torch.where(
            y_train > 0.5,
            torch.ones_like(y_train),
            torch.where(
                is_unsafe_train > 0.5,
                torch.full_like(y_train, unsafe_weight),
                torch.full_like(y_train, neg_weight),
            ),
        )
        (per_row * w).mean().backward()
        opt.step()
    # Calibrate on calibration set — trunk weights are NEVER updated
    x_cal, y_cal, _ = to_tensors(cal_rows, model_name, ablation_zeros)
    with torch.no_grad():
        raw_cal = model(x_cal)
    scale, shift = fit_platt(raw_cal, y_cal)
    probs_cal = torch.sigmoid(scale * raw_cal + shift).tolist()
    threshold, cal_stats = select_threshold(probs_cal, cal_rows)
    return model, scale, shift, threshold, cal_stats


def mix_train(nat_train: list[dict], hn_train: list[dict],
              ratio_nat: float, seed: int) -> list[dict]:
    """Mix natural training rows with hard-negative enrichment."""
    rng = random.Random(seed ^ 0xC0C0BABE)
    n_nat = len(nat_train)
    if ratio_nat >= 1.0 or not hn_train:
        return list(nat_train)
    n_hn_target = int(n_nat * (1.0 - ratio_nat) / max(ratio_nat, 1e-9))
    n_hn = min(n_hn_target, len(hn_train))
    hn_sample = rng.sample(hn_train, n_hn)
    combined = list(nat_train) + hn_sample
    rng.shuffle(combined)
    return combined


def select_best(results: list[dict]) -> dict:
    feasible = [r for r in results
                if r["cal_stats"].get("feasible", False)
                and r["cal_stats"].get("unsafe_activations", 1) == 0
                and r["cal_stats"].get("net_nodes_saved", 0) > 0]
    if not feasible:
        feasible = [r for r in results
                    if r["cal_stats"].get("unsafe_activations", 1) == 0]
    if not feasible:
        feasible = results
    return max(feasible,
               key=lambda r: (r["cal_stats"].get("net_nodes_saved", 0),
                               r["cal_stats"].get("wilson_lower_95", 0)))


# ═══════════════════════════════════════════════════════════════════════════════
# Handcrafted rule baselines
# ═══════════════════════════════════════════════════════════════════════════════

def rule_baseline(rows: list[dict], pred) -> dict:
    n_act   = sum(1 for r in rows if pred(r))
    n_tp    = sum(1 for r in rows if pred(r) and r.get("activate_plus_one"))
    n_uns   = sum(1 for r in rows if pred(r) and r.get("sample_status") == "UNSAFE")
    tp_d    = sum(int(r.get("net_nodes_saved", 0)) for r in rows if pred(r) and r.get("activate_plus_one"))
    sfp_d   = sum(int(r.get("net_nodes_saved", 0)) for r in rows
                  if pred(r) and r.get("sample_status") == "SAFE" and not r.get("activate_plus_one"))
    u_d     = sum(int(r.get("net_nodes_saved", 0)) for r in rows if pred(r) and r.get("sample_status") == "UNSAFE")
    gross   = tp_d + sfp_d + u_d
    net     = gross - INFERENCE_COST_NODES * n_act
    prec    = n_tp / n_act if n_act else 0.0
    return {
        "activations": n_act, "true_positives": n_tp, "unsafe": n_uns,
        "tp_delta": tp_d, "safe_fp_delta": sfp_d, "unsafe_delta": u_d,
        "gross": gross, "net": round(net, 2),
        "precision": round(prec, 4),
        "wilson_lower_95": round(wilson_lower(n_tp, n_act), 4),
    }

HANDCRAFTED_RULES: dict[str, Any] = {
    "native_lmr_unchanged": lambda r: False,  # 0 activations = reference
    "always_plus1":         lambda r: True,
    "base_reduction_ge2":   lambda r: int(r.get("base_reduction", 0)) >= 2,
    "move_index_ge32":      lambda r: int(r.get("move_index", 0)) >= 32,
    "depth_ge6_mi_ge12":    lambda r: int(r.get("depth", 0)) >= 6 and int(r.get("move_index", 0)) >= 12,
    "rank_pct_ge50":        lambda r: (r.get("rank_percentile") or 0.0) >= 0.5,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Binary artifact serialisation
# ═══════════════════════════════════════════════════════════════════════════════

def write_artifact_v3(
    path: Path,
    model: nn.Module,
    model_name: str,
    trunk_hash: str,
    scale: float,
    shift: float,
    threshold: float,
) -> str:
    """Write a variable-dim LMR artifact.  Returns artifact SHA-256."""
    if model_name == "L":
        weights = model.linear.weight.detach().cpu().double().flatten().tolist()
        bias    = float(model.linear.bias.detach().cpu())
        dims    = 5
        tag     = 3
    elif model_name == "P":
        weights = model.linear.weight.detach().cpu().double().flatten().tolist()
        bias    = float(model.linear.bias.detach().cpu())
        dims    = 32
        tag     = 0
    elif model_name == "PL":
        weights = model.linear.weight.detach().cpu().double().flatten().tolist()
        bias    = float(model.linear.bias.detach().cpu())
        dims    = 37
        tag     = 1
    elif model_name == "PL-NL8":
        # Serialise as two-layer; loader must support this
        w1 = model.fc1.weight.detach().cpu().double().flatten().tolist()
        b1 = model.fc1.bias.detach().cpu().double().tolist()
        w2 = model.fc2.weight.detach().cpu().double().flatten().tolist()
        b2 = float(model.fc2.bias.detach().cpu())
        payload = bytearray(MAGIC_V3)
        # header: schema_ver=1, layer_count=2, model_name_tag=2 (PL-NL8)
        payload.extend(struct.pack("<III", 1, 2, 2))
        payload.extend(bytes.fromhex(trunk_hash))
        # layer 1: in=37, out=8, activation=1 (relu)
        payload.extend(struct.pack("<III", 37, 8, 1))
        payload.extend(struct.pack(f"<{37*8}d", *w1))
        payload.extend(struct.pack(f"<8d", *b1))
        # layer 2: in=8, out=1, activation=0 (linear/sigmoid)
        payload.extend(struct.pack("<III", 8, 1, 0))
        payload.extend(struct.pack(f"<8d", *w2))
        payload.extend(struct.pack("<d", b2))
        # calibration + threshold
        payload.extend(struct.pack("<ddd", scale, shift, threshold))
        digest = hashlib.sha256(bytes(payload)).digest()
        artifact = bytes(payload) + digest
        path.write_bytes(artifact)
        return hashlib.sha256(artifact).hexdigest()
    else:
        raise ValueError(f"Unknown model: {model_name!r}")

    # Single-layer models (P, PL)
    payload = bytearray(MAGIC_V3)
    # schema_ver=1, layer_count=1, model_name_tag=0(P)/1(PL)/3(L)
    payload.extend(struct.pack("<III", 1, 1, tag))
    payload.extend(bytes.fromhex(trunk_hash))
    payload.extend(struct.pack("<I", dims))
    payload.extend(struct.pack(f"<{dims}d", *weights))
    payload.extend(struct.pack("<dddd", bias, scale, shift, threshold))
    digest = hashlib.sha256(bytes(payload)).digest()
    artifact = bytes(payload) + digest
    path.write_bytes(artifact)
    return hashlib.sha256(artifact).hexdigest()


# ═══════════════════════════════════════════════════════════════════════════════
# Sweep runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_sweep(
    nat_train: list[dict],
    nat_cal: list[dict],
    hn_train: list[dict],
    trunk_sha: str,
    model_names: list[str],
    mixing_ratios: list[float],
    neg_weights: list[float],
    unsafe_weights: list[float],
    seeds: list[int],
    epochs: int,
    lr: float,
    label: str,
    ablation_zeros: set[int] | None = None,
) -> list[dict]:
    results = []
    total = len(model_names) * len(mixing_ratios) * len(neg_weights) * len(unsafe_weights) * len(seeds)
    done = 0
    for mname in model_names:
        for ratio_nat in mixing_ratios:
            for neg_w in neg_weights:
                for uw in unsafe_weights:
                    for seed in seeds:
                        done += 1
                        print(f"  [{label}] {done}/{total} {mname} "
                              f"r={ratio_nat:.0%} neg={neg_w} uw={uw} s={seed}",
                              end="  ", flush=True)
                        train_rows = mix_train(nat_train, hn_train, ratio_nat, seed)
                        model, scale, shift, thr, cal_stats = train_one(
                            train_rows, nat_cal,
                            seed=seed, model_name=mname,
                            neg_weight=neg_w, unsafe_weight=uw,
                            epochs=epochs, lr=lr,
                            ablation_zeros=ablation_zeros,
                        )
                        net = cal_stats.get("net_nodes_saved", 0)
                        safe_f = "SAFE" if cal_stats.get("unsafe_activations", 1) == 0 else "UNSAFE"
                        print(f"thr={thr} net={net:.1f} {safe_f}")
                        results.append({
                            "model": mname,
                            "ratio_nat": ratio_nat,
                            "neg_weight": neg_w,
                            "unsafe_weight": uw,
                            "seed": seed,
                            "train_rows": len(train_rows),
                            "scale": round(scale, 6),
                            "shift": round(shift, 6),
                            "threshold": thr,
                            "cal_stats": cal_stats,
                            "_model_state": model.state_dict(),
                            "_model_name": mname,
                            "_trunk_sha": trunk_sha,
                            "_ablation_zeros": list(ablation_zeros) if ablation_zeros else [],
                        })
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Feature ablation
# ═══════════════════════════════════════════════════════════════════════════════

ABLATION_SETUPS = {
    "full":             set(),
    "no_move_index":    {1},
    "no_rank_pct":      {6} if False else set(),  # context7 only; skip if context5
    "no_remaining_dep": {0},
    "no_base_red":      {2},
    "no_move_class":    {3, 4},  # is_horizontal, is_vertical
    "no_move_index_no_rank_pct": {1},  # same as no_move_index for context5
}


def run_ablation(
    nat_train: list[dict],
    nat_cal: list[dict],
    hn_train: list[dict],
    trunk_sha: str,
    frozen_config: dict,
    seeds: list[int],
    epochs: int,
    lr: float,
) -> dict[str, list[dict]]:
    abl_results: dict[str, list[dict]] = {}
    for abl_name, zeros in ABLATION_SETUPS.items():
        print(f"\n  Ablation: PL [{abl_name}]")
        res = run_sweep(
            nat_train, nat_cal, hn_train, trunk_sha,
            model_names=["PL"],
            mixing_ratios=[frozen_config["ratio_nat"]],
            neg_weights=[frozen_config["neg_weight"]],
            unsafe_weights=[frozen_config["unsafe_weight"]],
            seeds=seeds,
            epochs=epochs, lr=lr,
            label=f"abl-{abl_name}",
            ablation_zeros=zeros if zeros else None,
        )
        abl_results[abl_name] = res
    return abl_results


# ═══════════════════════════════════════════════════════════════════════════════
# Holdout enforcement
# ═══════════════════════════════════════════════════════════════════════════════

def check_holdout_sealed(manifest_path: Path, out_dir: Path) -> None:
    """Abort if holdout was opened before the manifest was written."""
    holdout_marker = out_dir / ".holdout_opened"
    if holdout_marker.exists() and not manifest_path.exists():
        raise RuntimeError(
            "FATAL: final holdout was opened before the experiment manifest was frozen!\n"
            "The holdout result is no longer pristine.  Start a fresh dataset."
        )


def decode_moves(row: dict) -> list[str]:
    encoded = row.get("moves_bin")
    if not encoded:
        return []
    return unpack_moves(base64.b64decode(encoded))


def run_engine_json(command: list[str]) -> tuple[subprocess.CompletedProcess[str], dict]:
    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    payloads: list[dict] = []
    for stream in (result.stdout, result.stderr):
        for line in stream.splitlines():
            line = line.strip()
            if line.startswith("info json "):
                payloads.append(json.loads(line[10:]))
            elif line.startswith("{") and line.endswith("}"):
                try:
                    payloads.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    payload = payloads[-1] if payloads else {}
    return result, payload


def select_shadow_rows(rows: list[dict], *, per_phase: int = 1, max_rows: int = 8) -> list[dict]:
    selected: list[dict] = []
    seen_phase: dict[str, int] = {}
    for row in rows:
        phase = str(row.get("phase_tag") or "unknown")
        if seen_phase.get(phase, 0) >= per_phase:
            continue
        if not row.get("moves_bin"):
            continue
        selected.append(row)
        seen_phase[phase] = seen_phase.get(phase, 0) + 1
        if len(selected) >= max_rows:
            break
    if selected:
        return selected
    return [row for row in rows if row.get("moves_bin")][:max_rows]


def run_shadow_validation(
    rows: list[dict],
    artifact_path: Path,
    out_dir: Path,
) -> dict:
    engine_bin = ROOT / "engine" / "target" / "release" / "titanium.exe"
    if not engine_bin.exists():
        raise RuntimeError(f"engine binary missing: {engine_bin}")
    selected = select_shadow_rows(rows)
    if not selected:
        raise RuntimeError("no rows available for shadow validation")
    comparisons = []
    total_eval = 0
    total_hyp = 0
    total_inference_nanos = 0
    for row in selected:
        moves = decode_moves(row)
        depth = max(4, int(row.get("depth", 5)))
        baseline_cmd = [
            str(engine_bin), "genmove", "--engine", "ace-v13-ti",
            "--depth", str(depth), "--full", *moves,
        ]
        shadow_cmd = [
            str(engine_bin), "reduction-shadow",
            "--depth", str(depth),
            "--sidecar", str(artifact_path),
            *moves,
        ]
        baseline_proc, baseline_payload = run_engine_json(baseline_cmd)
        shadow_proc, shadow_payload = run_engine_json(shadow_cmd)
        if baseline_proc.returncode != 0:
            raise RuntimeError(f"baseline shadow parity failed: {baseline_proc.stderr[-800:]}")
        if shadow_proc.returncode != 0:
            raise RuntimeError(f"shadow run failed: {shadow_proc.stderr[-800:]}")
        baseline_best = ""
        for line in baseline_proc.stdout.splitlines():
            if line.startswith("bestmove "):
                baseline_best = line.split(" ", 1)[1].strip()
        shadow_best = str(shadow_payload.get("bestmove", ""))
        same = {
            "bestmove": baseline_best == shadow_best,
            "score": baseline_payload.get("rootScore") == shadow_payload.get("score"),
            "depth": baseline_payload.get("searchDepth") == shadow_payload.get("depth"),
            "nodes": baseline_payload.get("nodes") == shadow_payload.get("nodes"),
        }
        comparisons.append({
            "phase_tag": row.get("phase_tag"),
            "position_ply": row.get("position_ply"),
            "depth": depth,
            "baseline_bestmove": baseline_best,
            "shadow_bestmove": shadow_best,
            "baseline_score": baseline_payload.get("rootScore"),
            "shadow_score": shadow_payload.get("score"),
            "baseline_nodes": baseline_payload.get("nodes"),
            "shadow_nodes": shadow_payload.get("nodes"),
            "same": same,
            "runtime_changed": shadow_payload.get("runtime_changed"),
            "evaluations": shadow_payload.get("evaluations"),
            "hypothetical_activations": shadow_payload.get("hypothetical_activations"),
            "inference_nanos": shadow_payload.get("inference_nanos"),
        })
        total_eval += int(shadow_payload.get("evaluations", 0) or 0)
        total_hyp += int(shadow_payload.get("hypothetical_activations", 0) or 0)
        total_inference_nanos += int(shadow_payload.get("inference_nanos", 0) or 0)
    report = {
        "artifact_path": str(artifact_path),
        "positions": len(comparisons),
        "comparisons": comparisons,
        "identical_bestmove": all(c["same"]["bestmove"] for c in comparisons),
        "identical_score": all(c["same"]["score"] for c in comparisons),
        "identical_depth": all(c["same"]["depth"] for c in comparisons),
        "identical_nodes": all(c["same"]["nodes"] for c in comparisons),
        "runtime_changed_false": all(c.get("runtime_changed") is False for c in comparisons),
        "total_evaluations": total_eval,
        "total_hypothetical_activations": total_hyp,
        "total_inference_nanos": total_inference_nanos,
    }
    (out_dir / "shadow_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--natural", required=True)
    parser.add_argument("--hard-negatives", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--phase",
                        choices=["narrowing", "stability", "manifest",
                                 "holdout", "shadow", "full"],
                        default="narrowing")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "experiment_manifest.json"
    holdout_marker = out_dir / ".holdout_opened"

    # ── A. Load data ───────────────────────────────────────────────────────────
    print("=== Loading data ===")
    nat_rows = load_rows(Path(args.natural))
    hn_rows  = load_rows(Path(args.hard_negatives)) if args.hard_negatives else []
    print(f"  natural: {len(nat_rows)} rows")
    print(f"  hard-negatives: {len(hn_rows)} rows")

    trunk_sha = hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()
    print(f"\n  trunk SHA-256 (frozen): {trunk_sha}")

    # Trunk binding verification
    print("\n=== Trunk binding verification ===")
    errors  = verify_trunk_binding(nat_rows, trunk_sha, "natural")
    errors += verify_trunk_binding(hn_rows,  trunk_sha, "hard_negatives")
    if errors:
        for e in errors:
            print(f"  ERROR: {e}", file=sys.stderr)
        return 1
    print("  Trunk binding: OK (all rows match or have no trunk_sha256)")

    # Feature schema check
    n_v2 = sum(1 for r in nat_rows if r.get("feature_schema") == FEATURE_SCHEMA_V2)
    print(f"  Feature schema v2 rows: {n_v2}/{len(nat_rows)}")

    # ── B. Group leakage check ─────────────────────────────────────────────────
    print("\n=== Group leakage check ===")
    leakage_errors = check_group_leakage(nat_rows, hn_rows)
    if any("DETECTED" in e for e in leakage_errors):
        for e in leakage_errors:
            print(f"  ERROR: {e}", file=sys.stderr)
        return 1
    for e in leakage_errors:
        print(f"  WARN: {e}")
    if not leakage_errors:
        print("  No group leakage detected")

    nat_train, nat_cal, nat_test = split_natural(nat_rows)
    hn_train_all = filter_hn_for_training(nat_rows, hn_rows)

    print(f"\n  natural train: {len(nat_train)} rows, "
          f"{sum(r.get('activate_plus_one',False) for r in nat_train)} pos, "
          f"{sum(r.get('sample_status')=='UNSAFE' for r in nat_train)} unsafe")
    print(f"  natural cal:   {len(nat_cal)} rows, "
          f"{sum(r.get('activate_plus_one',False) for r in nat_cal)} pos, "
          f"{sum(r.get('sample_status')=='UNSAFE' for r in nat_cal)} unsafe")
    print(f"  natural test:  {len(nat_test)} rows (SEALED until manifest frozen)")
    print(f"  HN for train:  {len(hn_train_all)} rows (enrichment only)")

    # Distribution analysis
    from train_reduction_sidecar_v2 import distribution_summary
    nat_dist = distribution_summary(nat_train + nat_cal)
    for group in ("move_index", "base_reduction", "depth"):
        nat_g = nat_dist.get(group, {})
        print(f"  {group} distribution: " +
              ", ".join(f"{k}:{v['n']}(pos={v['pos']})" for k, v in sorted(nat_g.items())))

    if args.dry_run:
        print("\n[dry-run] Stopping before training.")
        return 0

    if len(nat_cal) < 50:
        print(f"\nWARN: calibration has only {len(nat_cal)} rows — "
              f"collect more data before drawing conclusions")

    # ── Feature provenance output ──────────────────────────────────────────────
    prov_path = out_dir / "feature_provenance.json"
    prov_path.write_text(json.dumps(FEATURE_PROVENANCE, indent=2), encoding="utf-8")
    print(f"\n  Feature provenance written: {prov_path}")

    # ── C. Three-seed narrowing ────────────────────────────────────────────────
    if args.phase in ("narrowing", "full"):
        print("\n=== Three-seed narrowing sweep (P / PL / PL-NL8) ===")
        narrow_results = run_sweep(
            nat_train, nat_cal, hn_train_all, trunk_sha,
            model_names=["P", "L", "PL", "PL-NL8"],
            mixing_ratios=[0.70, 1.0],
            neg_weights=NEG_WEIGHTS,
            unsafe_weights=UNSAFE_WEIGHTS,
            seeds=SEEDS_NARROW,
            epochs=args.epochs, lr=args.lr,
            label="narrow",
        )
        # Save narrowing results
        narrow_path = out_dir / "narrowing_results.json"
        narrow_path.write_text(
            json.dumps([{k: v for k, v in r.items() if not k.startswith("_")}
                        for r in narrow_results], indent=2),
            encoding="utf-8"
        )
        print(f"\n  Narrowing results: {narrow_path}")

        # Select best per model
        best_per_model: dict[str, dict] = {}
        for mname in ("P", "L", "PL", "PL-NL8"):
            mresults = [r for r in narrow_results if r["model"] == mname]
            if mresults:
                best = select_best(mresults)
                best_per_model[mname] = best
                print(f"  Best {mname}: ratio={best['ratio_nat']:.0%} "
                      f"neg_w={best['neg_weight']} uw={best['unsafe_weight']} "
                      f"s={best['seed']} "
                      f"thr={best['threshold']} net={best['cal_stats'].get('net_nodes_saved',0):.1f}")

        if args.phase == "narrowing":
            print("\n  [phase=narrowing] Done. Run with --phase stability next.")
            return 0

    # ── D/E. Freeze families + ten-seed stability ──────────────────────────────
    if args.phase in ("stability", "full"):
        # Load narrowing if not already in memory
        if "narrow_results" not in dir():
            narrow_path = out_dir / "narrowing_results.json"
            if not narrow_path.exists():
                raise SystemExit("Run --phase narrowing first.")
            # Can't reload _model_state from JSON; re-run the sweep
            print("  Note: re-running narrowing to recover model states for stability sweep")
            narrow_results = run_sweep(
                nat_train, nat_cal, hn_train_all, trunk_sha,
                model_names=["P", "L", "PL", "PL-NL8"],
                mixing_ratios=[0.70, 1.0],
                neg_weights=NEG_WEIGHTS,
                unsafe_weights=UNSAFE_WEIGHTS,
                seeds=SEEDS_NARROW,
                epochs=args.epochs, lr=args.lr,
                label="narrow-replay",
            )
            best_per_model = {}
            for mname in ("P", "L", "PL", "PL-NL8"):
                mresults = [r for r in narrow_results if r["model"] == mname]
                if mresults:
                    best_per_model[mname] = select_best(mresults)

        # For each model, run ten-seed stability at its best narrow config
        print("\n=== Ten-seed stability comparison ===")
        stability_results: dict[str, list[dict]] = {}
        for mname in ("P", "L", "PL", "PL-NL8"):
            if mname not in best_per_model:
                continue
            best = best_per_model[mname]
            print(f"\n  {mname} — frozen config: ratio={best['ratio_nat']:.0%} "
                  f"neg_w={best['neg_weight']} uw={best['unsafe_weight']}")
            stab = run_sweep(
                nat_train, nat_cal, hn_train_all, trunk_sha,
                model_names=[mname],
                mixing_ratios=[best["ratio_nat"]],
                neg_weights=[best["neg_weight"]],
                unsafe_weights=[best["unsafe_weight"]],
                seeds=SEEDS_STABILITY,
                epochs=args.epochs, lr=args.lr,
                label=f"stab-{mname}",
            )
            stability_results[mname] = stab
            nets = [r["cal_stats"].get("net_nodes_saved", 0) for r in stab]
            print(f"  {mname} stability: min={min(nets):.1f} med={statistics.median(nets):.1f} "
                  f"max={max(nets):.1f} all_safe="
                  f"{all(r['cal_stats'].get('unsafe_activations',1)==0 for r in stab)}")

        stab_path = out_dir / "stability_results.json"
        stab_path.write_text(
            json.dumps({k: [{kk: vv for kk, vv in r.items() if not kk.startswith("_")}
                             for r in v]
                        for k, v in stability_results.items()}, indent=2),
            encoding="utf-8",
        )
        print(f"\n  Stability results: {stab_path}")

        # Feature ablation from best PL config
        print("\n=== Feature ablation (leave-one-out from PL) ===")
        best_PL = best_per_model.get("PL")
        if best_PL:
            abl_results = run_ablation(
                nat_train, nat_cal, hn_train_all, trunk_sha,
                frozen_config={
                    "ratio_nat": best_PL["ratio_nat"],
                    "neg_weight": best_PL["neg_weight"],
                    "unsafe_weight": best_PL["unsafe_weight"],
                },
                seeds=SEEDS_NARROW,
                epochs=args.epochs, lr=args.lr,
            )
            abl_summary: dict[str, Any] = {}
            print("\n  Ablation summary (calibration median net savings):")
            for abl_name, res in abl_results.items():
                nets = [r["cal_stats"].get("net_nodes_saved", 0) for r in res]
                med  = statistics.median(nets) if nets else float("nan")
                abl_summary[abl_name] = {"median_cal_net": round(med, 2), "seeds": len(res)}
                print(f"    PL [{abl_name:<22}]  median_net={med:.2f}")

            abl_path = out_dir / "ablation_results.json"
            abl_path.write_text(json.dumps(abl_summary, indent=2), encoding="utf-8")
            print(f"  Ablation results: {abl_path}")
        else:
            abl_results = {}

        # Handcrafted rule baselines on calibration
        print("\n=== Handcrafted rule baselines (calibration) ===")
        cal_baselines: dict[str, dict] = {}
        for rule_name, pred in HANDCRAFTED_RULES.items():
            rb = rule_baseline(nat_cal, pred)
            cal_baselines[rule_name] = rb
            print(f"  {rule_name:<25} act={rb['activations']:5d} tp={rb['true_positives']:4d} "
                  f"unsafe={rb['unsafe']:3d} net={rb['net']:8.2f} prec={rb['precision']:.4f}")

        (out_dir / "baseline_comparisons_cal.json").write_text(
            json.dumps(cal_baselines, indent=2), encoding="utf-8"
        )

        if args.phase == "stability":
            print("\n  [phase=stability] Done. Run with --phase manifest next.")
            return 0

    # ── H. Freeze experiment manifest ─────────────────────────────────────────
    if args.phase in ("manifest", "full"):
        check_holdout_sealed(manifest_path, out_dir)

        # Select the single best overall candidate from stability results
        if "stability_results" not in dir():
            stab_path = out_dir / "stability_results.json"
            if not stab_path.exists():
                raise SystemExit("Run --phase stability first.")
            stability_results = json.loads(stab_path.read_text(encoding="utf-8"))

        # Pick the model family with highest median stable cal_net
        best_mname = max(
            stability_results.keys(),
            key=lambda m: statistics.median(
                r["cal_stats"].get("net_nodes_saved", 0)
                for r in stability_results[m]
            ),
        )
        best_stab_run = select_best(stability_results[best_mname])
        print(f"\n=== Freezing manifest: model={best_mname} "
              f"seed={best_stab_run['seed']} thr={best_stab_run['threshold']} ===")

        # Re-train the frozen model once
        frozen_train = mix_train(nat_train, hn_train_all,
                                 best_stab_run["ratio_nat"],
                                 best_stab_run["seed"])
        frozen_model, frozen_scale, frozen_shift, frozen_thr, _ = train_one(
            frozen_train, nat_cal,
            seed=best_stab_run["seed"],
            model_name=best_mname,
            neg_weight=best_stab_run["neg_weight"],
            unsafe_weight=best_stab_run["unsafe_weight"],
            epochs=args.epochs, lr=args.lr,
        )
        artifact_path = out_dir / f"lmr_sidecar_{best_mname.lower().replace('-','_')}.bin"
        artifact_sha = write_artifact_v3(
            artifact_path, frozen_model, best_mname,
            trunk_sha, frozen_scale, frozen_shift, frozen_thr,
        )

        # Dataset hashes
        nat_hash = hashlib.sha256(Path(args.natural).read_bytes()).hexdigest()
        hn_hash  = hashlib.sha256(Path(args.hard_negatives).read_bytes()).hexdigest() \
                   if args.hard_negatives else "N/A"

        manifest = {
            "phase": "manifest",
            "frozen": True,
            "trunk_sha256": trunk_sha,
            "natural_dataset_sha256": nat_hash,
            "hard_negative_dataset_sha256": hn_hash,
            "natural_rows": len(nat_rows),
            "hn_rows": len(hn_rows),
            "split": {
                "train": len(nat_train), "calibration": len(nat_cal), "final_test": len(nat_test),
                "seed": "stable_partition(game_key, split_seed)",
                "group": "source_game_key (SHA-256 of packed moves)",
            },
            "feature_schema": FEATURE_SCHEMA_V2,
            "context5_order": CONTEXT5_NAMES,
            "selected_model": best_mname,
            "selected_config": {
                "ratio_nat": best_stab_run["ratio_nat"],
                "neg_weight": best_stab_run["neg_weight"],
                "unsafe_weight": best_stab_run["unsafe_weight"],
                "epochs": args.epochs,
                "lr": args.lr,
            },
            "selected_seed": best_stab_run["seed"],
            "seed_selection_rule": "max cal_net_nodes_saved among seeds with unsafe==0",
            "threshold": frozen_thr,
            "threshold_selection_rule":
                "max expected signed net savings on natural calibration, subject to unsafe_activations==0",
            "scale": frozen_scale,
            "shift": frozen_shift,
            "artifact_path": str(artifact_path),
            "artifact_sha256": artifact_sha,
            "inference_cost_node_equiv": INFERENCE_COST_NODES,
            "runtime_active": False,
            "final_holdout_opened": False,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"  Manifest frozen: {manifest_path}")
        print(f"  Artifact: {artifact_path} (sha256={artifact_sha[:16]}...)")

        if args.phase == "manifest":
            print("\n  [phase=manifest] Done. Run with --phase holdout next.")
            return 0

    # ── I. Open fresh final holdout once ──────────────────────────────────────
    if args.phase in ("holdout", "full"):
        if not manifest_path.exists():
            raise SystemExit("Manifest must be frozen before opening the holdout. "
                             "Run --phase manifest first.")
        if holdout_marker.exists():
            raise SystemExit("Final holdout has already been opened once. "
                             "Do not re-open it.")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        best_mname   = manifest["selected_model"]
        frozen_thr   = manifest["threshold"]
        frozen_scale = manifest["scale"]
        frozen_shift = manifest["shift"]
        frozen_config = manifest["selected_config"]

        # Re-train the frozen model (needed for evaluation)
        frozen_train = mix_train(nat_train, hn_train_all,
                                 frozen_config["ratio_nat"],
                                 manifest["selected_seed"])
        frozen_model, _, _, _, _ = train_one(
            frozen_train, nat_cal,
            seed=manifest["selected_seed"],
            model_name=best_mname,
            neg_weight=frozen_config["neg_weight"],
            unsafe_weight=frozen_config["unsafe_weight"],
            epochs=args.epochs, lr=args.lr,
        )

        print("\n=== OPENING FINAL HOLDOUT (once) ===")
        holdout_marker.write_text("opened")

        test_eval = evaluate_at_threshold(
            frozen_model, nat_test,
            frozen_scale, frozen_shift, frozen_thr, best_mname,
        )
        test_sweep = full_threshold_sweep(
            frozen_model, nat_test, frozen_scale, frozen_shift, best_mname,
        )

        print(f"  model={best_mname}  threshold={frozen_thr}")
        print(f"  rows={test_eval['rows']}  positives={test_eval['positives']}")
        print(f"  activations={test_eval['activations']}  TP={test_eval['true_positives']}  "
              f"unsafe={test_eval['unsafe_activations']}")
        print(f"  precision={test_eval['precision']:.4f} "
              f"(wl95={test_eval['precision_wilson_lower_95']:.4f})  "
              f"recall={test_eval['recall']:.4f}")
        print(f"  net_nodes_saved={test_eval['net_nodes_saved']:.2f}  "
              f"unsafe_rate={test_eval['unsafe_rate']:.4f}")

        # Handcrafted baselines on holdout
        print("\n  Handcrafted rule baselines (final holdout):")
        holdout_baselines: dict[str, dict] = {}
        for rule_name, pred in HANDCRAFTED_RULES.items():
            rb = rule_baseline(nat_test, pred)
            holdout_baselines[rule_name] = rb
            print(f"    {rule_name:<25} act={rb['activations']:5d} "
                  f"tp={rb['true_positives']:4d} unsafe={rb['unsafe']:3d} "
                  f"net={rb['net']:8.2f}")

        # GO/NO-GO determination
        net = test_eval["net_nodes_saved"]
        unsafe = test_eval["unsafe_activations"]
        n_unsafe_holdout = sum(1 for r in nat_test if r.get("sample_status") == "UNSAFE")
        best_rule_net = max(rb["net"] for rb in holdout_baselines.values())
        model_beats_rules = net > best_rule_net

        if net > 0 and unsafe == 0 and model_beats_rules:
            verdict = "GO"
        elif unsafe > 0:
            verdict = f"NO-GO (unsafe activations: {unsafe})"
        elif net <= 0:
            verdict = f"NO-GO (net_saved={net:.2f} <= 0)"
        elif not model_beats_rules:
            verdict = f"NO-GO (model net={net:.2f} < best rule net={best_rule_net:.2f})"
        else:
            verdict = "NO-GO"

        print(f"\n  FINAL HOLDOUT VERDICT: {verdict}")
        print(f"  Sample size caution: n={len(nat_test)} holdout rows — "
              f"uncertainty is large; interpret GO/NO-GO with appropriate CI")
        print(f"  Unsafe coverage in holdout: {n_unsafe_holdout} genuine unsafe events")
        if n_unsafe_holdout == 0:
            print("  WARNING: Zero unsafe events in holdout — safety claim is not empirically validated")

        holdout_report = {
            "model": best_mname,
            "threshold": frozen_thr,
            "test_eval": test_eval,
            "test_sweep": test_sweep,
            "holdout_baselines": holdout_baselines,
            "unsafe_events_in_holdout": n_unsafe_holdout,
            "verdict": verdict,
            "runtime_active": False,
        }
        (out_dir / "holdout_report.json").write_text(
            json.dumps(holdout_report, indent=2), encoding="utf-8"
        )

        # Update manifest to record holdout opened
        manifest["final_holdout_opened"] = True
        manifest["final_holdout_verdict"] = verdict
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        trunk_after = hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()
        if trunk_sha != trunk_after:
            print("FATAL: trunk hash changed during training!", file=sys.stderr)
            return 1
        print(f"\n  Trunk freeze verified: unchanged ({trunk_sha[:16]}...)")
        print(f"  Runtime activation: OFF")
        print(f"  Nothing was pushed")

        if args.phase == "holdout":
            return 0 if verdict.startswith("GO") else 3

    if args.phase in ("shadow", "full"):
        if not manifest_path.exists():
            raise SystemExit("Manifest must be frozen before shadow validation. Run --phase manifest first.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        artifact_path = Path(manifest["artifact_path"])
        if not artifact_path.exists():
            raise SystemExit(f"Shadow artifact missing: {artifact_path}")
        shadow_rows = [r for r in nat_rows if r.get("sample_status") != "UNKNOWN"]
        print("\n=== Shadow validation ===")
        shadow_report = run_shadow_validation(shadow_rows, artifact_path, out_dir)
        print(f"  positions={shadow_report['positions']}")
        print(
            f"  identical bestmove/score/depth/nodes = "
            f"{shadow_report['identical_bestmove']}/"
            f"{shadow_report['identical_score']}/"
            f"{shadow_report['identical_depth']}/"
            f"{shadow_report['identical_nodes']}"
        )
        print(
            f"  hypothetical activations={shadow_report['total_hypothetical_activations']} "
            f"evaluations={shadow_report['total_evaluations']} "
            f"inference_nanos={shadow_report['total_inference_nanos']}"
        )
        if not (
            shadow_report["identical_bestmove"]
            and shadow_report["identical_score"]
            and shadow_report["identical_depth"]
            and shadow_report["identical_nodes"]
            and shadow_report["runtime_changed_false"]
        ):
            raise SystemExit("Shadow validation failed parity invariants.")
        if args.phase == "shadow":
            print("\n  [phase=shadow] Done.")
            return 0

    print(f"\n=== Complete ===")
    print(f"  Runtime activation: OFF")
    print(f"  Nothing was pushed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
