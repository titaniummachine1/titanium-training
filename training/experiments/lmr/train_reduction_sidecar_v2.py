#!/usr/bin/env python3
"""Stage-1 LMR sidecar — full training sweep with ablations.

Selection rule: calibration only. Final-test opened once after freeze.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.lmr.reduction_counterfactual_schema import (
    FEATURE_SCHEMA,
    FEATURE_SCHEMA_V2,
    SCHEMA,
    SIDECAR_SCHEMA,
    rank_percentile,
    validate_row,
    wilson_lower,
)

WEIGHTS = ROOT / "engine" / "src" / "acev13" / "net_weights.bin"
SIDECAR_OUT_DIR = ROOT / "training" / "checkpoints"

# Inference cost in node-equivalents (583 ns / 833 ns-per-node)
INFERENCE_COST_NODES = 0.7
# Safety requirement on calibration: max unsafe activations allowed
MAX_UNSAFE_CAL = 0

SEEDS = [42, 137, 271, 512, 1337, 2027, 4099, 8191, 16381, 65537]
THRESHOLD_GRID = [
    0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
    0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 0.95, 0.99,
]

# Stage-1 ablation variants (binary-format canonical width = 37)
# A: hidden32 + context5 (37 total)
# B: hidden32 + context4 (move_index removed → 36)
# C: context5 only (5)
# D: hidden32 only (32)
#
# Stage-2 feature-group variants (require v2 probe data with total_legal_moves / history_score)
# P:       hidden32 only (32)                 — Group P
# L:       context5 only (5)                  — Group L
# PL:      hidden32 + context5 (37)           — Groups P+L  (= variant A)
# PLO:     hidden32 + context5 + history (38) — Groups P+L+O
# PLOB:    hidden32 + context5 + history + rank_pct (39) — Groups P+L+O+B
#
# Leave-one-out (starting from PLOB / PLO best):
# PL_no_depth:    PL with remaining_depth zeroed
# PL_no_midx:     PL with move_index zeroed (= variant B at 37 dims)
# PL_no_red:      PL with base_reduction zeroed
VARIANTS = {
    # Stage-1 ablation set
    "A":  {"dims": 37, "use_hidden": True,  "context_src": "context5",
           "zero_move_index": False, "pad_to": 37},
    "B":  {"dims": 36, "use_hidden": True,  "context_src": "context5",
           "zero_move_index": True,  "pad_to": 37},
    "C":  {"dims": 5,  "use_hidden": False, "context_src": "context5",
           "zero_move_index": False, "pad_to": 37},
    "D":  {"dims": 32, "use_hidden": True,  "context_src": None,
           "zero_move_index": False, "pad_to": 37},
    # Stage-2 feature-group set (require v2 data)
    "P":    {"dims": 32, "use_hidden": True,  "context_src": None,
             "zero_move_index": False, "pad_to": None, "requires_v2": True},
    "L":    {"dims": 5,  "use_hidden": False, "context_src": "context5",
             "zero_move_index": False, "pad_to": None, "requires_v2": True},
    "PL":   {"dims": 37, "use_hidden": True,  "context_src": "context5",
             "zero_move_index": False, "pad_to": None, "requires_v2": True},
    "PLO":  {"dims": 38, "use_hidden": True,  "context_src": "context5+history",
             "zero_move_index": False, "pad_to": None, "requires_v2": True},
    "PLOB": {"dims": 39, "use_hidden": True,  "context_src": "context5+history+rank",
             "zero_move_index": False, "pad_to": None, "requires_v2": True},
}


# ── data loading ────────────────────────────────────────────────────────────

def load_file(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        validate_row(row)
        rows.append(row)
    return rows


def integrity_check(natural_rows: list[dict], strat_rows: list[dict], trunk_sha: str) -> list[str]:
    errors = []
    for i, row in enumerate(natural_rows + strat_rows):
        if row.get("schema") != SCHEMA:
            errors.append(f"schema mismatch row {i}")
        fs = row.get("feature_schema")
        if fs not in (FEATURE_SCHEMA, FEATURE_SCHEMA_V2):
            errors.append(f"feature_schema mismatch row {i}: {fs!r}")
        h32 = row.get("hidden32", [])
        c5  = row.get("context5", [])
        if len(h32) != 32 or len(c5) != 5:
            errors.append(f"feature dim mismatch row {i}")
        c7 = row.get("context7") or []
        if c7 and len(c7) != 7:
            errors.append(f"context7 dim mismatch row {i}: got {len(c7)}")
        if any(not isinstance(v, (int, float)) or v != v or v == float('inf') or v == float('-inf')
               for v in h32 + c5 + c7):
            errors.append(f"non-finite feature in row {i}")
        if row.get("sample_status") == "UNKNOWN":
            errors.append(f"UNKNOWN row must be excluded from supervised data, row {i}")
        act = row.get("activate_plus_one")
        safe = row.get("safe_plus_one_reduction")
        worthy = row.get("worthwhile_net_savings")
        if act and not (safe and worthy):
            errors.append(f"label inconsistency: activate without safety/worthwhile in row {i}")
        if row.get("trunk_sha256") and row["trunk_sha256"] != trunk_sha:
            errors.append(f"trunk hash mismatch row {i}")

    nat_train  = {r["source_game_key"] for r in natural_rows if r.get("split") == "train"}
    nat_cal    = {r["source_game_key"] for r in natural_rows if r.get("split") == "calibration"}
    nat_test   = {r["source_game_key"] for r in natural_rows if r.get("split") == "final_test"}
    if nat_train & nat_cal:
        errors.append(f"game-key overlap train∩calibration: {len(nat_train&nat_cal)} keys")
    if nat_train & nat_test:
        errors.append(f"game-key overlap train∩final_test: {len(nat_train&nat_test)} keys")
    if nat_cal & nat_test:
        errors.append(f"game-key overlap calibration∩final_test: {len(nat_cal&nat_test)} keys")

    # Event-level dedup: (parent_hash, child_hash, move) in natural vs stratified
    nat_events = {(r.get("parent_hash"), r.get("child_hash"), r.get("move"))
                  for r in natural_rows}
    dup_count = sum(
        1 for r in strat_rows
        if (r.get("parent_hash"), r.get("child_hash"), r.get("move")) in nat_events
    )
    if dup_count:
        errors.append(f"WARN: {dup_count} stratified events also appear in natural stream (will be excluded from stratified training set)")

    return errors


def dedup_stratified(natural_rows: list[dict], strat_rows: list[dict]) -> list[dict]:
    nat_events = {(r.get("parent_hash"), r.get("child_hash"), r.get("move"))
                  for r in natural_rows}
    return [r for r in strat_rows
            if (r.get("parent_hash"), r.get("child_hash"), r.get("move")) not in nat_events]


# ── features ────────────────────────────────────────────────────────────────

def get_features(row: dict, variant: str) -> list[float]:
    cfg = VARIANTS[variant]
    h32 = [float(v) for v in row["hidden32"]]
    c5  = [float(v) for v in row["context5"]]

    if cfg.get("zero_move_index"):
        c5 = [c5[0], 0.0, c5[2], c5[3], c5[4]]  # zero move_index (index 1)

    ctx_src = cfg.get("context_src")
    parts: list[float] = []

    if cfg["use_hidden"]:
        parts.extend(h32)

    if ctx_src == "context5":
        if cfg.get("zero_move_index"):
            # drop zeroed slot to get true 36-dim
            parts.extend([c5[0], c5[2], c5[3], c5[4]])
        else:
            parts.extend(c5)
    elif ctx_src == "context5+history":
        parts.extend(c5)
        # history_score: soft-clip to [-10000,10000] then normalise to [0,1]
        raw_h = int(row.get("history_score") or 0)
        parts.append(max(0.0, min(1.0, (raw_h + 10000) / 20000.0)))
    elif ctx_src == "context5+history+rank":
        parts.extend(c5)
        raw_h = int(row.get("history_score") or 0)
        parts.append(max(0.0, min(1.0, (raw_h + 10000) / 20000.0)))
        mi = int(row.get("move_index", 0))
        n  = int(row.get("total_legal_moves") or 128)
        parts.append(rank_percentile(mi, n))
    # ctx_src == None: no context features appended

    assert len(parts) == cfg["dims"], f"variant {variant}: expected {cfg['dims']} got {len(parts)}"
    return parts


def to_tensors(rows: list[dict], variant: str):
    x = torch.tensor([get_features(r, variant) for r in rows], dtype=torch.float32)
    y = torch.tensor([float(r["activate_plus_one"]) for r in rows], dtype=torch.float32)
    is_unsafe = torch.tensor([float(r["sample_status"] == "UNSAFE") for r in rows], dtype=torch.float32)
    return x, y, is_unsafe


# ── source mixing ────────────────────────────────────────────────────────────

def mix_train(nat_train: list[dict], strat_all: list[dict], ratio_nat: float, seed: int) -> list[dict]:
    """Return training rows with natural/stratified ratio.
    Keeps all natural train rows; samples stratified to match ratio.
    ratio_nat = fraction of total desired from natural (0.33/0.50/0.67).
    """
    rng = random.Random(seed ^ 0xABCD)
    n_nat = len(nat_train)
    if ratio_nat >= 1.0:
        return nat_train[:]
    n_strat_target = int(n_nat * (1.0 - ratio_nat) / ratio_nat)
    n_strat = min(n_strat_target, len(strat_all))
    strat_sample = rng.sample(strat_all, n_strat)
    combined = nat_train + strat_sample
    rng.shuffle(combined)
    return combined


# ── Platt scaling ────────────────────────────────────────────────────────────

def fit_platt(logits: torch.Tensor, labels: torch.Tensor):
    if len(labels) < 4 or labels.min() == labels.max():
        return 1.0, 0.0
    scale = nn.Parameter(torch.ones(()))
    shift = nn.Parameter(torch.zeros(()))
    opt = torch.optim.LBFGS([scale, shift], max_iter=100, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(scale * logits + shift, labels)
        loss.backward()
        return loss
    opt.step(closure)
    return float(scale.detach()), float(shift.detach())


# ── threshold selection (calibration only) ───────────────────────────────────

def select_threshold(probs: list[float], rows: list[dict]) -> tuple[float, dict]:
    """Select threshold that maximises net savings on calibration, unsafe==0 only."""
    best_t = 1.0  # default: never activate
    best_stats: dict = {
        "threshold": 1.0, "activations": 0, "true_positives": 0,
        "unsafe_activations": 0, "precision": 0.0,
        "wilson_lower_95": 0.0, "recall": 0.0,
        "tp_delta": 0, "safe_fp_delta": 0, "unsafe_delta": 0,
        "gross_nodes_saved": 0, "inference_cost_nodes": 0.0,
        "net_nodes_saved": 0.0, "activation_rate": 0.0,
        "feasible": False,
    }
    total_positives = sum(1 for r in rows if r["activate_plus_one"])

    for t in THRESHOLD_GRID:
        activated = [(p, r) for p, r in zip(probs, rows) if p >= t]
        n_act = len(activated)
        n_unsafe = sum(1 for _, r in activated if r["sample_status"] == "UNSAFE")
        if n_unsafe > MAX_UNSAFE_CAL:
            continue  # safety constraint
        n_tp = sum(1 for _, r in activated if r["activate_plus_one"])
        # Signed economics: ALL activated events contribute their true delta.
        # TPs save nodes; safe-FPs may contribute a small delta; UNSAFEs a large negative.
        tp_delta = sum(int(r.get("net_nodes_saved", 0)) for _, r in activated if r["activate_plus_one"])
        safe_fp_delta = sum(int(r.get("net_nodes_saved", 0)) for _, r in activated
                            if r["sample_status"] == "SAFE" and not r["activate_plus_one"])
        unsafe_delta = sum(int(r.get("net_nodes_saved", 0)) for _, r in activated
                           if r["sample_status"] == "UNSAFE")
        gross = tp_delta + safe_fp_delta + unsafe_delta
        inference_cost = INFERENCE_COST_NODES * n_act
        net = gross - inference_cost
        precision = n_tp / n_act if n_act else 0.0
        wl = wilson_lower(n_tp, n_act)
        recall = n_tp / max(1, total_positives)
        act_rate = n_act / len(rows)
        stats = {
            "threshold": t,
            "activations": n_act,
            "true_positives": n_tp,
            "unsafe_activations": n_unsafe,
            "precision": round(precision, 4),
            "wilson_lower_95": round(wl, 4),
            "recall": round(recall, 4),
            "tp_delta": tp_delta,
            "safe_fp_delta": safe_fp_delta,
            "unsafe_delta": unsafe_delta,
            "gross_nodes_saved": gross,
            "inference_cost_nodes": round(inference_cost, 2),
            "net_nodes_saved": round(net, 2),
            "activation_rate": round(act_rate, 4),
            "feasible": True,
        }
        # Select: net > 0 AND improves on current best by net_saved
        if net > 0 and net > best_stats.get("net_nodes_saved", -999):
            best_t = t
            best_stats = stats

    return best_t, best_stats


def evaluate_at_threshold(
    model: nn.Linear, rows: list[dict], scale: float, shift: float,
    threshold: float, variant: str,
) -> dict:
    if not rows:
        return {"rows": 0}
    x, y, is_unsafe = to_tensors(rows, variant)
    with torch.no_grad():
        logits = model(x).squeeze(1)
        probs = torch.sigmoid(scale * logits + shift)
    active_mask = probs >= threshold
    n_act = int(active_mask.sum())
    n_tp = int(((y > 0.5) & active_mask).sum())
    n_unsafe_act = int((is_unsafe.bool() & active_mask).sum())
    active_list = active_mask.tolist()
    # Signed economics: ALL activated events contribute their true signed delta.
    tp_delta = sum(int(r.get("net_nodes_saved", 0)) for r, a in zip(rows, active_list)
                   if a and r["activate_plus_one"])
    safe_fp_delta = sum(int(r.get("net_nodes_saved", 0)) for r, a in zip(rows, active_list)
                        if a and r["sample_status"] == "SAFE" and not r["activate_plus_one"])
    unsafe_delta = sum(int(r.get("net_nodes_saved", 0)) for r, a in zip(rows, active_list)
                       if a and r["sample_status"] == "UNSAFE")
    gross = tp_delta + safe_fp_delta + unsafe_delta
    inference_cost = INFERENCE_COST_NODES * n_act
    net = gross - inference_cost
    total_pos = int(y.sum())
    return {
        "rows": len(rows),
        "positives": total_pos,
        "activations": n_act,
        "true_positives": n_tp,
        "unsafe_activations": n_unsafe_act,
        "false_activations_safe": n_act - n_tp - n_unsafe_act,
        "precision": round(n_tp / n_act, 4) if n_act else 0.0,
        "precision_wilson_lower_95": round(wilson_lower(n_tp, n_act), 4),
        "recall": round(n_tp / max(1, total_pos), 4),
        "tp_delta": tp_delta,
        "safe_fp_delta": safe_fp_delta,
        "unsafe_delta": unsafe_delta,
        "gross_nodes_saved": gross,
        "inference_cost_nodes": round(inference_cost, 2),
        "net_nodes_saved": round(net, 2),
        "activation_rate": round(n_act / len(rows), 4),
        "unsafe_rate": round(n_unsafe_act / max(1, n_act), 4),
    }


def full_threshold_sweep(
    model: nn.Linear, rows: list[dict], scale: float, shift: float, variant: str,
) -> list[dict]:
    if not rows:
        return []
    x, y, is_unsafe = to_tensors(rows, variant)
    with torch.no_grad():
        logits = model(x).squeeze(1)
        probs = torch.sigmoid(scale * logits + shift)
    probs_list = probs.tolist()
    total_pos = int(y.sum())
    results = []
    for t in THRESHOLD_GRID:
        active_mask = [p >= t for p in probs_list]
        n_act = sum(active_mask)
        n_tp = sum(a and r["activate_plus_one"] for a, r in zip(active_mask, rows))
        n_unsafe_act = sum(a and r["sample_status"] == "UNSAFE" for a, r in zip(active_mask, rows))
        tp_delta = sum(int(r.get("net_nodes_saved", 0)) for a, r in zip(active_mask, rows)
                       if a and r["activate_plus_one"])
        safe_fp_delta = sum(int(r.get("net_nodes_saved", 0)) for a, r in zip(active_mask, rows)
                            if a and r["sample_status"] == "SAFE" and not r["activate_plus_one"])
        unsafe_delta = sum(int(r.get("net_nodes_saved", 0)) for a, r in zip(active_mask, rows)
                           if a and r["sample_status"] == "UNSAFE")
        gross = tp_delta + safe_fp_delta + unsafe_delta
        net = gross - INFERENCE_COST_NODES * n_act
        results.append({
            "threshold": t,
            "activations": n_act,
            "true_positives": n_tp,
            "unsafe_activations": n_unsafe_act,
            "precision": round(n_tp / n_act, 4) if n_act else 0.0,
            "wilson_lower_95": round(wilson_lower(n_tp, n_act), 4),
            "recall": round(n_tp / max(1, total_pos), 4),
            "tp_delta": tp_delta,
            "safe_fp_delta": safe_fp_delta,
            "unsafe_delta": unsafe_delta,
            "gross_nodes_saved": gross,
            "net_nodes_saved": round(net, 2),
            "activation_rate": round(n_act / max(1, len(rows)), 4),
        })
    return results


# ── training ─────────────────────────────────────────────────────────────────

def train_one(
    train_rows: list[dict],
    cal_rows: list[dict],
    *,
    seed: int,
    variant: str,
    neg_weight: float,
    unsafe_weight: float,
    epochs: int,
    lr: float,
) -> tuple[nn.Linear, float, float, float, dict]:
    dims = VARIANTS[variant]["dims"]
    torch.manual_seed(seed)
    model = nn.Linear(dims, 1)
    nn.init.zeros_(model.weight)
    nn.init.constant_(model.bias, -6.0)

    x_train, y_train, is_unsafe_train = to_tensors(train_rows, variant)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for _ in range(epochs):
        opt.zero_grad()
        logits = model(x_train).squeeze(1)
        per_row = F.binary_cross_entropy_with_logits(logits, y_train, reduction="none")
        # Three-tier weighting: positive=1, ordinary_neg=neg_weight, unsafe=unsafe_weight
        w = torch.where(
            y_train > 0.5,
            torch.ones_like(y_train),
            torch.where(is_unsafe_train > 0.5,
                        torch.full_like(y_train, unsafe_weight),
                        torch.full_like(y_train, neg_weight)),
        )
        loss = (per_row * w).mean()
        loss.backward()
        opt.step()

    # Calibrate on calibration set
    x_cal, y_cal, _ = to_tensors(cal_rows, variant)
    with torch.no_grad():
        raw_cal = model(x_cal).squeeze(1)
    scale, shift = fit_platt(raw_cal, y_cal)
    probs_cal = torch.sigmoid(scale * raw_cal + shift).tolist()
    threshold, cal_stats = select_threshold(probs_cal, cal_rows)
    return model, scale, shift, threshold, cal_stats


# ── binary serialisation ──────────────────────────────────────────────────────

MAGIC = b"TISRDX1\0"
INPUTS_CANONICAL = 37


def write_binary(path: Path, model: nn.Linear, variant: str, trunk_hash: str,
                 scale: float, shift: float, threshold: float) -> tuple[str, str]:
    """Write canonical 37-input binary artifact.

    Stage-1 ablation variants (A/B/C/D) are padded to 37.
    Stage-2 feature-group variants with pad_to=None cannot be serialised to
    the 37-input format and must not call this function.
    """
    cfg = VARIANTS[variant]
    if cfg.get("pad_to") is None:
        raise ValueError(
            f"variant {variant!r} has pad_to=None; binary format requires 37 inputs. "
            "Extend loader before writing artifact for this variant."
        )
    dims = cfg["dims"]
    weights_vec = model.weight.detach().cpu().double().flatten().tolist()
    bias = float(model.bias.detach().cpu())

    # The loader always expects 37 inputs (INPUTS in reduction_sidecar.rs).
    # For variants with fewer dims, we pad with zeros.
    # For variant B (36 dims, move_index dropped), we re-insert a zero weight at position 33.
    full_weights = list(weights_vec)
    if variant == "B":
        # Reinsert zero weight for the dropped move_index at context position 1 (global index 33)
        full_weights = full_weights[:33] + [0.0] + full_weights[33:]
    elif variant == "C":
        full_weights = [0.0] * 32 + full_weights  # pad 32 hidden zeros
    elif variant == "D":
        full_weights = full_weights + [0.0] * 5   # pad 5 context zeros

    assert len(full_weights) == INPUTS_CANONICAL, f"expected 37, got {len(full_weights)}"

    payload = bytearray(MAGIC)
    payload.extend(struct.pack("<III", 1, 1, INPUTS_CANONICAL))
    payload.extend(bytes.fromhex(trunk_hash))
    payload.extend(struct.pack("<II", 1, 1))
    payload.extend(struct.pack(f"<{INPUTS_CANONICAL}d", *full_weights))
    payload.extend(struct.pack("<dddd", bias, scale, shift, threshold))
    digest = hashlib.sha256(bytes(payload)).digest()
    artifact = bytes(payload) + digest
    path.write_bytes(artifact)
    return digest.hex(), hashlib.sha256(artifact).hexdigest()


# ── full sweep ────────────────────────────────────────────────────────────────

def run_sweep(
    nat_train: list[dict], nat_cal: list[dict], nat_test: list[dict],
    strat_train: list[dict],
    trunk_sha: str,
    variant: str,
    mixing_ratios: list[float],
    neg_weights: list[float],
    unsafe_weights: list[float],
    seeds: list[int],
    epochs: int,
    lr: float,
    label: str,
) -> list[dict]:
    results = []
    total = len(mixing_ratios) * len(neg_weights) * len(unsafe_weights) * len(seeds)
    done = 0
    for ratio_nat in mixing_ratios:
        for neg_w in neg_weights:
            for uw in unsafe_weights:
                for seed in seeds:
                    done += 1
                    print(f"  [{label}] {done}/{total} ratio={ratio_nat:.0%} "
                          f"neg_w={neg_w} uw={uw} seed={seed}", end="  ", flush=True)
                    train_set = mix_train(nat_train, strat_train, ratio_nat, seed)
                    model, scale, shift, threshold, cal_stats = train_one(
                        train_set, nat_cal,
                        seed=seed, variant=variant,
                        neg_weight=neg_w, unsafe_weight=uw,
                        epochs=epochs, lr=lr,
                    )
                    cal_eval = evaluate_at_threshold(
                        model, nat_cal, scale, shift, threshold, variant)
                    # Do NOT open final_test here
                    run = {
                        "variant": variant,
                        "ratio_nat": ratio_nat,
                        "neg_weight": neg_w,
                        "unsafe_weight": uw,
                        "seed": seed,
                        "train_rows": len(train_set),
                        "scale": round(scale, 6),
                        "shift": round(shift, 6),
                        "threshold": threshold,
                        "cal_selection_stats": cal_stats,
                        "cal_eval": cal_eval,
                        # Save model state for later
                        "_model_state": model.state_dict(),
                        "_trunk_sha": trunk_sha,
                    }
                    results.append(run)
                    net = cal_stats.get("net_nodes_saved", 0)
                    safe_flag = "SAFE" if cal_stats.get("unsafe_activations", 1) == 0 else "UNSAFE"
                    print(f"thr={threshold} net={net:.1f} {safe_flag}")
    return results


def select_best(results: list[dict]) -> dict:
    """Select best run using calibration expected net savings only."""
    feasible = [r for r in results if r["cal_selection_stats"].get("feasible", False)
                and r["cal_selection_stats"].get("unsafe_activations", 1) == 0
                and r["cal_selection_stats"].get("net_nodes_saved", 0) > 0]
    if not feasible:
        # Fall back to any feasible (unsafe==0), then any
        feasible = [r for r in results
                    if r["cal_selection_stats"].get("unsafe_activations", 1) == 0]
    if not feasible:
        feasible = results
    return max(feasible,
               key=lambda r: (r["cal_selection_stats"].get("net_nodes_saved", 0),
                               r["cal_selection_stats"].get("precision_wilson_lower_95",
                                   r["cal_selection_stats"].get("wilson_lower_95", 0))))


# ── analysis helpers ──────────────────────────────────────────────────────────

def unsafe_case_analysis(rows: list[dict]) -> list[dict]:
    return [
        {
            "parent_hash": r.get("parent_hash"),
            "child_hash": r.get("child_hash"),
            "move": r.get("move"),
            "position_ply": r.get("position_ply"),
            "search_ply": r.get("search_ply"),
            "move_index": r.get("move_index"),
            "depth": r.get("depth"),
            "base_reduction": r.get("base_reduction"),
            # Top-level fields from classify_pair
            "baseline_nodes": r.get("baseline_nodes"),
            "counterfactual_nodes": r.get("counterfactual_nodes"),
            "net_nodes_saved": r.get("net_nodes_saved"),
            # Nested final decision records from pipeline_decision()
            "baseline_bound": r.get("baseline_final", {}).get("bound"),
            "counterfactual_bound": r.get("counterfactual_final", {}).get("bound"),
            "baseline_score": r.get("baseline_final", {}).get("score"),
            "counterfactual_score": r.get("counterfactual_final", {}).get("score"),
            "verification_triggered_baseline": r.get("baseline_final", {}).get("verification_triggered"),
            "verification_triggered_cf": r.get("counterfactual_final", {}).get("verification_triggered"),
            "move_class": r.get("move_class"),
            "source": r.get("source"),
            "status_reason": r.get("status_reason"),
        }
        for r in rows if r.get("sample_status") == "UNSAFE"
    ]


def distribution_summary(rows: list[dict]) -> dict:
    def bucket_mi(mi):
        if mi <= 7: return "4-7"
        if mi <= 11: return "8-11"
        if mi <= 19: return "12-19"
        if mi <= 39: return "20-39"
        if mi <= 69: return "40-69"
        return "70+"

    def bucket_nodes(n):
        if n <= 1: return "1"
        if n <= 5: return "2-5"
        if n <= 15: return "6-15"
        if n <= 50: return "16-50"
        if n <= 200: return "51-200"
        return "200+"

    buckets: dict = {}
    for r in rows:
        mi = r.get("move_index", 0)
        br = r.get("base_reduction", 0)
        depth = r.get("depth", 0)
        bl_nodes = r.get("baseline_nodes") or 1
        key_mi = bucket_mi(mi)
        key_br = str(br)
        key_d = str(depth)
        key_nodes = bucket_nodes(bl_nodes)
        for group_key, val in [("move_index", key_mi), ("base_reduction", key_br),
                                 ("depth", key_d), ("baseline_nodes", key_nodes)]:
            if group_key not in buckets:
                buckets[group_key] = {}
            if val not in buckets[group_key]:
                buckets[group_key][val] = {"n": 0, "pos": 0, "unsafe": 0}
            buckets[group_key][val]["n"] += 1
            if r.get("activate_plus_one"):
                buckets[group_key][val]["pos"] += 1
            if r.get("sample_status") == "UNSAFE":
                buckets[group_key][val]["unsafe"] += 1
    return buckets


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--natural", required=True)
    parser.add_argument("--stratified", required=True)
    parser.add_argument("--out-dir", default=str(SIDECAR_OUT_DIR))
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--debug-probs", action="store_true",
                        help="Train one model and print calibration probability distribution, then exit.")
    args = parser.parse_args()

    nat_path = Path(args.natural)
    strat_path = Path(args.stratified)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Loading data ===")
    nat_rows = load_file(nat_path)
    strat_rows = load_file(strat_path)
    print(f"  natural: {len(nat_rows)} rows")
    print(f"  stratified: {len(strat_rows)} rows")

    trunk_sha = hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()
    print(f"  trunk sha256: {trunk_sha[:16]}...")

    print("\n=== Integrity checks ===")
    errors = integrity_check(nat_rows, strat_rows, trunk_sha)
    warnings = [e for e in errors if e.startswith("WARN:")]
    hard_errors = [e for e in errors if not e.startswith("WARN:")]
    for w in warnings:
        print(f"  {w}")
    if hard_errors:
        for e in hard_errors:
            print(f"  ERROR: {e}", file=sys.stderr)
        return 1
    print(f"  {len(hard_errors)} errors, {len(warnings)} warnings — OK")

    # Remove UNKNOWN from all sets
    nat_known = [r for r in nat_rows if r["sample_status"] != "UNKNOWN"]
    strat_known = [r for r in strat_rows if r["sample_status"] != "UNKNOWN"]

    # Dedup stratified against natural
    strat_deduped = dedup_stratified(nat_known, strat_known)
    print(f"  Stratified after dedup: {len(strat_deduped)} rows (was {len(strat_known)})")

    nat_train = [r for r in nat_known if r.get("split") == "train"]
    nat_cal   = [r for r in nat_known if r.get("split") == "calibration"]
    nat_test  = [r for r in nat_known if r.get("split") == "final_test"]
    # Stratified: only "train" split rows — same partition hash means a stratified
    # row whose game_key hashes to "final_test" or "calibration" could leak test-game
    # signal even though it's a different population sample.
    strat_train = [r for r in strat_deduped if r.get("split") == "train"]

    print(f"\n  natural train:  {len(nat_train)} rows, "
          f"{sum(r['activate_plus_one'] for r in nat_train)} pos, "
          f"{sum(r['sample_status']=='UNSAFE' for r in nat_train)} unsafe")
    print(f"  natural cal:    {len(nat_cal)} rows, "
          f"{sum(r['activate_plus_one'] for r in nat_cal)} pos, "
          f"{sum(r['sample_status']=='UNSAFE' for r in nat_cal)} unsafe")
    print(f"  natural test:   {len(nat_test)} rows (SEALED)")
    print(f"  stratified:     {len(strat_train)} rows, "
          f"{sum(r['activate_plus_one'] for r in strat_train)} pos")

    # Distribution analysis
    print("\n=== Distribution analysis ===")
    nat_dist = distribution_summary(nat_known)
    strat_dist = distribution_summary(strat_deduped)
    for group in ("move_index", "base_reduction", "depth"):
        nat_g = nat_dist.get(group, {})
        print(f"  {group} (natural): " +
              ", ".join(f"{k}:{v['n']}(pos={v['pos']})" for k, v in sorted(nat_g.items())))

    # Unsafe case analysis
    print("\n=== Unsafe case analysis ===")
    nat_unsafe = [r for r in nat_known if r["sample_status"] == "UNSAFE"]
    strat_unsafe = [r for r in strat_deduped if r["sample_status"] == "UNSAFE"]
    nat_unsafe_cases = unsafe_case_analysis(nat_unsafe)
    print(f"  natural UNSAFE: {len(nat_unsafe_cases)}")
    for case in nat_unsafe_cases:
        print(f"    ply={case['position_ply']} move={case['move']} mi={case['move_index']} "
              f"depth={case['depth']} br={case['base_reduction']} "
              f"bl_nodes={case['baseline_nodes']} cf_nodes={case['counterfactual_nodes']} "
              f"bl_bound={case['baseline_bound']} cf_bound={case['counterfactual_bound']}")
    print(f"  stratified UNSAFE: {len(strat_unsafe)}")

    (out_dir / "unsafe_cases.json").write_text(
        json.dumps({"natural": nat_unsafe_cases, "stratified_count": len(strat_unsafe)}, indent=2),
        encoding="utf-8")

    if args.dry_run:
        print("\n[dry-run] Stopping before training.")
        return 0

    if args.debug_probs:
        print("\n=== Debug: training one seed, printing calibration probability distribution ===")
        train_set_dbg = mix_train(nat_train, strat_train, 0.50, 42)
        model_dbg, scale_dbg, shift_dbg, thr_dbg, cs_dbg = train_one(
            train_set_dbg, nat_cal, seed=42, variant="A",
            neg_weight=2.0, unsafe_weight=20.0, epochs=args.epochs, lr=args.lr)
        x_dbg, y_dbg, u_dbg = to_tensors(nat_cal, "A")
        with torch.no_grad():
            logits_dbg = model_dbg(x_dbg).squeeze(1)
            raw_probs = torch.sigmoid(logits_dbg)
            cal_probs = torch.sigmoid(scale_dbg * logits_dbg + shift_dbg)
        print(f"  Platt scale={scale_dbg:.4f}, shift={shift_dbg:.4f}")
        print(f"  Raw sigmoid: min={raw_probs.min():.4f} mean={raw_probs.mean():.4f} "
              f"max={raw_probs.max():.4f} >0.1={int((raw_probs>0.1).sum())}/{len(nat_cal)}")
        print(f"  Calibrated:  min={cal_probs.min():.4f} mean={cal_probs.mean():.4f} "
              f"max={cal_probs.max():.4f} >0.1={int((cal_probs>0.1).sum())}/{len(nat_cal)}")
        print("\n  Positive examples (activate_plus_one=True) calibrated probs:")
        pos_p = [(float(cal_probs[i]), nat_cal[i].get("net_nodes_saved",0), nat_cal[i].get("sample_status"))
                 for i in range(len(nat_cal)) if nat_cal[i]["activate_plus_one"]]
        for p, ns, st in sorted(pos_p, reverse=True):
            print(f"    p={p:.4f}  net_saved={ns:+d}  status={st}")
        print("\n  Unsafe examples calibrated probs:")
        un_p = [(float(cal_probs[i]), nat_cal[i].get("move"))
                for i in range(len(nat_cal)) if nat_cal[i]["sample_status"] == "UNSAFE"]
        for p, mv in un_p:
            print(f"    p={p:.4f}  move={mv}")
        print(f"\n  Selected threshold={thr_dbg}, cal_stats={cs_dbg}")
        return 0

    # === Main sweep: Variant A ===
    print("\n=== Main sweep: Variant A (hidden32 + context5) ===")
    A_results = run_sweep(
        nat_train, nat_cal, nat_test, strat_train, trunk_sha,
        variant="A",
        mixing_ratios=[0.50, 0.33, 0.67],
        neg_weights=[2.0, 5.0],
        unsafe_weights=[20.0, 50.0, 100.0],
        seeds=SEEDS,
        epochs=args.epochs,
        lr=args.lr,
        label="A",
    )

    # Select best A config using calibration ONLY
    best_A = select_best(A_results)
    frozen_config = {
        "variant": "A",
        "ratio_nat": best_A["ratio_nat"],
        "neg_weight": best_A["neg_weight"],
        "unsafe_weight": best_A["unsafe_weight"],
        "epochs": args.epochs,
        "lr": args.lr,
    }
    print(f"\n=== Best A config (calibration) ===")
    print(f"  ratio_nat={frozen_config['ratio_nat']:.0%} "
          f"neg_w={frozen_config['neg_weight']} uw={frozen_config['unsafe_weight']} "
          f"seed={best_A['seed']}")
    print(f"  threshold={best_A['threshold']} "
          f"cal_net_saved={best_A['cal_selection_stats'].get('net_nodes_saved', 0):.1f}")

    # Per-seed summary for A at best config
    A_best_config_runs = [
        r for r in A_results
        if r["ratio_nat"] == frozen_config["ratio_nat"]
        and r["neg_weight"] == frozen_config["neg_weight"]
        and r["unsafe_weight"] == frozen_config["unsafe_weight"]
    ]
    print(f"\n=== Variant A seed stability (best config, {len(A_best_config_runs)} seeds) ===")
    for r in sorted(A_best_config_runs, key=lambda x: x["seed"]):
        net = r["cal_selection_stats"].get("net_nodes_saved", 0)
        uu = r["cal_selection_stats"].get("unsafe_activations", 0)
        print(f"  seed={r['seed']:6d} thr={r['threshold']} net={net:6.1f} "
              f"act={r['cal_eval'].get('activations',0):3d} unsafe={uu}")

    # === Ablation: variants B, C, D at best A config ===
    print(f"\n=== Ablations (B, C, D) at frozen config ===")
    ablation_results = {"A": A_best_config_runs}
    for var in ("B", "C", "D"):
        print(f"  Variant {var}...")
        var_results = run_sweep(
            nat_train, nat_cal, nat_test, strat_train, trunk_sha,
            variant=var,
            mixing_ratios=[frozen_config["ratio_nat"]],
            neg_weights=[frozen_config["neg_weight"]],
            unsafe_weights=[frozen_config["unsafe_weight"]],
            seeds=SEEDS,
            epochs=args.epochs,
            lr=args.lr,
            label=var,
        )
        ablation_results[var] = var_results

    # Ablation summary
    print("\n=== Ablation summary (calibration net savings, median over seeds) ===")
    import statistics
    for var in ("A", "B", "C", "D"):
        nets = [r["cal_selection_stats"].get("net_nodes_saved", 0)
                for r in ablation_results[var]
                if r["cal_selection_stats"].get("unsafe_activations", 1) == 0]
        thresholds = [r["threshold"] for r in ablation_results[var]]
        acts = [r["cal_eval"].get("activations", 0) for r in ablation_results[var]]
        if nets:
            med_net = statistics.median(nets)
            med_act = statistics.median(acts)
            print(f"  {var}: median_cal_net={med_net:.1f} median_act={med_act:.0f} "
                  f"feasible={sum(1 for n in nets if n > 0)}/{len(ablation_results[var])} seeds")
        else:
            print(f"  {var}: no feasible runs")

    # === Freeze ===
    # Config is frozen: variant=A, best mixing ratio, neg_weight, unsafe_weight
    # Now re-train the specific best seed to get the model for artifact writing
    print(f"\n=== Producing frozen artifact from best A seed ===")
    best_seed = best_A["seed"]
    train_set_frozen = mix_train(nat_train, strat_train, frozen_config["ratio_nat"], best_seed)
    model_frozen, scale_frozen, shift_frozen, threshold_frozen, _ = train_one(
        train_set_frozen, nat_cal,
        seed=best_seed,
        variant="A",
        neg_weight=frozen_config["neg_weight"],
        unsafe_weight=frozen_config["unsafe_weight"],
        epochs=args.epochs,
        lr=args.lr,
    )

    # === Coefficient inspection (frozen model) ===
    print(f"\n=== Coefficient inspection (frozen seed={best_seed}) ===")
    w_frozen = model_frozen.weight.detach().cpu().flatten().tolist()
    b_frozen = float(model_frozen.bias.detach().cpu())
    h32_w = w_frozen[:32]
    c5_w  = w_frozen[32:]
    c5_names = ["remaining_depth", "move_index", "base_reduction", "is_horizontal", "is_vertical"]
    print(f"  bias={b_frozen:.4f}  (Platt: scale={scale_frozen:.4f} shift={shift_frozen:.4f})")
    print(f"  hidden32 weight norm={sum(x**2 for x in h32_w)**0.5:.4f}  "
          f"max={max(abs(x) for x in h32_w):.4f}  "
          f"mean_abs={sum(abs(x) for x in h32_w)/32:.4f}")
    print("  context5 weights:")
    for name, wt in zip(c5_names, c5_w):
        print(f"    {name:<22} {wt:+.6f}")

    # Per-seed coefficient variance
    A_best_config_runs_coeff = [
        r for r in A_results
        if r["ratio_nat"] == frozen_config["ratio_nat"]
        and r["neg_weight"] == frozen_config["neg_weight"]
        and r["unsafe_weight"] == frozen_config["unsafe_weight"]
    ]
    per_seed_coeff: list[list[float]] = []
    for r in A_best_config_runs_coeff:
        dims = VARIANTS["A"]["dims"]
        m_ = nn.Linear(dims, 1)
        m_.load_state_dict(r["_model_state"])
        per_seed_coeff.append(m_.weight.detach().cpu().flatten().tolist())
    if per_seed_coeff:
        import statistics as _stat
        print(f"\n  Context5 weight variance across {len(per_seed_coeff)} seeds:")
        for j, name in enumerate(c5_names):
            vals = [wv[32 + j] for wv in per_seed_coeff]
            print(f"    {name:<22} mean={_stat.mean(vals):+.4f}  "
                  f"stdev={_stat.stdev(vals):.4f}  "
                  f"min={min(vals):+.4f}  max={max(vals):+.4f}")

    # Full calibration threshold sweep for reporting
    cal_sweep = full_threshold_sweep(model_frozen, nat_cal, scale_frozen, shift_frozen, "A")
    print("\n  Calibration threshold sweep:")
    print(f"  {'thr':6} {'act':5} {'tp':5} {'unsafe':7} {'prec':7} {'wl95':7} "
          f"{'gross':7} {'net':7} {'act_rate':9}")
    for row in cal_sweep:
        print(f"  {row['threshold']:.3f} {row['activations']:5d} {row['true_positives']:5d} "
              f"{row['unsafe_activations']:7d} {row['precision']:7.4f} {row['wilson_lower_95']:7.4f} "
              f"{row['gross_nodes_saved']:7d} {row['net_nodes_saved']:7.2f} {row['activation_rate']:9.4f}")

    # === Final test — open ONCE, now that everything is frozen ===
    print(f"\n=== FINAL TEST (opened once) ===")
    test_eval = evaluate_at_threshold(
        model_frozen, nat_test, scale_frozen, shift_frozen, threshold_frozen, "A")
    test_sweep = full_threshold_sweep(model_frozen, nat_test, scale_frozen, shift_frozen, "A")
    print(f"  threshold={threshold_frozen}")
    print(f"  rows={test_eval['rows']} positives={test_eval['positives']}")
    print(f"  activations={test_eval['activations']} TP={test_eval['true_positives']} "
          f"unsafe={test_eval['unsafe_activations']}")
    print(f"  precision={test_eval['precision']:.4f} (wl95={test_eval['precision_wilson_lower_95']:.4f})")
    print(f"  recall={test_eval['recall']:.4f}")
    print(f"  gross_saved={test_eval['gross_nodes_saved']} "
          f"net_saved={test_eval['net_nodes_saved']:.2f}")
    print(f"  unsafe_rate={test_eval['unsafe_rate']:.4f}")

    # Profitability
    total_eligible_est = test_eval["rows"]  # calibration rows are eligible events
    total_inference_cost_if_deployed = INFERENCE_COST_NODES * total_eligible_est
    print(f"\n  Profitability (at threshold {threshold_frozen}):")
    print(f"    total eligible events in test: {total_eligible_est}")
    print(f"    inference cost (all eligible): {total_inference_cost_if_deployed:.1f} node-equiv")
    print(f"    gross savings from activations: {test_eval['gross_nodes_saved']} nodes")
    print(f"    net savings (gross - inference_cost_of_activations): {test_eval['net_nodes_saved']:.2f}")
    if test_eval['net_nodes_saved'] > 0 and test_eval['unsafe_activations'] == 0:
        verdict = "GO"
    elif test_eval['unsafe_activations'] > 0:
        verdict = "NO-GO (unsafe activations)"
    elif test_eval['net_nodes_saved'] <= 0:
        verdict = "NO-GO (not profitable)"
    else:
        verdict = "NO-GO"
    print(f"\n  FINAL TEST VERDICT: {verdict}")

    # Write binary artifact
    artifact_path = out_dir / "reduction_sidecar_v1.bin"
    payload_hash, sidecar_hash = write_binary(
        artifact_path, model_frozen, "A", trunk_sha,
        scale_frozen, shift_frozen, threshold_frozen,
    )
    print(f"\n  Written: {artifact_path}")
    print(f"  Sidecar SHA-256: {sidecar_hash}")

    # Trunk freeze proof
    trunk_after = hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()
    if trunk_sha != trunk_after:
        print("ERROR: trunk hash changed during training!", file=sys.stderr)
        return 1
    print(f"  Trunk freeze: unchanged ({trunk_sha[:16]}...)")

    # === Always-on baseline ===
    # Apply +1 to every eligible event: gross = sum of all net_nodes_saved, inference on all rows.
    always_on_tp_delta = sum(int(r.get("net_nodes_saved", 0)) for r in nat_test if r["activate_plus_one"])
    always_on_safe_fp_delta = sum(int(r.get("net_nodes_saved", 0)) for r in nat_test
                                   if r["sample_status"] == "SAFE" and not r["activate_plus_one"])
    always_on_unsafe_delta = sum(int(r.get("net_nodes_saved", 0)) for r in nat_test
                                  if r["sample_status"] == "UNSAFE")
    always_on_gross = always_on_tp_delta + always_on_safe_fp_delta + always_on_unsafe_delta
    always_on_net = always_on_gross - INFERENCE_COST_NODES * len(nat_test)
    always_on_n_unsafe = sum(1 for r in nat_test if r["sample_status"] == "UNSAFE")
    print(f"\n=== Always-on baseline (activate every eligible event) ===")
    print(f"  rows={len(nat_test)} unsafe={always_on_n_unsafe}")
    print(f"  tp_delta={always_on_tp_delta} safe_fp_delta={always_on_safe_fp_delta} "
          f"unsafe_delta={always_on_unsafe_delta}")
    print(f"  gross={always_on_gross} inference_cost={INFERENCE_COST_NODES*len(nat_test):.1f} "
          f"net={always_on_net:.2f}")
    always_on_verdict = "GO" if always_on_net > 0 and always_on_n_unsafe == 0 else "NO-GO"
    print(f"  Always-on verdict: {always_on_verdict}")

    # === Handcrafted rule baselines (test set) ===
    def rule_baseline(rows: list[dict], pred_fn) -> dict:
        n = len(rows)
        tp = sum(1 for r in rows if pred_fn(r) and r["activate_plus_one"])
        n_act = sum(1 for r in rows if pred_fn(r))
        n_unsafe = sum(1 for r in rows if pred_fn(r) and r["sample_status"] == "UNSAFE")
        tp_d = sum(int(r.get("net_nodes_saved", 0)) for r in rows if pred_fn(r) and r["activate_plus_one"])
        sfp_d = sum(int(r.get("net_nodes_saved", 0)) for r in rows
                    if pred_fn(r) and r["sample_status"] == "SAFE" and not r["activate_plus_one"])
        u_d = sum(int(r.get("net_nodes_saved", 0)) for r in rows
                  if pred_fn(r) and r["sample_status"] == "UNSAFE")
        gross = tp_d + sfp_d + u_d
        net = gross - INFERENCE_COST_NODES * n_act
        return {"activations": n_act, "true_positives": tp, "unsafe": n_unsafe,
                "tp_delta": tp_d, "safe_fp_delta": sfp_d, "unsafe_delta": u_d,
                "gross": gross, "net": round(net, 2),
                "precision": round(tp / n_act, 4) if n_act else 0.0}

    print(f"\n=== Handcrafted rule baselines (test set) ===")
    rules = {
        "mi>=12_d>=6": lambda r: int(r["move_index"]) >= 12 and int(r["depth"]) >= 6,
        "mi>=24_d>=6": lambda r: int(r["move_index"]) >= 24 and int(r["depth"]) >= 6,
        "red>=2":      lambda r: int(r["base_reduction"]) >= 2,
        "red==3":      lambda r: int(r["base_reduction"]) == 3,
        "mi>=32":      lambda r: int(r["move_index"]) >= 32,
    }
    for name, pred in rules.items():
        rb = rule_baseline(nat_test, pred)
        print(f"  {name:<20} act={rb['activations']:3d} tp={rb['true_positives']:3d} "
              f"unsafe={rb['unsafe']:2d} net={rb['net']:7.2f} prec={rb['precision']:.4f}")

    # === Ablation results (each variant uses ITS OWN calibrated threshold — fair comparison) ===
    print("\n=== Ablation final-test metrics (per-variant independent threshold) ===")
    abl_test_results = {}
    for var in ("A", "B", "C", "D"):
        best_var = select_best(ablation_results[var])
        dims = VARIANTS[var]["dims"]
        torch.manual_seed(best_var["seed"])
        m_abl = nn.Linear(dims, 1)
        m_abl.load_state_dict(best_var["_model_state"])
        # Use each variant's own independently calibrated threshold, not threshold_frozen.
        own_thr = best_var["threshold"]
        abl_test = evaluate_at_threshold(
            m_abl, nat_test,
            best_var["scale"], best_var["shift"], own_thr, var)
        abl_test_results[var] = abl_test
        print(f"  {var} (thr={own_thr}): act={abl_test['activations']} tp={abl_test['true_positives']} "
              f"unsafe={abl_test['unsafe_activations']} "
              f"net={abl_test['net_nodes_saved']:.2f} prec={abl_test['precision']:.4f}")

    # Save machine-readable summaries
    per_seed_csv = out_dir / "per_seed_metrics.csv"
    all_runs = A_results
    if all_runs:
        fields = ["variant", "ratio_nat", "neg_weight", "unsafe_weight", "seed", "train_rows",
                  "threshold", "cal_activations", "cal_tp", "cal_unsafe", "cal_net_saved"]
        with per_seed_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in all_runs:
                w.writerow({
                    "variant": r["variant"],
                    "ratio_nat": r["ratio_nat"],
                    "neg_weight": r["neg_weight"],
                    "unsafe_weight": r["unsafe_weight"],
                    "seed": r["seed"],
                    "train_rows": r["train_rows"],
                    "threshold": r["threshold"],
                    "cal_activations": r["cal_eval"].get("activations", 0),
                    "cal_tp": r["cal_eval"].get("true_positives", 0),
                    "cal_unsafe": r["cal_eval"].get("unsafe_activations", 0),
                    "cal_net_saved": r["cal_selection_stats"].get("net_nodes_saved", 0),
                })

    abl_csv = out_dir / "ablation_metrics.csv"
    with abl_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "seed", "own_threshold",
                                           "cal_net_saved", "test_net_saved",
                                           "test_activations", "test_unsafe"])
        w.writeheader()
        for var, runs in ablation_results.items():
            for r in runs:
                dims = VARIANTS[var]["dims"]
                m_ = nn.Linear(dims, 1)
                m_.load_state_dict(r["_model_state"])
                # Each variant evaluated at its own calibrated threshold (fair comparison).
                own_thr = r["threshold"]
                te = evaluate_at_threshold(m_, nat_test, r["scale"], r["shift"], own_thr, var)
                w.writerow({
                    "variant": var,
                    "seed": r["seed"],
                    "own_threshold": own_thr,
                    "cal_net_saved": r["cal_selection_stats"].get("net_nodes_saved", 0),
                    "test_net_saved": te["net_nodes_saved"],
                    "test_activations": te["activations"],
                    "test_unsafe": te["unsafe_activations"],
                })

    cal_sweep_path = out_dir / "cal_threshold_sweep.json"
    cal_sweep_path.write_text(json.dumps(cal_sweep, indent=2), encoding="utf-8")

    # === Feature-group sweep (Stage-2, requires v2 probe data) ===
    # v2 data has feature_schema == FEATURE_SCHEMA_V2 and carries history_score + total_legal_moves.
    v2_nat_train = [r for r in nat_train if r.get("feature_schema") == FEATURE_SCHEMA_V2]
    v2_nat_cal   = [r for r in nat_cal   if r.get("feature_schema") == FEATURE_SCHEMA_V2]
    v2_nat_test  = [r for r in nat_test  if r.get("feature_schema") == FEATURE_SCHEMA_V2]
    fg_results: dict[str, list[dict]] = {}
    if len(v2_nat_cal) >= 30:
        print(f"\n=== Feature-group sweep (v2 data: {len(v2_nat_train)} train / "
              f"{len(v2_nat_cal)} cal / {len(v2_nat_test)} test) ===")
        v2_strat_train = [r for r in strat_train if r.get("feature_schema") == FEATURE_SCHEMA_V2]
        for fg_var in ("P", "L", "PL", "PLO", "PLOB"):
            print(f"  Feature group {fg_var}...")
            fg_res = run_sweep(
                v2_nat_train, v2_nat_cal, v2_nat_test, v2_strat_train, trunk_sha,
                variant=fg_var,
                mixing_ratios=[frozen_config["ratio_nat"]],
                neg_weights=[frozen_config["neg_weight"]],
                unsafe_weights=[frozen_config["unsafe_weight"]],
                seeds=SEEDS,
                epochs=args.epochs,
                lr=args.lr,
                label=fg_var,
            )
            fg_results[fg_var] = fg_res

        import statistics as _stat2
        print(f"\n=== Feature-group summary (calibration, independently calibrated thresholds) ===")
        for fg_var, fg_res in fg_results.items():
            nets = [r["cal_selection_stats"].get("net_nodes_saved", 0) for r in fg_res
                    if r["cal_selection_stats"].get("unsafe_activations", 1) == 0]
            thrs = [r["threshold"] for r in fg_res]
            feasible = sum(1 for n in nets if n > 0)
            med = _stat2.median(nets) if nets else float("nan")
            med_thr = _stat2.median(thrs) if thrs else float("nan")
            print(f"  {fg_var:<6} median_cal_net={med:.1f}  med_threshold={med_thr:.3f}  "
                  f"feasible={feasible}/{len(fg_res)}")

        if v2_nat_test:
            print(f"\n=== Feature-group test evaluation (per-variant independent threshold) ===")
            for fg_var, fg_res in fg_results.items():
                best_fg = select_best(fg_res)
                dims = VARIANTS[fg_var]["dims"]
                m_fg = nn.Linear(dims, 1)
                m_fg.load_state_dict(best_fg["_model_state"])
                te_fg = evaluate_at_threshold(
                    m_fg, v2_nat_test,
                    best_fg["scale"], best_fg["shift"], best_fg["threshold"], fg_var)
                print(f"  {fg_var:<6} thr={best_fg['threshold']} "
                      f"act={te_fg['activations']} tp={te_fg['true_positives']} "
                      f"unsafe={te_fg['unsafe_activations']} "
                      f"net={te_fg['net_nodes_saved']:.2f} prec={te_fg['precision']:.4f}")
    else:
        print(f"\n[feature-group sweep skipped: only {len(v2_nat_cal)} v2 cal rows "
              f"(need >=30); recollect data with new engine build first]")

    final_report = {
        "trunk_sha256": trunk_sha,
        "trunk_frozen": trunk_sha == trunk_after,
        "frozen_config": frozen_config,
        "best_seed": best_seed,
        "threshold": threshold_frozen,
        "calibration_scale": scale_frozen,
        "calibration_shift": shift_frozen,
        "calibration_eval": evaluate_at_threshold(
            model_frozen, nat_cal, scale_frozen, shift_frozen, threshold_frozen, "A"),
        "calibration_sweep": cal_sweep,
        "final_test_eval": test_eval,
        "final_test_sweep": test_sweep,
        "final_test_verdict": verdict,
        "ablation_test_results": abl_test_results,
        "always_on_baseline": {
            "net": always_on_net, "gross": always_on_gross,
            "unsafe_activations": always_on_n_unsafe, "verdict": always_on_verdict,
        },
        "feature_group_results": {
            k: select_best(v)["cal_selection_stats"] for k, v in fg_results.items()
        } if fg_results else {},
        "unsafe_natural_cases": nat_unsafe_cases,
        "unsafe_stratified_count": len(strat_unsafe),
        "artifact_path": str(artifact_path),
        "sidecar_sha256": sidecar_hash,
        "runtime_enabled": False,
    }
    report_path = out_dir / "reduction_sidecar_v1_report.json"
    report_path.write_text(json.dumps(final_report, indent=2, default=str), encoding="utf-8")
    print(f"\n  Final report: {report_path}")

    print(f"\n=== FINAL VERDICT: {verdict} ===")
    print(f"  Runtime activation remains OFF")
    print(f"  Nothing was pushed")
    return 0 if verdict == "GO" else 3


if __name__ == "__main__":
    raise SystemExit(main())
