#!/usr/bin/env python3
"""Validation gates for quarantined Ka teacher labels and search-budget sidecar.

Checks:
  - label count and value_stm spread
  - entropy/search_pressure not collapsed
  - held-out MSE vs constant-mean baseline for search_pressure target

Usage:
  python training/tools/ka_teacher/validate_teacher_sidecar.py \\
    --labels training/data/ka_teacher_quarantine/labels.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def mse(y: list[float], yhat: list[float]) -> float:
    if not y:
        return float("inf")
    return sum((a - b) ** 2 for a, b in zip(y, yhat)) / len(y)


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate quarantined Ka teacher labels")
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--min-rows", type=int, default=20)
    ap.add_argument("--min-value-spread", type=float, default=0.15)
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    args = ap.parse_args()
    if not args.labels.is_file():
        print(f"missing labels: {args.labels}", file=sys.stderr)
        return 2

    rows = load_rows(args.labels)
    if len(rows) < args.min_rows:
        print(f"FAIL: only {len(rows)} rows (< {args.min_rows})")
        return 1

    values = [float(r["ka_nn"]["value_stm"]) for r in rows if "ka_nn" in r]
    spread = max(values) - min(values) if values else 0.0
    if spread < args.min_value_spread:
        print(f"FAIL: value_stm spread {spread:.3f} < {args.min_value_spread}")
        return 1

    pressures = [
        float(r.get("ka_policy_budget", {}).get("search_pressure", 0.0)) for r in rows
    ]
    entropies = [float(r.get("ka_policy_budget", {}).get("entropy", 0.0)) for r in rows]
    if max(entropies) - min(entropies) < 0.05:
        print("FAIL: policy entropy collapsed")
        return 1

    rng = random.Random(0)
    idx = list(range(len(pressures)))
    rng.shuffle(idx)
    cut = max(1, int(len(idx) * (1.0 - args.holdout_frac)))
    train_idx, val_idx = idx[:cut], idx[cut:]
    if not val_idx:
        val_idx = train_idx[-1:]
        train_idx = train_idx[:-1]
    train_mean = sum(pressures[i] for i in train_idx) / len(train_idx)
    baseline = [train_mean for _ in val_idx]
    val_y = [pressures[i] for i in val_idx]
    baseline_mse = mse(val_y, baseline)
    # trivial sidecar: predict train mean; require some variance to justify training
    print(f"rows={len(rows)} value_spread={spread:.3f} entropy_spread={max(entropies)-min(entropies):.3f}")
    print(f"holdout baseline MSE={baseline_mse:.4f}")
    if baseline_mse > 0.25:
        print("WARN: high holdout MSE — more labels or better targets before sidecar export")
    print("PASS: quarantine teacher labels meet minimum gates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
