#!/usr/bin/env python3
"""Compare zero-ink paired attention pressure with native alpha-beta pressure."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))

from collect_search_importance import probe, score_class, search_pressure_target, target_components


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs)
    dy = sum((y - my) ** 2 for y in ys)
    return num / math.sqrt(dx * dy) if dx > 0 and dy > 0 else float("nan")


def ranks(values: list[float]) -> list[float]:
    result = [0.0] * len(values)
    ordered = sorted(range(len(values)), key=values.__getitem__)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        rank = (start + end - 1) / 2.0
        for index in ordered[start:end]:
            result[index] = rank
        start = end
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zero", action="append", required=True, help="zero-search-budget JSONL")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--engine", default="titanium-v15")
    ap.add_argument("--shallow-depth", type=int, default=2)
    ap.add_argument("--deep-depth", type=int, default=5)
    ap.add_argument("--time", type=float, default=1.0)
    args = ap.parse_args()

    rows = []
    for path in args.zero:
        rows.extend(
            json.loads(line)
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    pairs: list[tuple[float, float]] = []
    skipped = 0
    for row in rows[:args.limit]:
        shallow = probe(row["moves"], args.shallow_depth, args.time, args.engine)
        deep = probe(row["moves"], args.deep_depth, args.time, args.engine)
        if (
            shallow.get("best") in (None, "(none)")
            or deep.get("best") in (None, "(none)")
            or score_class(shallow["score"])[0] != "cp"
            or score_class(deep["score"])[0] != "cp"
        ):
            skipped += 1
            continue
        native = search_pressure_target(target_components(shallow, deep))
        pairs.append((native, float(row["search_pressure"])))

    native = [x for x, _ in pairs]
    zero = [y for _, y in pairs]
    print(f"rows={len(rows[:args.limit])} usable={len(pairs)} skipped_overrides={skipped}")
    if len(pairs) < 2:
        return 1
    print(
        f"pearson={pearson(native, zero):+.4f} "
        f"spearman={pearson(ranks(native), ranks(zero)):+.4f} "
        f"native_mean={sum(native)/len(native):+.4f} zero_mean={sum(zero)/len(zero):+.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
