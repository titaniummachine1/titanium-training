#!/usr/bin/env python3
"""Aggregate per-think flamegraphs + thinks.jsonl into ply averages."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_PARSER = _REPO / "training" / "tools" / "analysis" / "parse_flamegraph.py"
if str(_PARSER.parent) not in sys.path:
    sys.path.insert(0, str(_PARSER.parent))

from parse_flamegraph import CATEGORIES, exclusive_samples, read_frames  # noqa: E402


def categorize(name: str) -> str:
    for label, needles in CATEGORIES:
        if any(n in name for n in needles):
            return label
    return "other"


def load_thinks(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    out_dir = args.out_dir
    thinks_path = out_dir / "thinks.jsonl"
    fg_dir = out_dir / "flamegraphs"
    if not thinks_path.is_file():
        print(f"missing {thinks_path}", file=sys.stderr)
        return 2

    thinks = load_thinks(thinks_path)
    by_ply: dict[int, list[dict]] = defaultdict(list)
    for t in thinks:
        by_ply[int(t["ply"])].append(t)

    ply_timing = {}
    for ply, rows in sorted(by_ply.items()):
        ply_timing[str(ply)] = {
            "count": len(rows),
            "mean_allotted_ms": sum(r["allotted_ms"] for r in rows) / len(rows),
            "mean_used_ms": sum(r["used_ms"] for r in rows) / len(rows),
            "median_used_ms": sorted(r["used_ms"] for r in rows)[len(rows) // 2],
        }

    buckets = {
        "0-9": range(0, 10),
        "10-19": range(10, 20),
        "20-29": range(20, 30),
        "30-49": range(30, 50),
        "50+": range(50, 10_000),
    }
    bucket_timing = {}
    for label, rng in buckets.items():
        rows = [t for t in thinks if int(t["ply"]) in rng]
        if not rows:
            continue
        bucket_timing[label] = {
            "count": len(rows),
            "mean_allotted_ms": sum(r["allotted_ms"] for r in rows) / len(rows),
            "mean_used_ms": sum(r["used_ms"] for r in rows) / len(rows),
        }

    # Flamegraph exclusive samples aggregated by ply + category.
    ply_cats: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    ply_frames: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    ply_fg_count: dict[str, int] = defaultdict(int)
    svg_errors: list[str] = []

    if fg_dir.is_dir():
        for svg in sorted(fg_dir.glob("g*_ply*_s*.svg")):
            # g000_ply012_s0.svg
            parts = svg.stem.split("_")
            try:
                ply = int(parts[1].replace("ply", ""))
            except (IndexError, ValueError):
                continue
            try:
                total, frames = read_frames(svg)
                excl = exclusive_samples(frames)
            except Exception as exc:  # noqa: BLE001
                svg_errors.append(f"{svg.name}: {exc}")
                continue
            if total <= 0:
                continue
            ply_key = str(ply)
            ply_fg_count[ply_key] += 1
            for name, samples in excl.items():
                share = samples / total
                ply_frames[ply_key][name] += share
                ply_cats[ply_key][categorize(name)] += share

    # Average shares across SVGs of the same ply.
    ply_cat_avg = {}
    ply_frame_top = {}
    for ply_key, n in ply_fg_count.items():
        cats = {k: v / n for k, v in ply_cats[ply_key].items()}
        ply_cat_avg[ply_key] = dict(sorted(cats.items(), key=lambda kv: -kv[1]))
        frames = {k: v / n for k, v in ply_frames[ply_key].items()}
        top = sorted(frames.items(), key=lambda kv: -kv[1])[: args.top]
        ply_frame_top[ply_key] = [{ "name": n, "mean_exclusive_share": s} for n, s in top]

    # Bucket category averages.
    bucket_cats = {}
    for label, rng in buckets.items():
        shares: dict[str, float] = defaultdict(float)
        n = 0
        for ply in rng:
            key = str(ply)
            if key not in ply_cat_avg:
                continue
            n += 1
            for cat, share in ply_cat_avg[key].items():
                shares[cat] += share
        if n == 0:
            continue
        bucket_cats[label] = {
            k: v / n for k, v in sorted(shares.items(), key=lambda kv: -kv[1])
        }

    summary = {
        "thinks": len(thinks),
        "games": len({t["game"] for t in thinks}),
        "flamegraphs_parsed": sum(ply_fg_count.values()),
        "svg_errors": svg_errors[:20],
        "timing_by_ply": ply_timing,
        "timing_by_bucket": bucket_timing,
        "categories_by_ply": ply_cat_avg,
        "categories_by_bucket": bucket_cats,
        "top_frames_by_ply": ply_frame_top,
    }

    json_path = out_dir / "ply_summary.json"
    txt_path = out_dir / "ply_summary.txt"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        f"thinks={summary['thinks']} games={summary['games']} "
        f"flamegraphs_parsed={summary['flamegraphs_parsed']}",
        "",
        "=== timing by ply bucket ===",
    ]
    for label, row in bucket_timing.items():
        lines.append(
            f"  {label}: n={row['count']} mean_allotted={row['mean_allotted_ms']:.0f}ms "
            f"mean_used={row['mean_used_ms']:.0f}ms"
        )
    lines.append("")
    lines.append("=== category exclusive share by ply bucket ===")
    for label, cats in bucket_cats.items():
        top = ", ".join(f"{k}={v*100:.1f}%" for k, v in list(cats.items())[:8])
        lines.append(f"  {label}: {top}")
    lines.append("")
    lines.append("=== top frames by ply (first 15 plies with SVGs) ===")
    shown = 0
    for ply_key in sorted(ply_frame_top, key=int):
        if shown >= 15:
            break
        frames = ply_frame_top[ply_key][:5]
        if not frames:
            continue
        lines.append(f"  ply {ply_key}:")
        for fr in frames:
            lines.append(f"    {fr['mean_exclusive_share']*100:5.2f}%  {fr['name']}")
        shown += 1

    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(txt_path.read_text(encoding="utf-8"))
    print(f"wrote {json_path}")
    print(f"wrote {txt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
