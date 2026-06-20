#!/usr/bin/env python3
"""Report exclusive function and category samples from an Inferno flamegraph SVG."""

from __future__ import annotations

import argparse
import html
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


CATEGORIES = (
    (
        "BFF / wall legality",
        (
            "pbff_",
            "expand_wave",
            "wall_keeps_paths_open",
            "collect_wall_orientation",
            "generate_wall_moves_slice",
        ),
    ),
    (
        "geometric legal wall cache",
        ("geometric_legal_wall", "GeometricWallCache", "geometric_wall_len_cached"),
    ),
    (
        "DirMasks / route fields",
        (
            "DirMasks::",
            "AceGame::can_step",
            "refresh_dist",
            "fill_ace_dist",
            "flood_scatter",
            "shortest_route_bits",
            "route_feature_score",
            "expand_frontier",
        ),
    ),
    (
        "NNUE / eval",
        ("AceSearch::evaluate", "field_plane_contrib", "halfpw", "nnue"),
    ),
    (
        "TT",
        ("tt_grow", "tt_probe", "tt_store", "transposition", "cache_tier_bits"),
    ),
    (
        "progress / time checks",
        ("check_time", "emit_stream_progress", "emit_ace_progress", "postMessage"),
    ),
    (
        "search overhead",
        (
            "AceSearch::ab",
            "AceSearch::gen_moves",
            "AceSearch::order_moves",
            "AceSearch::think",
        ),
    ),
)

TITLE_RE = re.compile(r"^(.*) \(([0-9,]+) samples(?:, [^)]+)?\)$")


@dataclass(frozen=True)
class Frame:
    name: str
    x: int
    width: int
    y: int


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def namespaced_attr(element: ET.Element, name: str) -> str:
    for key, value in element.attrib.items():
        if key != name and local_name(key) == name:
            return value
    raise KeyError(name)


def read_frames(path: Path) -> tuple[int, list[Frame]]:
    root = ET.parse(path).getroot()
    frames_node = next(
        node for node in root.iter() if local_name(node.tag) == "svg" and node.get("id") == "frames"
    )
    total = int(frames_node.get("total_samples", "0"))
    frames: list[Frame] = []
    for group in frames_node:
        title = next((child for child in group if local_name(child.tag) == "title"), None)
        rect = next((child for child in group if local_name(child.tag) == "rect"), None)
        if title is None or rect is None or not title.text:
            continue
        match = TITLE_RE.match(html.unescape(title.text))
        if not match:
            continue
        frames.append(Frame(
            name=match.group(1),
            x=int(namespaced_attr(rect, "x")),
            width=int(namespaced_attr(rect, "w")),
            y=int(float(rect.get("y", "0"))),
        ))
    if total <= 0 or not frames:
        raise ValueError(f"{path} contains no Inferno frame samples")
    return total, frames


def exclusive_samples(frames: list[Frame]) -> dict[str, int]:
    by_y: dict[int, list[Frame]] = defaultdict(list)
    for frame in frames:
        by_y[frame.y].append(frame)
    result: dict[str, int] = defaultdict(int)
    for frame in frames:
        child_width = 0
        right = frame.x + frame.width
        for child in by_y.get(frame.y - 16, ()):
            if child.x >= frame.x and child.x + child.width <= right:
                child_width += child.width
        result[frame.name] += max(0, frame.width - child_width)
    return result


def category_for(name: str) -> str:
    short = name.split("`")[-1].casefold()
    for category, needles in CATEGORIES:
        if any(needle.casefold() in short for needle in needles):
            return category
    return "other"


def report(path: Path, top: int) -> None:
    total, frames = read_frames(path)
    exclusive = exclusive_samples(frames)
    grouped: dict[str, int] = defaultdict(int)
    for name, samples in exclusive.items():
        grouped[category_for(name)] += samples

    print(f"\n=== {path} ===")
    print(f"Samples: {total:,}")
    print("\nTop exclusive functions:")
    shown = 0
    for name, samples in sorted(exclusive.items(), key=lambda item: item[1], reverse=True):
        if samples <= 0 or name.startswith(("0x", "`0x")):
            continue
        print(f"  {samples:8,d}  {samples / total:6.2%}  {name.split('`')[-1]}")
        shown += 1
        if shown >= top:
            break

    print("\nExclusive grouped samples:")
    for category in [name for name, _ in CATEGORIES] + ["other"]:
        samples = grouped.get(category, 0)
        print(f"  {samples:8,d}  {samples / total:6.2%}  {category}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("svg", nargs="+", type=Path)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()
    for path in args.svg:
        report(path, args.top)


if __name__ == "__main__":
    main()
