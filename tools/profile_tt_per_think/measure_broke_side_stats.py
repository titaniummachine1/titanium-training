#!/usr/bin/env python3
"""Replay thinks with one side broke; aggregate broke_* race stats from search_bench."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def wall_hands(moves: list[str]) -> tuple[int, int]:
    wl = [10, 10]
    for i, mv in enumerate(moves):
        if mv[-1] in ("h", "v"):
            wl[i % 2] -= 1
    return wl[0], wl[1]


def main() -> int:
    thinks_path = Path(sys.argv[1])
    exe = Path(sys.argv[2])
    out_path = Path(sys.argv[3])
    max_ms = int(sys.argv[4]) if len(sys.argv) > 4 else 800

    rows = []
    totals = {
        "thinks_total": 0,
        "thinks_one_side_broke": 0,
        "broke_calls": 0,
        "broke_decisive": 0,
        "broke_unknown": 0,
        "broke_lower": 0,
        "broke_upper": 0,
        "broke_cut_fail_high": 0,
        "broke_cut_fail_low": 0,
        "nodes_sum": 0,
        "nps_sum": 0.0,
        "nps_n": 0,
    }

    env = os.environ.copy()
    env.pop("TITANIUM_ALLOW_SUBOPTIMAL", None)

    for ln in thinks_path.read_text(encoding="utf-8-sig").splitlines():
        if not ln.strip():
            continue
        t = json.loads(ln)
        totals["thinks_total"] += 1
        w0, w1 = wall_hands(t["moves"])
        if (w0 == 0) == (w1 == 0):
            continue
        totals["thinks_one_side_broke"] += 1
        ms = max(1, min(max_ms, int(round(float(t["allotted_ms"])))))
        args = [str(exe), "think", "--ms", str(ms), "--full", "--threads", "1"]
        if t["moves"]:
            args += ["--moves", " ".join(t["moves"])]
        proc = subprocess.run(args, capture_output=True, text=True, env=env, check=False)
        lines = [x for x in proc.stdout.splitlines() if x.startswith("{")]
        if not lines:
            continue
        primary = json.loads(lines[0])
        stats = {}
        for x in lines[1:]:
            try:
                obj = json.loads(x)
            except json.JSONDecodeError:
                continue
            if "broke_calls" in obj:
                stats = obj
                break
        for k in (
            "broke_calls",
            "broke_decisive",
            "broke_unknown",
            "broke_lower",
            "broke_upper",
            "broke_cut_fail_high",
            "broke_cut_fail_low",
        ):
            totals[k] += int(stats.get(k, 0) or 0)
        nodes = int(primary.get("nodes") or 0)
        nps = primary.get("nps")
        totals["nodes_sum"] += nodes
        if nps is not None:
            totals["nps_sum"] += float(nps)
            totals["nps_n"] += 1
        rows.append(
            {
                "game": t["game"],
                "ply": t["ply"],
                "wl": [w0, w1],
                "ms": ms,
                "nodes": nodes,
                "nps": nps,
                "stats": {k: stats.get(k, 0) for k in (
                    "broke_calls",
                    "broke_decisive",
                    "broke_unknown",
                    "broke_lower",
                    "broke_upper",
                    "broke_cut_fail_high",
                    "broke_cut_fail_low",
                )},
            }
        )
        print(
            f"g{t['game']} ply{t['ply']} wl={w0}/{w1} "
            f"calls={stats.get('broke_calls', 0)} "
            f"dec={stats.get('broke_decisive', 0)} "
            f"fh={stats.get('broke_cut_fail_high', 0)} "
            f"fl={stats.get('broke_cut_fail_low', 0)} "
            f"nps={nps}",
            flush=True,
        )

    if totals["nps_n"]:
        totals["mean_nps"] = round(totals["nps_sum"] / totals["nps_n"], 0)
    else:
        totals["mean_nps"] = None
    if totals["broke_calls"]:
        totals["decisive_rate_pct"] = round(
            100.0 * totals["broke_decisive"] / totals["broke_calls"], 2
        )
        totals["cut_rate_pct"] = round(
            100.0
            * (totals["broke_cut_fail_high"] + totals["broke_cut_fail_low"])
            / totals["broke_calls"],
            2,
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"summary": totals, "rows": rows}, indent=2), encoding="utf-8"
    )
    print(json.dumps(totals, indent=2))
    print(f"wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
