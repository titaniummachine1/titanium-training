#!/usr/bin/env python3
"""A/B race2w (all nodes) vs race2pv for one game's thinks; report two_wall stats + NPS."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def wall_hands(moves: list[str]) -> tuple[int, int, int]:
    """Return (wl0, wl1, sum) after applying moves from startpos."""
    wl = [10, 10]
    for i, mv in enumerate(moves):
        if mv[-1] in ("h", "v"):
            wl[i % 2] -= 1
    return wl[0], wl[1], wl[0] + wl[1]


def run_think(exe: Path, ms: int, moves: list[str], engine: str, env_base: dict) -> dict:
    env = dict(env_base)
    env["TITANIUM_BENCH_ENGINE"] = engine
    args = [str(exe), "think", "--ms", str(ms), "--full", "--threads", "1"]
    if moves:
        args += ["--moves", " ".join(moves)]
    proc = subprocess.run(
        args,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
    if not lines:
        return {"error": proc.stderr[-500:], "returncode": proc.returncode}
    primary = json.loads(lines[0])
    stats = None
    for ln in lines[1:]:
        try:
            obj = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if "two_wall_calls" in obj or "one_wall_calls" in obj:
            stats = obj
            break
    primary["_race_stats"] = stats
    return primary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thinks", required=True)
    ap.add_argument("--exe", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only-sum-wl", type=int, default=None, help="Only thinks with this hand sum")
    args = ap.parse_args()

    exe = Path(args.exe)
    rows = []
    env_base = os.environ.copy()
    env_base.pop("TITANIUM_ALLOW_SUBOPTIMAL", None)

    with open(args.thinks, encoding="utf-8") as f:
        thinks = [json.loads(ln) for ln in f if ln.strip()]

    engines = [
        ("race2w_all_nodes", "titanium-v17-race2w"),
        ("race2pv_only", "titanium-v17-race2pv"),
        ("race_off", "titanium-v17-race1w"),  # 1w on, 2w off — closer control than raw v15
    ]

    for t in thinks:
        moves = list(t["moves"])
        wl0, wl1, wsum = wall_hands(moves)
        if args.only_sum_wl is not None and wsum != args.only_sum_wl:
            continue
        ms = max(1, int(round(float(t["allotted_ms"]))))
        rec = {
            "game": t["game"],
            "ply": t["ply"],
            "side": t["side"],
            "wl": [wl0, wl1],
            "wl_sum": wsum,
            "allotted_ms": ms,
            "variants": {},
        }
        for label, eng in engines:
            r = run_think(exe, ms, moves, eng, env_base)
            stats = r.get("_race_stats") or {}
            rec["variants"][label] = {
                "engine": eng,
                "nps": r.get("nps"),
                "nodes": r.get("nodes"),
                "depth": r.get("depth"),
                "score": r.get("score"),
                "move": r.get("move"),
                "two_wall_calls": stats.get("two_wall_calls", 0),
                "two_wall_decisive": stats.get("two_wall_decisive", 0),
                "two_wall_unknown": stats.get("two_wall_unknown", 0),
                "one_wall_calls": stats.get("one_wall_calls", 0),
                "one_wall_decisive": stats.get("one_wall_decisive", 0),
                "engine_mode": r.get("engine_mode"),
            }
            print(
                f"ply={t['ply']} wl={wl0}+{wl1}={wsum} {label}: "
                f"nps={rec['variants'][label]['nps']} "
                f"2w={stats.get('two_wall_calls', 0)}/"
                f"{stats.get('two_wall_decisive', 0)}/"
                f"{stats.get('two_wall_unknown', 0)}",
                flush=True,
            )
        rows.append(rec)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "thinks": len(rows),
        "by_wl_sum": {},
        "totals": {},
    }
    for label, _ in engines:
        tw_c = sum(r["variants"][label]["two_wall_calls"] or 0 for r in rows)
        tw_d = sum(r["variants"][label]["two_wall_decisive"] or 0 for r in rows)
        tw_u = sum(r["variants"][label]["two_wall_unknown"] or 0 for r in rows)
        nps = [r["variants"][label]["nps"] for r in rows if r["variants"][label]["nps"]]
        summary["totals"][label] = {
            "two_wall_calls": tw_c,
            "two_wall_decisive": tw_d,
            "two_wall_unknown": tw_u,
            "mean_nps": round(sum(nps) / len(nps), 0) if nps else None,
            "median_nps": sorted(nps)[len(nps) // 2] if nps else None,
        }
    for r in rows:
        summary["by_wl_sum"].setdefault(str(r["wl_sum"]), 0)
        summary["by_wl_sum"][str(r["wl_sum"])] += 1

    payload = {"summary": summary, "rows": rows}
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
