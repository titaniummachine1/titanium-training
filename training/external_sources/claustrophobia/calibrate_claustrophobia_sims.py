#!/usr/bin/env python3
"""Cheap Claustrophobia sim calibration — SEPARATE from frozen clean_v1 20-sim bench.

Uses a small legal opening subset (not overwriting clean_v1).
Default: 5 paired games per sims level in {1,2,4,8,20}.

Selection metric is NOT Claustrophobia win rate alone — track unique positions /
disagreement roots / runtime. Run only AFTER clean_v1 finishes.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "training"))

# Reuse crossplay helpers by importing after path setup
EXT = Path(__file__).resolve().parent
sys.path.insert(0, str(EXT))

from crossplay_titanium_ladder import (  # noqa: E402
    DEFAULT_ENGINE,
    TitaniumSession,
    play_one,
)

OUT_ROOT = EXT / "eval_games" / "sim_calibration"
# Keep clean_v1 untouched — use first 5 openings from frozen manifest only.
MANIFEST = EXT / "frozen_openings" / "claustro_titanium_openings_v1.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--titanium-weights", type=Path, required=True)
    ap.add_argument("--titanium-bin", type=Path, default=DEFAULT_ENGINE)
    ap.add_argument("--label", default="epoch2_calib")
    ap.add_argument("--games-per-level", type=int, default=5)
    ap.add_argument("--sims-levels", default="1,2,4,8,20")
    ap.add_argument("--time-sec", type=float, default=1.0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    # Guard: never write into clean_v1
    out = OUT_ROOT / args.label
    out.mkdir(parents=True, exist_ok=True)
    (out / "DENYLIST.json").write_text(
        json.dumps(
            {
                "purpose": "sim_calibration_only",
                "do_not_overwrite_clean_v1": True,
                "do_not_train_on": True,
                "do_not_import_to_labels_db": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    openings = (manifest.get("openings") or [])[: args.games_per_level]
    levels = [int(x) for x in args.sims_levels.split(",") if x.strip()]

    summary_levels = []
    for sims in levels:
        level_dir = out / f"sims_{sims}"
        level_dir.mkdir(parents=True, exist_ok=True)
        ti = TitaniumSession(args.titanium_bin, args.titanium_weights, args.time_sec)
        rows = []
        t0 = time.time()
        try:
            with (level_dir / "results.jsonl").open("w", encoding="utf-8") as fh:
                for g, meta in enumerate(openings):
                    titanium_first = g % 2 == 0
                    row = play_one(
                        titanium_first=titanium_first,
                        opening=tuple(meta["moves"]),
                        sims=sims,
                        device=args.device,
                        ti=ti,
                    )
                    row["game_idx"] = g
                    row["opening_id"] = meta["opening_id"]
                    row["sims"] = sims
                    row["evaluation_only"] = True
                    if row.get("termination") == "PROTOCOL_ERROR":
                        print(f"HARNESS_FAIL sims={sims} game={g}: {row.get('error')}")
                        fh.write(json.dumps(row) + "\n")
                        return 3
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                    rows.append(row)
                    print(
                        f"sims={sims} game={g} winner={row.get('winner_side')} "
                        f"plies={row.get('plies')} sec={row.get('seconds', 0):.1f}"
                    )
        finally:
            ti.close()
        wall = time.time() - t0
        prefixes = []
        for r in rows:
            moves = r.get("moves") or []
            # harvest mid-game prefixes as crude unique-state proxies
            for k in (4, 8, 12, 16):
                if len(moves) >= k:
                    prefixes.append(tuple(moves[:k]))
        unique_pref = len(set(prefixes))
        ti_wins = sum(1 for r in rows if r.get("winner_side") == "titanium")
        level_sum = {
            "sims": sims,
            "games": len(rows),
            "protocol_errors": 0,
            "titanium_wins": ti_wins,
            "claustrophobia_wins": len(rows) - ti_wins,
            "avg_plies": sum(r.get("plies") or 0 for r in rows) / max(1, len(rows)),
            "wall_clock_sec": wall,
            "sec_per_game": wall / max(1, len(rows)),
            "unique_prefix_keys": unique_pref,
            "unique_prefixes_per_cpu_min": unique_pref / max(wall / 60.0, 1e-6),
        }
        (level_dir / "summary.json").write_text(json.dumps(level_sum, indent=2) + "\n", encoding="utf-8")
        summary_levels.append(level_sum)

    report = {
        "purpose": "choose_cheapest_useful_claustrophobia_sims",
        "preserve_clean_v1_20sim_benchmark": True,
        "likely_bulk_range": "1-4",
        "keep_20_for_occasional_upper_bound": True,
        "levels": summary_levels,
        "selection_note": (
            "Prefer lowest sims with stable protocol and good unique_prefixes_per_cpu_min; "
            "do not select by Claustrophobia win rate alone."
        ),
    }
    (out / "calibration_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
