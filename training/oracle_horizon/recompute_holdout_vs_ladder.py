"""Recompute holdout move metrics against proven ladder moves (not terminal oracle move)."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "training/runs/oracle_horizon_pilot_v1/continuation_e3/eval"
HOLDOUT = ROOT / "training/runs/oracle_horizon_pilot_v1/continuation_e3/holdout_labels.jsonl"
OUT = EVAL / "diagnostics" if (EVAL / "diagnostics").is_dir() else ROOT / "training/runs/oracle_horizon_pilot_v1/continuation_e3/diagnostics"


def main() -> None:
    holdout = [json.loads(l) for l in HOLDOUT.read_text(encoding="utf-8").splitlines() if l.strip()]
    proof = [
        json.loads(l)
        for l in (EVAL / "HOLDOUT_PROOF_HORIZON.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    ladder_move: dict[str, str] = {}
    terminal_vs_ladder = 0
    with_ladder = 0
    for r in holdout:
        proven = [x for x in (r.get("ladder") or []) if x.get("proven")]
        if proven:
            with_ladder += 1
            lm = proven[-1]["selected_move"]
            ladder_move[r["packed_state_hex"]] = lm
            if r.get("best_move") != lm:
                terminal_vs_ladder += 1
        else:
            ladder_move[r["packed_state_hex"]] = r.get("best_move")

    stats: dict = defaultdict(lambda: defaultdict(lambda: {"n": 0, "match_any": 0, "match_50k": 0, "first_nodes": []}))
    pairwise_diffs = 0
    by_hex: dict[str, dict] = defaultdict(dict)
    for r in proof:
        hx = r["packed_state_hex"]
        lab = r["label"]
        band = str(r["band"])
        target = ladder_move.get(hx)
        by_hex[hx][lab] = [st.get("selected_move") for st in r["stages"]]
        s = stats[lab][band]
        s["n"] += 1
        first = None
        for st in r["stages"]:
            if st.get("selected_move") == target:
                if first is None:
                    first = st["nodes"]
                if st["nodes"] == 50_000:
                    s["match_50k"] += 1
        if first is not None:
            s["match_any"] += 1
            s["first_nodes"].append(first)

    for hx, nets in by_hex.items():
        if set(nets) == {"parent", "raw", "ema"} and not (nets["parent"] == nets["raw"] == nets["ema"]):
            pairwise_diffs += 1

    report = {
        "schema": "holdout-recomputed-vs-ladder-v1",
        "labeling_bug": {
            "description": "run_cycle1.py sets best_move/selected_move from the ORACLE ENTRY position for every backward band row, not the ply-legal move. Ladder selected_move is the ply proof move.",
            "code_locus": "training/oracle_horizon/run_cycle1.py lines assigning best_move=oracle.get('selected_move')",
            "holdout_rows_best_ne_ladder": terminal_vs_ladder,
            "holdout_rows_with_ladder": with_ladder,
        },
        "move_accuracy_vs_ladder_proven": {},
        "pairwise_search_move_sequence_diffs_parent_raw_ema": pairwise_diffs,
        "note": "Original Band1-3 move accuracy 0/44 used terminal best_move and is INVALID as an action metric. Recomputed figures below use ladder proven moves.",
    }
    for lab in ("parent", "raw", "ema"):
        bands = {}
        tot_n = tot_m = tot_50 = 0
        first_all = []
        for band in ("0", "1", "2", "3"):
            s = stats[lab][band]
            mean = sum(s["first_nodes"]) / len(s["first_nodes"]) if s["first_nodes"] else None
            bands[band] = {
                "n": s["n"],
                "match_any_budget": s["match_any"],
                "match_50k": s["match_50k"],
                "rate_any": s["match_any"] / max(1, s["n"]),
                "rate_50k": s["match_50k"] / max(1, s["n"]),
                "mean_first_nodes": mean,
            }
            tot_n += s["n"]
            tot_m += s["match_any"]
            tot_50 += s["match_50k"]
            first_all.extend(s["first_nodes"])
        b23_n = bands["2"]["n"] + bands["3"]["n"]
        b23_m = bands["2"]["match_any_budget"] + bands["3"]["match_any_budget"]
        report["move_accuracy_vs_ladder_proven"][lab] = {
            "total_rate_any": tot_m / max(1, tot_n),
            "total_match_any": f"{tot_m}/{tot_n}",
            "total_rate_50k": tot_50 / max(1, tot_n),
            "band2_3_rate_any": b23_m / max(1, b23_n),
            "band2_3_match_any": f"{b23_m}/{b23_n}",
            "mean_first_nodes": (sum(first_all) / len(first_all)) if first_all else None,
            "bands": bands,
        }

    # parent vs raw/ema deltas on ladder metric
    p = report["move_accuracy_vs_ladder_proven"]["parent"]
    for lab in ("raw", "ema"):
        c = report["move_accuracy_vs_ladder_proven"][lab]
        report.setdefault("deltas_vs_parent_ladder_metric", {})[lab] = {
            "total_rate_any_delta": c["total_rate_any"] - p["total_rate_any"],
            "band2_3_rate_any_delta": c["band2_3_rate_any"] - p["band2_3_rate_any"],
        }

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "HOLDOUT_RECOMPUTED_VS_LADDER.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("wrote", path)


if __name__ == "__main__":
    main()
