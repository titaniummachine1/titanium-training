"""Rank oracle-labeled moves using leaf NNUE values and a shallow proxy."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))

from titanium_training.store.state import (
    PositionState, apply_move, decode_move, legal_wall_codes, valid_pawn_destinations,
)

RUN = ROOT / "training/runs/oracle_horizon_pilot_v1/continuation_e3"
ENGINE = ROOT / "engine/target-catv5-accepted-03856fe/release/titanium.exe"
NETS = {
    "parent": ROOT / "training/runs/v16/accepted/epoch_0003.bin",
    "raw": RUN / "exports/continuation_raw.bin",
    "ema": RUN / "exports/continuation_ema.bin",
}


def packed_eval(net: Path, items: list[bytes]) -> list[dict]:
    payload = b"".join(struct.pack("<I24s", i, p) for i, p in enumerate(items))
    env = os.environ.copy()
    env.update(TITANIUM_BOOK_MODE="off", TITANIUM_NET_WEIGHTS_PATH=str(net.resolve()))
    p = subprocess.run([str(ENGINE), "eval-packed-batch"], input=payload,
                       capture_output=True, cwd=ROOT, env=env, timeout=900)
    if p.returncode:
        raise RuntimeError(p.stderr.decode(errors="replace")[-1000:])
    return [json.loads(x) for x in p.stdout.decode().splitlines() if x.strip()]


def score_out(net: Path, packed: bytes) -> dict:
    env = os.environ.copy()
    env.update(TITANIUM_BOOK_MODE="off", TITANIUM_NET_WEIGHTS_PATH=str(net.resolve()))
    p = subprocess.run([str(ENGINE), "score-out", "--nodes", "50000",
                        "--packed", packed.hex()], capture_output=True,
                       text=True, cwd=ROOT, env=env, timeout=120)
    for line in reversed(p.stdout.splitlines()):
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return {"ok": False, "returncode": p.returncode, "stderr": p.stderr[-500:]}


def leaf_score(rec: dict) -> float:
    for key in ("score", "value_i16", "value", "eval"):
        if isinstance(rec.get(key), (int, float)):
            return float(rec[key])
    return 0.0


def legal_moves(state: PositionState) -> list[str]:
    moves = []
    for cell in valid_pawn_destinations(state):
        from titanium_training.store.state import cell_to_notation
        moves.append(cell_to_notation(cell))
    for code in legal_wall_codes(state):
        moves.append(decode_move(state, code))
    return moves


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", type=Path, default=RUN / "holdout_labels.jsonl")
    ap.add_argument("--out", type=Path, default=RUN / "diagnostics/STATIC_CHILD_RANKING.json")
    ap.add_argument("--md", type=Path, default=RUN / "diagnostics/STATIC_CHILD_RANKING.md")
    ap.add_argument("--max-parents", type=int, default=59)
    args = ap.parse_args()
    rows = [json.loads(x) for x in args.holdout.read_text().splitlines() if x.strip()][:args.max_parents]
    records = []
    t0 = time.time()
    for index, row in enumerate(rows):
        parent = PositionState.unpack_state(bytes.fromhex(row["packed_state_hex"]))
        moves = legal_moves(parent)
        children = [apply_move(parent, m).packed_state() for m in moves]
        leaf_by_net = {}
        for name, net in NETS.items():
            leaf_by_net[name] = packed_eval(net, children)
        # score-out is intentionally only used for the proxy labels, with parent weights.
        proxy = []
        for child in children:
            result = score_out(NETS["parent"], child)
            score = result.get("score")
            proxy.append(None if not isinstance(score, (int, float)) else -float(score))
        expected = {"W": 1, "D": 0, "L": -1}.get(str(row.get("oracle_wdl", "")).upper(), 0)
        preserving = [i for i, value in enumerate(proxy)
                      if value is not None and (value > 0) == (expected > 0)]
        losing = [i for i, value in enumerate(proxy)
                  if value is not None and (value > 0) != (expected > 0)]
        item = {
            "position_id": row.get("position_id"), "band": row.get("band"),
            "best_move": row.get("best_move"), "oracle_wdl": row.get("oracle_wdl"),
            "side_to_move": parent.side_to_move, "wall_position": bool(parent.horizontal_walls or parent.vertical_walls),
            "legal_moves": moves, "proxy_scores_root_stm": proxy,
            "preserving_proxy_indices": preserving, "losing_proxy_indices": losing,
            "nets": {},
        }
        for name, results in leaf_by_net.items():
            scores = [leaf_score(rec) * (-1 if parent.side_to_move == 1 else 1)
                      for rec in results]
            order = sorted(range(len(moves)), key=lambda i: scores[i], reverse=True)
            best_idx = moves.index(row["best_move"]) if row.get("best_move") in moves else None
            ptop = [i for i in preserving if scores[i] == max((scores[j] for j in preserving), default=float("-inf"))]
            ltop = [i for i in losing if scores[i] == max((scores[j] for j in losing), default=float("-inf"))]
            margin = (max((scores[i] for i in preserving), default=0.0) -
                      max((scores[i] for i in losing), default=0.0))
            item["nets"][name] = {
                "scores_root_stm": scores, "rank_best_move": order.index(best_idx) + 1 if best_idx is not None else None,
                "leaf_top1_move": moves[order[0]] if order else None,
                "proxy_preserving_top1": bool(ptop and order[0] in ptop),
                "proxy_preserving_count": len(preserving), "margin_preserving_minus_losing": margin,
                "only_defense_top1": len(preserving) == 1 and order[0] in preserving,
            }
        records.append(item)
        print(f"{index + 1}/{len(rows)}", flush=True)

    summary = {}
    for name in NETS:
        vals = [r["nets"][name] for r in records]
        valid = [v for v in vals if v["rank_best_move"] is not None]
        def group(pred):
            selected = [r["nets"][name] for r in records if pred(r)]
            usable = [v for v in selected if v["rank_best_move"] is not None]
            return {
                "parents": len(selected),
                "labeled_top1_rate": sum(v["rank_best_move"] == 1 for v in usable) / max(1, len(usable)),
                "preserving_proxy_top1_rate": sum(v["proxy_preserving_top1"] for v in selected) / max(1, len(selected)),
            }
        summary[name] = {
            "parents": len(vals),
            "labeled_move_legal_count": len(valid),
            "labeled_move_missing_count": len(vals) - len(valid),
            "labeled_top1_rate": sum(v["rank_best_move"] == 1 for v in valid) / max(1, len(valid)),
            "mean_rank_best_move": sum(v["rank_best_move"] for v in valid) / max(1, len(valid)),
            "result_preserving_proxy_top1_rate": sum(v["proxy_preserving_top1"] for v in vals) / max(1, len(vals)),
            "mean_margin_preserving_minus_losing": sum(v["margin_preserving_minus_losing"] for v in vals) / max(1, len(vals)),
            "only_defense_top1_rate": (sum(v["only_defense_top1"] for v in vals if v["proxy_preserving_count"] == 1) /
                                       max(1, sum(v["proxy_preserving_count"] == 1 for v in vals))),
            "band": {str(b): group(lambda r, b=b: r["band"] == b) for b in range(4)},
            "wall_vs_pawn": {
                "wall": group(lambda r: r["wall_position"]),
                "pawn_only": group(lambda r: not r["wall_position"]),
            },
        }
    data = {"schema": "static-child-ranking-v1", "proxy_definition": "parent 50k score-out on each child; child score negated to parent STM; sign match is preserving", "proxy_is_not_certified_oracle": True, "elapsed_sec": time.time() - t0, "summary": summary, "positions": records}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    md = ["# Static child ranking", "", "Proxy only: parent-weight 50k score-out on each child, sign-normalized to parent STM. This is not a certified multi-move oracle set.", "", f"Parents: {len(records)}"]
    for name, val in summary.items():
        md.append(f"- **{name}**: labeled top-1 {val['labeled_top1_rate']:.3f}; preserving-proxy top-1 {val['result_preserving_proxy_top1_rate']:.3f}; mean margin {val['mean_margin_preserving_minus_losing']:.2f}; only-defense top-1 {val['only_defense_top1_rate']:.3f}.")
    args.md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.out), "summary": summary}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
