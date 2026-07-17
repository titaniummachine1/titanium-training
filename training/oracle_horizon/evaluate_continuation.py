"""Evaluate the quarantined oracle-horizon continuation.

The script is intentionally create-only: it never changes accepted artifacts,
starts training, or promotes a candidate.  Run it after the continuation
trainer has written ``exports/continuation_{raw,ema}.bin``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "training/runs/oracle_horizon_pilot_v1/continuation_e3"
DEFAULT_ENGINE = ROOT / "engine/target-catv5-accepted-03856fe/release/titanium.exe"
DEFAULT_PARENT = ROOT / "training/runs/v16/accepted/epoch_0003.bin"
PARENT_SHA = "869ad228cfea8bb8964d98d05d6cf5e67a21b27661a36259a3976f60d486be56"
LADDER = (50_000, 200_000, 800_000, 3_200_000)
OPENINGS = (
    ("e2", "e8", "e3", "e7"),
    ("e2", "e8", "d2", "e7"),
    ("e2", "e8", "e3", "d8"),
    ("d1", "e8", "d2", "e7"),
    ("e2", "f8", "e3", "f7"),
    ("e2", "e8", "e3", "e6h"),
    ("e2", "e8", "d5h", "e7"),
    ("e2", "e8", "f2", "e7"),
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def wdl(score: Any) -> str:
    n = int(score or 0)
    return "W" if n > 0 else "L" if n < 0 else "D"


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def score_out(engine: Path, weights: Path, packed: str, nodes: int, timeout: float) -> dict[str, Any]:
    env = os.environ.copy()
    env["TITANIUM_BOOK_MODE"] = "off"
    env["TITANIUM_NET_WEIGHTS_PATH"] = str(weights.resolve())
    p = subprocess.run(
        [str(engine), "score-out", "--nodes", str(nodes), "--packed", packed],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout,
    )
    if p.returncode:
        raise RuntimeError(f"score-out failed ({p.returncode}): {p.stderr[-400:]}")
    frames = [json.loads(x) for x in p.stdout.splitlines() if x.strip()]
    if len(frames) != 1 or not isinstance(frames[0], dict):
        raise RuntimeError("score-out did not return exactly one JSON object")
    return frames[0]


def eval_position(row: dict[str, Any], label: str, weights: Path, engine: Path,
                  timeout: float) -> dict[str, Any]:
    oracle_wdl = str(row.get("oracle_wdl", "")).upper()
    best_move = row.get("best_move") or row.get("selected_move")
    stages: list[dict[str, Any]] = []
    first_wdl = first_move = None
    for nodes in LADDER:
        result = score_out(engine, weights, str(row["packed_state_hex"]), nodes, timeout)
        got_wdl = wdl(result.get("score"))
        move = result.get("selected_move")
        rec = {
            "nodes": nodes, "score": result.get("score"), "wdl": got_wdl,
            "proven": result.get("proven"), "depth": result.get("depth"),
            "selected_move": move, "nodes_reported": result.get("nodes"),
            "wdl_correct": got_wdl == oracle_wdl,
            "move_correct": move == best_move,
        }
        stages.append(rec)
        if first_wdl is None and rec["wdl_correct"] and bool(result.get("proven")):
            first_wdl = nodes
        if first_move is None and rec["move_correct"]:
            first_move = nodes
        if first_wdl is not None and first_move is not None:
            break
    return {
        "position_id": row.get("position_id", row.get("packed_state_hex")),
        "label": label, "packed_state_hex": row["packed_state_hex"],
        "band": str(row.get("band", "unknown")),
        "label_class": row.get("label_class"),
        "subset": "exact" if row.get("label_class") == "EXACT_ORACLE" else "oracle_backed",
        "best_move": best_move, "oracle_wdl": oracle_wdl,
        "plies_to_oracle_entry": row.get("plies_to_oracle_entry"),
        "needs_learning_reasons": row.get("needs_learning_reasons", []),
        "first_correct_wdl_nodes": first_wdl,
        "first_correct_move_nodes": first_move,
        "stages": stages,
    }


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize(details: list[dict[str, Any]], label: str) -> dict[str, Any]:
    def grouped(key: str) -> dict[str, Any]:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in details:
            groups[str(row.get(key, "unknown"))].append(row)
        return {name: summarize_core(items) for name, items in sorted(groups.items())}

    def summarize_core(items: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "rows": len(items),
            "wdl_accuracy": sum(bool(x["stages"] and x["stages"][-1]["wdl_correct"]) for x in items) / max(1, len(items)),
            "move_accuracy": sum(x["first_correct_move_nodes"] is not None for x in items) / max(1, len(items)),
            "mean_nodes_to_correct_wdl": mean([x["first_correct_wdl_nodes"] for x in items if x["first_correct_wdl_nodes"]]),
            "mean_nodes_to_correct_move": mean([x["first_correct_move_nodes"] for x in items if x["first_correct_move_nodes"]]),
            "mean_plies_to_oracle_entry": mean([float(x["plies_to_oracle_entry"]) for x in items if x.get("plies_to_oracle_entry") is not None]),
            "missed_only_defense_rate": (
                sum("missed_only_defense" in x.get("needs_learning_reasons", []) for x in items) / len(items)
                if items and any(x.get("needs_learning_reasons") is not None for x in items) else None
            ),
        }

    core = summarize_core(details)
    core["label"] = label
    core["bands"] = grouped("band")
    core["subsets"] = grouped("subset")
    core["wall_best_move"] = summarize_core([x for x in details if str(x.get("best_move", "")).endswith(("h", "v"))])
    core["pawn_best_move"] = summarize_core([x for x in details if not str(x.get("best_move", "")).endswith(("h", "v"))])
    return core


def run_proof(args: argparse.Namespace, weights: dict[str, Path]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    holdout_path = args.holdout
    if not holdout_path.is_file():
        return {"status": "WAITING", "reason": f"missing holdout: {holdout_path}"}, []
    rows = load_rows(holdout_path)
    if args.max_positions:
        rows = rows[:args.max_positions]
    integrity = {
        "holdout_rows": len(rows), "expected_holdout_rows": 59,
        "holdout_count_ok": len(rows) == 59 if not args.max_positions else True,
        "parent_exists": args.parent.is_file(),
        "parent_sha256": sha256(args.parent) if args.parent.is_file() else None,
        "parent_sha_ok": args.parent.is_file() and sha256(args.parent) == PARENT_SHA,
        "engine_exists": args.engine.is_file(),
        "weights": {k: {"path": str(v), "exists": v.is_file(), "sha256": sha256(v) if v.is_file() else None}
                   for k, v in weights.items()},
    }
    missing = [str(v) for v in weights.values() if not v.is_file()]
    if missing or not args.engine.is_file():
        return {
            "status": "WAITING",
            "reason": "post-training artifacts or engine are not ready",
            "missing": missing + ([] if args.engine.is_file() else [str(args.engine)]),
            "integrity": integrity,
        }, []
    details: list[dict[str, Any]] = []
    for label, path in weights.items():
        for i, row in enumerate(rows, 1):
            print(f"proof {label} {i}/{len(rows)}", flush=True)
            details.append(eval_position(row, label, path, args.engine, args.score_timeout))
    summaries = {label: summarize([x for x in details if x["label"] == label], label) for label in weights}
    parent = summaries.get("parent", {})
    for label, summary in summaries.items():
        if label == "parent":
            continue
        summary["gain_vs_parent"] = {
            "wdl_accuracy_delta": summary.get("wdl_accuracy", 0) - parent.get("wdl_accuracy", 0),
            "move_accuracy_delta": summary.get("move_accuracy", 0) - parent.get("move_accuracy", 0),
            "nodes_to_wdl_reduction": (parent.get("mean_nodes_to_correct_wdl") or 0) - (summary.get("mean_nodes_to_correct_wdl") or 0),
            "nodes_to_move_reduction": (parent.get("mean_nodes_to_correct_move") or 0) - (summary.get("mean_nodes_to_correct_move") or 0),
        }
    return {"status": "PASS", "integrity": integrity, "summaries": summaries}, details


def game_over(moves: list[str]) -> str | None:
    if not moves or moves[-1][-1:] in ("h", "v"):
        return None
    row = moves[-1][-1]
    if len(moves) % 2 and row == "9":
        return "white"
    if not len(moves) % 2 and row == "1":
        return "black"
    return None


def genmove(engine: Path, weights: Path, moves: list[str], seconds: float) -> str | None:
    env = os.environ.copy()
    env["TITANIUM_BOOK_MODE"] = "off"
    env["TITANIUM_NET_WEIGHTS_PATH"] = str(weights.resolve())
    p = subprocess.run([str(engine), "genmove", "--time", str(seconds), *moves],
                       cwd=ROOT, env=env, capture_output=True, text=True,
                       timeout=max(30.0, seconds * 4 + 10))
    lines = [x.strip() for x in p.stdout.splitlines() if x.strip()]
    for line in reversed(lines):
        if line.startswith("bestmove "):
            move = line.split()[1]
            return None if move == "(none)" else move
    return lines[-1] if lines else None


def run_screen(args: argparse.Namespace, candidate: Path, games: int,
               opponent: Path | None = None) -> dict[str, Any]:
    opponent = opponent or args.parent
    points = wins = losses = draws = 0.0
    records = []
    for i in range(games):
        a_white = i % 2 == 0
        moves = list(OPENINGS[i % len(OPENINGS)])
        while len(moves) < 200:
            winner = game_over(moves)
            if winner:
                break
            a_turn = (len(moves) % 2 == 0) == a_white
            path = candidate if a_turn else opponent
            move = genmove(args.engine, path, moves, args.screen_time)
            if not move:
                winner = "black" if len(moves) % 2 == 0 else "white"
                break
            moves.append(move)
        winner = game_over(moves) or "draw"
        a_won = (winner == "white") == a_white if winner != "draw" else None
        if a_won is True:
            wins += 1; points += 1
        elif a_won is False:
            losses += 1
        else:
            draws += 1; points += 0.5
        records.append({"game": i + 1, "candidate_color": "white" if a_white else "black",
                        "winner": winner, "plies": len(moves)})
        print(f"screen {candidate.name} {i + 1}/{games}: {winner}", flush=True)
    score = points / max(1, games)
    return {"status": "PASS", "candidate": str(candidate), "games": games,
            "wins": wins, "losses": losses, "draws": draws, "score": score,
            "records": records}


def discover_anchor() -> Path | None:
    for root in (ROOT / "training/runs", ROOT / "training"):
        for p in root.rglob("*.bin"):
            if any(token in p.name.lower() for token in ("frozen", "anchor")):
                return p
    return None


def recommendation(proof: dict[str, Any], screens: dict[str, Any]) -> str:
    if proof.get("status") != "PASS" or not proof.get("integrity", {}).get("parent_sha_ok") or not proof.get("integrity", {}).get("engine_exists"):
        return "NEED_MORE_EVIDENCE"
    ema = proof.get("summaries", {}).get("ema")
    if not ema or "gain_vs_parent" not in ema:
        return "NEED_MORE_EVIDENCE"
    gain = ema["gain_vs_parent"]
    improved = gain["wdl_accuracy_delta"] > 0 or gain["move_accuracy_delta"] > 0 or gain["nodes_to_wdl_reduction"] > 0
    screen = screens.get("ema", {})
    screen_ok = screen.get("status") == "PASS" and screen.get("score", 0) >= 0.35
    anchor = screens.get("frozen_anchor")
    anchor_ok = anchor is None or anchor.get("status") != "PASS" or anchor.get("score", 0) >= 0.35
    if improved and screen_ok and anchor_ok:
        return "PROMOTE"
    if improved and (not screen_ok or not anchor_ok):
        return "QUARANTINE"
    return "REJECT" if not screen_ok else "NEED_MORE_EVIDENCE"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-bin", type=Path)
    ap.add_argument("--ema-bin", type=Path)
    ap.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    ap.add_argument("--holdout", type=Path, default=DEFAULT_OUT / "holdout_labels.jsonl")
    ap.add_argument("--engine", type=Path, default=DEFAULT_ENGINE)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--anchor-bin", type=Path, help="Frozen-anchor weights; auto-discovery is best effort.")
    ap.add_argument("--screen-time", type=float, default=1.0)
    ap.add_argument("--score-timeout", type=float, default=180.0)
    ap.add_argument("--max-positions", type=int, default=0)
    ap.add_argument("--skip-screens", action="store_true")
    ap.add_argument("--full-gate", action="store_true", help="Run 100 games only after screen criteria pass.")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Exports live under the continuation run dir, which may differ from --out-dir
    # when evaluation artifacts are written to a nested eval/ folder.
    export_candidates = [
        args.out_dir / "exports",
        args.out_dir.parent / "exports",
        DEFAULT_OUT / "exports",
    ]
    exports = next((p for p in export_candidates if (p / "continuation_ema.bin").is_file() or (p / "continuation_raw.bin").is_file()), export_candidates[0])
    raw = args.raw_bin or exports / "continuation_raw.bin"
    ema = args.ema_bin or exports / "continuation_ema.bin"
    if not args.holdout.is_file():
        alt_holdout = DEFAULT_OUT / "holdout_labels.jsonl"
        if alt_holdout.is_file():
            args.holdout = alt_holdout
    weights = {"parent": args.parent}
    if raw.is_file(): weights["raw"] = raw
    if ema.is_file(): weights["ema"] = ema
    print(json.dumps({
        "weights_resolved": {k: str(v) for k, v in weights.items()},
        "exports_dir": str(exports),
        "holdout": str(args.holdout),
    }, indent=2), flush=True)
    proof, details = run_proof(args, weights)
    (args.out_dir / "HOLDOUT_PROOF_HORIZON.json").write_text(
        json.dumps(proof, indent=2) + "\n", encoding="utf-8")
    (args.out_dir / "HOLDOUT_PROOF_HORIZON.jsonl").write_text(
        "".join(json.dumps(x, sort_keys=True) + "\n" for x in details), encoding="utf-8")
    screens: dict[str, Any] = {}
    if not args.skip_screens and "ema" in weights:
        screens["ema"] = run_screen(args, weights["ema"], 20)
        if "raw" in weights:
            screens["raw"] = run_screen(args, weights["raw"], 20)
        anchor = args.anchor_bin or discover_anchor()
        if anchor and anchor.is_file():
            screens["frozen_anchor"] = run_screen(args, weights["ema"], 20, anchor)
            screens["frozen_anchor"]["anchor_bin"] = str(anchor)
        else:
            screens["frozen_anchor"] = {"status": "N/A", "reason": "no frozen-anchor .bin discovered; pass --anchor-bin"}
        if args.full_gate and recommendation(proof, screens) == "PROMOTE":
            screens["full_gate_100"] = run_screen(args, weights["ema"], 100)
    screens["recommendation"] = recommendation(proof, screens)
    (args.out_dir / "SCREEN_20.json").write_text(json.dumps(screens, indent=2) + "\n", encoding="utf-8")
    (args.out_dir / "FROZEN_ANCHOR_SCREEN.json").write_text(
        json.dumps(screens.get("frozen_anchor", {"status": "N/A", "reason": "screens skipped"}), indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({"proof": str(args.out_dir / "HOLDOUT_PROOF_HORIZON.json"),
                      "screen": str(args.out_dir / "SCREEN_20.json"),
                      "recommendation": screens["recommendation"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
