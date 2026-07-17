#!/usr/bin/env python3
"""Relabel mined roots with book-off Titanium searches."""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "training"))
from engine_session import EngineSession

STABILITY_SCORE_MARGIN_CP = 50
UNKNOWN = "unknown_unavailable"
SEARCH_CONFIG = {"engine": "titanium-v17", "budgets_sec": [1.0, 4.0], "sims": 2, "opening_book": "off"}
SEARCH_CONFIG_HASH = hashlib.sha256(json.dumps(SEARCH_CONFIG, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def score(info):
    info = info or {}
    for key in ("rootScore", "score"):
        value = info.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        if isinstance(value, dict):
            for nested in ("cp", "centipawns", "value"):
                if isinstance(value.get(nested), (int, float)):
                    return float(value[nested])
    moves = info.get("rootMoves", info.get("root_moves", []))
    if moves and isinstance(moves[0], dict):
        return score({"score": moves[0].get("score")})
    return None

def stable_search_pair(first_move, second_move, first_score, second_score):
    if not first_move or first_move != second_move:
        return False
    return first_score is None or second_score is None or abs(float(second_score) - float(first_score)) <= STABILITY_SCORE_MARGIN_CP

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", type=Path, required=True)
    ap.add_argument("--titanium-bin", type=Path, required=True)
    ap.add_argument("--titanium-weights", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--paired-out", type=Path)
    ap.add_argument("--max-roots", type=int, default=0)
    args = ap.parse_args()
    os.environ["TITANIUM_BOOK_MODE"] = "off"
    roots_data = json.loads(args.roots.read_text(encoding="utf-8"))
    roots = roots_data.get("roots", roots_data if isinstance(roots_data, list) else [])
    selected = roots if not args.max_roots else roots[:args.max_roots]
    out, unstable = [], []
    started = time.perf_counter()
    session = EngineSession("titanium-v17", args.titanium_weights, engine_bin=args.titanium_bin)
    try:
        for root in selected:
            prefix = root.get("prefix_moves") or []
            if not session.sync(prefix):
                row = {**root, "relabel_status": "UNSTABLE", "label_kind": "UNSTABLE",
                       "training_eligible": False, "relabel_audit_pass": False, "failure": "protocol_error"}
                out.append(row); unstable.append(row); continue
            searches = [session.go_detailed(1.0), session.go_detailed(4.0)]
            short, long = searches
            info = long.get("info") or {}
            scores = [score(x.get("info")) for x in searches]
            best = long.get("bestmove")
            stable = stable_search_pair(short.get("bestmove"), best, scores[0], scores[1])
            label_kind = "STABLE_SEARCH" if stable else "UNSTABLE"
            status = "STABLE_SEARCH" if stable else "UNSTABLE"
            nested = root.get("provenance") or {}
            top3 = (info.get("rootMoves", info.get("root_moves", [])) or [])[:3]
            row = {**root, "titanium_best": best, "titanium_best_move": best,
                   "titanium_searches": searches, "rootMoves": info.get("rootMoves", info.get("root_moves", [])),
                   "top3": top3, "pv": info.get("pv", []), "nodes": [x.get("info", {}).get("nodes") for x in searches],
                   "time": [x.get("info", {}).get("time") for x in searches], "scores": scores,
                   "regret": info.get("regret"), "exact": False, "exact_label_kind": "unavailable",
                   "exact_status": "unavailable", "label_kind": label_kind, "label_stable": stable,
                   "relabel_status": status, "relabeling_status": status, "training_eligible": False,
                   "evaluation_eligible": False, "corpus": False, "corpus_eligible": False,
                   "opening_book": "off", "search_config": SEARCH_CONFIG, "search_config_hash": SEARCH_CONFIG_HASH,
                   "sims": 2, "cpu_wall_time_sec": time.perf_counter() - started,
                   "source_run_id": root.get("source_run_id", root.get("run_id", UNKNOWN)),
                   "source_process_session": root.get("source_process_session", UNKNOWN),
                   "source_titanium_net": nested.get("titanium_weights_sha256", UNKNOWN),
                   "verifier_note": "hash canonicalization fix: sort_keys+separators",
                   "verifier_hash_canonicalization": "sort_keys+separators",
                   "relabel_audit_pass": bool(stable), "provenance_complete": bool(root.get("provenance_complete")),
                   "engine_identity": "titanium-v17", "engine_weight_hash": nested.get("titanium_weights_sha256", UNKNOWN)}
            out.append(row)
            if not stable:
                unstable.append(row)
    finally:
        session.close()
    elapsed = time.perf_counter() - started
    # Hard accept cap applies only to stable/exact rows; unstable rows remain auditable.
    accepted = [r for r in out if r.get("label_kind") in ("STABLE_SEARCH", "EXACT")]
    if len(accepted) > 10000:
        accepted_ids = {r["root_id"] for r in accepted[:10000]}
        for row in out:
            if row.get("root_id") not in accepted_ids and row.get("label_kind") in ("STABLE_SEARCH", "EXACT"):
                row["relabel_status"] = "REJECTED_CAP"; row["relabel_audit_pass"] = False
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"rows": out, "n_rows": len(out), "search_budgets_sec": [1.0, 4.0],
               "exact_unavailable": True, "opening_book": "off", "search_config": SEARCH_CONFIG,
               "search_config_hash": SEARCH_CONFIG_HASH, "cpu_wall_time_sec": elapsed,
               "training_eligible": False, "relabel_audit_pass": False}
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    unstable_path = args.out.with_name("rejected_unstable.json")
    unstable_path.write_text(json.dumps({"rows": unstable, "n_rows": len(unstable), "training_eligible": False}, indent=2) + "\n", encoding="utf-8")
    paired = []
    for row in out:
        if row.get("label_kind") in ("STABLE_SEARCH", "EXACT") and row.get("titanium_best") != row.get("claustrophobia_action"):
            paired.append({**row, "fork_type": "stable_disagreement", "second_best": (row.get("top3") or [None, None])[1],
                           "regret": row.get("regret"), "training_eligible": False})
    paired_path = args.paired_out or args.out.with_name("paired_forks.json")
    paired_path.write_text(json.dumps(paired, indent=2) + "\n", encoding="utf-8")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
