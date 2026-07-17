#!/usr/bin/env python3
"""Build the non-evaluation Claustrophobia disagreement-mining openings."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_frozen_openings import (get, validate_line_claustrophobia,
                                   validate_line_titanium, EngineSession)

OUT_DEFAULT = Path(__file__).resolve().parent / "mining_openings" / "mining_openings_pilot_v1.json"
SEEDS = [
    ("a2h","e8"), ("b2h","e8"), ("c2h","e8"), ("f2h","e8"),
    ("g2h","e8"), ("h2h","e8"), ("a3v","e8"), ("b3v","e8"),
    ("f3v","e8"), ("g3v","e8"), ("h3v","e8"), ("d2h","e8"),
    ("a2h","e8","b2h"), ("b2h","e8","c2h","e7"),
    ("c2h","e8","d2h","e7","f2h"), ("f2h","e8","g2h","e7","h2h","e6"),
]

CLEAN = Path(__file__).resolve().parent / "eval_games" / "clean_v1"

def clean_exclusions() -> dict:
    """Derive immutable exclusions from clean_v1 artifacts, read-only."""
    tuples: set[tuple[str, ...]] = set()
    ids: set[str] = set()
    hashes: set[str] = set()
    for path in CLEAN.rglob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict):
            for key in ("opening_id", "openingId"):
                if data.get(key): ids.add(str(data[key]))
            for item in data.get("openings", []) if isinstance(data.get("openings"), list) else []:
                if isinstance(item, dict):
                    if item.get("opening_id"): ids.add(str(item["opening_id"]))
                    moves = item.get("moves") or item.get("opening_seed")
                    if isinstance(moves, list): tuples.add(tuple(map(str, moves)))
            for row in data.get("roots", []) if isinstance(data.get("roots"), list) else []:
                if isinstance(row, dict):
                    moves = row.get("opening_seed") or row.get("opening")
                    if isinstance(moves, list): tuples.add(tuple(map(str, moves)))
        if path.name == "openings_used.json":
            hashes.add(hashlib.sha256(path.read_bytes()).hexdigest())
    # Preserve the frozen manifest digest as evidence even when only summaries exist.
    policy = CLEAN / "RUN_POLICY.json"
    if policy.is_file():
        try:
            digest = json.loads(policy.read_text(encoding="utf-8")).get("openings_manifest_sha256")
            if digest: hashes.add(str(digest))
        except (OSError, json.JSONDecodeError):
            pass
    return {"forbidden_move_tuples": sorted([list(x) for x in tuples]),
            "forbidden_opening_ids": sorted(ids), "forbidden_hashes": sorted(hashes)}

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--count", type=int, default=24)
    ap.add_argument("--titanium-bin", type=Path, default=None)
    ap.add_argument("--titanium-weights", type=Path, default=None)
    args = ap.parse_args()
    exclusions = clean_exclusions()
    forbidden = {tuple(x) for x in exclusions["forbidden_move_tuples"]}
    accepted, rejected, online = [], [], False
    sess = None
    try:
        get("/api/state")
        if args.titanium_bin and args.titanium_weights:
            sess = EngineSession("titanium-v17", args.titanium_weights, engine_bin=args.titanium_bin)
            online = True
    except Exception:
        pass
    try:
        for moves in SEEDS:
            if len(accepted) >= args.count: break
            if any(not m for m in moves): continue
            if len(moves) < 2 or tuple(moves) in forbidden:
                rejected.append({"moves": list(moves), "reason": "clean_v1_exact_tuple_excluded"})
                continue
            if online:
                ok, why = validate_line_claustrophobia(moves)
                if ok: ok, why = validate_line_titanium(sess, moves)
            else:
                ok, why = True, "validated-offline: live Claustrophobia/Titanium unavailable"
            if not ok:
                rejected.append({"moves": list(moves), "reason": why}); continue
            accepted.append({"opening_id": f"mine-open-{len(accepted):04d}",
                             "moves": list(moves), "plies": len(moves),
                             "validated_by": ["offline_contract"] if not online else
                             ["claustrophobia_legal_replay","titanium_session_sync"]})
    finally:
        if sess: sess.close()
    body = {"version":"mining-openings-pilot-v1", "n_openings":len(accepted),
            "enough_for_games":120, "clean_v1_excluded":True,
            "clean_v1_exclusion_evidence": exclusions,
            "validation_mode":"dual-online" if online else "validated-offline",
            "offline_note":"Offline entries require live dual validation before mining.",
            "openings":accepted, "rejected":rejected}
    stable = {"version":body["version"], "openings":[{"opening_id":x["opening_id"],"moves":x["moves"]} for x in accepted]}
    body["manifest_sha256"] = hashlib.sha256(json.dumps(stable,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    body["openings_sha256"] = body["manifest_sha256"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(body, indent=2)+"\n", encoding="utf-8")
    return 0 if accepted else 2
if __name__ == "__main__": raise SystemExit(main())
