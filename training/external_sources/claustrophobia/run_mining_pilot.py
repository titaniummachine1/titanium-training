#!/usr/bin/env python3
"""Resumable, manifest-only 120-game Claustrophobia mining pilot."""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO/"training"))
from engine_session import EngineSession
from diversity.eval_denylist import is_evaluation_leakage
try:
    from .crossplay_titanium_ladder import TitaniumSession, play_one_with_retry
except ImportError:
    from crossplay_titanium_ladder import TitaniumSession, play_one_with_retry
CLAUSTRO_CHAMPION_SHA256 = "3a2d6d7085bf101d6500d705e57e6089f1d6b0e8d438f39b78bbb13381ea7639"

def _flat_moves(actions):
    return [a.get("move") if isinstance(a, dict) else a for a in (actions or [])]

def merge_pilot_row(play_result, metadata):
    """Make a manifest row without dropping fields returned by play_one."""
    row = dict(play_result or {})
    row.update(metadata or {})  # pilot identity/provenance wins over game data
    actions = row.get("actions") or []
    if not row.get("moves"):
        row["moves"] = _flat_moves(actions)
    if "opening_moves" not in row and row.get("opening") is not None:
        row["opening_moves"] = list(row["opening"])
    if "opening" not in row and row.get("opening_moves") is not None:
        row["opening"] = list(row["opening_moves"])
    return row

def sha(path):
    if not path or not Path(path).is_file(): return None
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--openings",type=Path,default=Path(__file__).parent/"mining_openings"/"mining_openings_pilot_v1.json")
    ap.add_argument("--out-dir",type=Path,default=Path(__file__).parent/"eval_games"/"mining_pilot_v1")
    ap.add_argument("--titanium-bin",type=Path,required=True)
    ap.add_argument("--titanium-weights",type=Path,default=None,
                    help="Backward-compatible epoch2 weights fallback")
    ap.add_argument("--epoch2-weights",type=Path,default=None)
    ap.add_argument("--candidate-weights",type=Path,default=None)
    ap.add_argument("--games",type=int,default=60); ap.add_argument("--sims",type=int,default=2)
    ap.add_argument("--time-sec",type=float,default=1.0); ap.add_argument("--max-roots",type=int,default=0)
    ap.add_argument("--run-id",default="claustro_mine_pilot_20260716_v1")
    args=ap.parse_args(); args.out_dir.mkdir(parents=True,exist_ok=True)
    manifest=args.out_dir/"manifest.json"; results=args.out_dir/"results.jsonl"
    opens=json.loads(args.openings.read_text(encoding="utf-8")).get("openings",[])
    done={}
    if results.is_file():
        for line in results.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r=json.loads(line); done[r.get("source_game_id")]=r
    resume_log = args.out_dir/"RESUME_LOG.jsonl"
    if results.is_file():
        skipped_ids = sorted(k for k in done if k)
        with resume_log.open("a", encoding="utf-8") as log:
            log.write(json.dumps({
                "ts": time.time(), "skipped_ids": skipped_ids,
                "n_skipped": len(skipped_ids), "reason": "results_exist_resume",
            }, sort_keys=True) + "\n")
    total=args.games*2
    requested=total if args.max_roots==0 else min(total,args.max_roots)
    epoch2_weights=args.epoch2_weights or args.titanium_weights
    candidate_weights=args.candidate_weights or args.titanium_weights or epoch2_weights
    if (not args.titanium_bin.is_file() or not epoch2_weights or
            not epoch2_weights.is_file() or not candidate_weights.is_file()):
        raise SystemExit("BLOCKED: Titanium binary and weights are required; no protocol run started")
    sessions={}
    try:
        with results.open("a",encoding="utf-8") as out:
            planned=[(style, idx) for style in ("epoch2","candidate") for idx in range(args.games)]
            for ordinal, (style, idx) in enumerate(planned[:requested]):
                gid=f"{args.run_id}:{style}:{idx:04d}"
                if gid in done: continue
                if style not in sessions:
                    style_weights=epoch2_weights if style=="epoch2" else candidate_weights
                    sessions[style]=TitaniumSession(args.titanium_bin,style_weights,args.time_sec)
                ti=sessions[style]
                opening_entry=opens[idx % len(opens)] if opens else {}
                opening=tuple(opening_entry.get("moves",[]))
                leaked, asset = is_evaluation_leakage(canonical_key=opening_entry.get("opening_id"),
                                                       lineage_id=opening_entry.get("lineage_id"))
                if leaked:
                    raise RuntimeError(f"clean_v1 opening denied: {opening_entry.get('opening_id')} ({asset})")
                r=play_one_with_retry(
                    titanium_first=((idx%2==0) if style=="epoch2" else (idx%2==1)),
                    opening=opening, sims=args.sims, device="cpu", ti=ti)
                if r.get("termination")=="PROTOCOL_ERROR":
                    error={"run_id":args.run_id,"source_game_id":gid,"status":"PROTOCOL_ERROR","detail":r.get("error")}
                    manifest.write_text(json.dumps(error,indent=2)+"\n",encoding="utf-8")
                    raise RuntimeError(f"protocol error in {gid}: {r.get('error')}")
                metadata={"source_game_id":gid,"run_id":args.run_id,"game_index":idx,
                     "style":style,"paired_color_index":idx,
                     "opening_id":opening_entry.get("opening_id"),
                     "opening":list(opening),"opening_moves":list(opening),
                     "dataset_kind":"training_eligible_crossplay","train_on_trajectory":False,
                     "train_on_relabeled_roots_only":True,"evaluation_eligible":False,
                     "training_eligible":False,"manifest_only":True,
                     "claustrophobia_release_tag":"v1.0.0","claustrophobia_checkpoint_sha256":None,
                     "repository_commit":"285e78d9e2023da2d4095ecdedc17bcf649948f6",
                     "titanium_binary_sha256":sha(args.titanium_bin),"titanium_weights_sha256":sha(epoch2_weights if style=="epoch2" else candidate_weights),
                     "canonical_key":f"mining_pilot_v1:{gid}",
                     "semantic_hash":hashlib.sha256(f"mining_pilot_v1:{gid}:{opening}".encode()).hexdigest(),
                     "generation_config":{"games_per_style":args.games,"sims":args.sims,"time_sec":args.time_sec,
                                          "paired_color":True},
                     "exact":False,"exact_status":"unavailable","relabeling_status":"not_started",
                     "actions":r.get("actions",[]),"termination":r.get("termination","complete"),
                     "provenance_complete":True}
                claustro_weights=REPO/"training/external_sources/claustrophobia/releases/latest/champion.pt"
                metadata["claustrophobia_checkpoint_sha256"]=sha(claustro_weights) or CLAUSTRO_CHAMPION_SHA256
                row=merge_pilot_row(r, metadata)
                out.write(json.dumps(row,sort_keys=True)+"\n"); out.flush()
    finally:
        for session in sessions.values(): session.close()
    completed_rows={}
    if results.is_file():
        for line in results.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row=json.loads(line)
                completed_rows[row.get("source_game_id")]=row
    payload={"run_id":args.run_id,"total_requested":total,"games_per_style":args.games,
             "completed":sum(1 for style,idx in planned[:requested]
                             if f"{args.run_id}:{style}:{idx:04d}" in completed_rows),
             "styles":["epoch2","candidate"],"manifest_only":True,
             "results_sha256":sha(results),"sims":args.sims,"time_sec":args.time_sec}
    manifest.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
    (args.out_dir/"DENYLIST.json").write_text(json.dumps({
        "run_id":args.run_id,"dataset_kind":"training_eligible_crossplay",
        "purpose":"raw_source_only_not_training",
        "training_eligible_crossplay_source_material":True,"train_on_trajectory":False,
        "train_on_relabeled_roots_only":True,"evaluation_eligible":False,
        "training_eligible":False,"corpus_eligible":False,
        "import_to_labels_db":False,"no_labels_db_import":True,"can_import":False,
        "raw_source_only":True,"manifest_only":True,
        "clean_v1_never_overwrite":True},indent=2)+"\n",encoding="utf-8")
    return 0
if __name__=="__main__": raise SystemExit(main())
