#!/usr/bin/env python3
"""Emit JSONL paired forks only for stable Titanium/Claustro disagreements."""
from __future__ import annotations
import argparse,json
from pathlib import Path
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--rows",type=Path,required=True); ap.add_argument("--out",type=Path,required=True); ap.add_argument("--max-roots",type=int,default=0); a=ap.parse_args()
    data=json.loads(a.rows.read_text(encoding="utf-8")).get("rows",[])
    if a.max_roots: data=data[:a.max_roots]
    out=[]
    for r in data:
        if not r.get("label_stable") or not r.get("titanium_best") or r.get("titanium_best")==r.get("claustrophobia_action"): continue
        base={k:r.get(k) for k in ("source_game_id","fork_lineage_id","canonical_key","semantic_hash","scores","exact","regret")}
        for kind,mv in (("titanium_best",r["titanium_best"]),("claustrophobia_action",r["claustrophobia_action"])):
            out.append({**base,"parent_key":r.get("canonical_key"),"branch":kind,"move":mv,
                        "training_eligible":False,"evaluation_eligible":False,"corpus":False,
                        "prep_fixture":True,"label_stable":True})
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in out))
if __name__=="__main__": raise SystemExit(main())
