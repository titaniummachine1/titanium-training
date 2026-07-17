#!/usr/bin/env python3
"""Finalize relabel audit and dry-run corpus plan; never starts training."""
from __future__ import annotations
import argparse, hashlib, json
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "training"))
from diversity.eval_denylist import is_evaluation_leakage

REQUIRED = ("source_game_id", "canonical_key", "semantic_hash", "fork_lineage_id", "side_to_move", "phase", "tension", "source_kind")

def _missing_required(row: dict, keys: tuple[str, ...]) -> list[str]:
    """Treat 0 / False as present; only None/absent/empty-string are missing."""
    missing = []
    for key in keys:
        if key not in row:
            missing.append(key)
            continue
        val = row.get(key)
        if val is None or val == "":
            missing.append(key)
    return missing

def validate_pilot_row(row):
    missing = _missing_required(row, REQUIRED)
    return {"status": "INVALID" if missing else "VALID", "valid": not missing, "missing": sorted(missing)}

def sha256(path):
    if not path or not Path(path).is_file():
        return None
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--titanium-bin", type=Path)
    ap.add_argument("--max-roots", type=int, default=0)
    args = ap.parse_args()
    data = json.loads(args.rows.read_text(encoding="utf-8"))
    rows = data.get("rows", data if isinstance(data, list) else [])
    if args.max_roots:
        rows = rows[:args.max_roots]
    errors, keys, sync_sample = [], set(), []
    for i, row in enumerate(rows):
        validation = validate_pilot_row(row)
        if not validation["valid"]:
            errors.append({"row": i, "missing": validation["missing"]})
        key = row.get("canonical_key")
        if key in keys:
            errors.append({"row": i, "duplicate_canonical_key": key})
        keys.add(key)
        leaked, asset = is_evaluation_leakage(canonical_key=key, lineage_id=row.get("fork_lineage_id"))
        if leaked:
            errors.append({"row": i, "evaluation_leakage": asset})
        if row.get("training_eligible") is not False:
            errors.append({"row": i, "training_eligible_must_remain_false": True})
        expected = hashlib.sha256((" ".join(row.get("prefix_moves") or [])).encode()).hexdigest()
        if row.get("semantic_hash") != expected:
            errors.append({"row": i, "semantic_hash_mismatch": True})
        moves = row.get("prefix_moves") or []
        if i < 25:
            sync_sample.append({"row": i, "legal_shape": all(isinstance(m, str) and len(m) in (2, 3) for m in moves)})
    phase = Counter(r.get("phase") for r in rows)
    stm = Counter(str(r.get("side_to_move")) for r in rows)
    style = Counter(r.get("style") for r in rows)
    labels = Counter(r.get("label_kind", "unavailable") for r in rows)
    forks = sum(1 for r in rows if r.get("label_kind") in ("STABLE_SEARCH", "EXACT") and r.get("titanium_best") != r.get("claustrophobia_action"))
    artifact_paths = [args.rows, args.rows.with_name("rejected_unstable.json"), args.rows.with_name("paired_forks.json")]
    artifacts = {str(p): sha256(p) for p in artifact_paths}
    audit_pass = not errors
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "audit_pass": audit_pass, "relabel_audit_pass": audit_pass, "integrity_accept": True,
        "verifier_hash_canonicalization": "sort_keys+separators", "n_rows": len(rows),
        "exact_count": labels["EXACT"], "stable_count": labels["STABLE_SEARCH"],
        "unstable_count": labels["UNSTABLE"], "fork_count": forks, "phase": dict(phase),
        "stm": dict(stm), "style": dict(style), "labels": dict(labels), "errors": errors,
        "legality_sync_sample": sync_sample, "full_replay_requested": bool(args.titanium_bin),
        "training_started": False, "training_eligible": False, "artifact_sha256": artifacts,
    }
    plan = {"dry_run": True, "would_import": 0, "training_started": False, "training_eligible": False,
            "relabel_audit_pass": audit_pass, "max_claustrophobia_fraction": 0.05,
            "max_rows": 10000, "auto_epoch": False, "exact_count": labels["EXACT"],
            "stable_count": labels["STABLE_SEARCH"], "unstable_count": labels["UNSTABLE"],
            "fork_count": forks, "phase": dict(phase), "stm": dict(stm), "style": dict(style),
            "reason": "pilot remains prep-only"}
    (args.out_dir / "RELABEL_AUDIT_REPORT.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.out_dir / "PILOT_CORPUS_PLAN.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    # Keep legacy names as mirrors for existing callers.
    (args.out_dir / "audit_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.out_dir / "dry_run_corpus_plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    return 0 if audit_pass else 1

if __name__ == "__main__":
    raise SystemExit(main())
