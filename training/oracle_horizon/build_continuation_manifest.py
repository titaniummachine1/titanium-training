"""Build the deterministic, holdout-safe Oracle Horizon continuation pool."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

SEED = 20260717
PARENT_SHA = "869ad228cfea8bb8964d98d05d6cf5e67a21b27661a36259a3976f60d486be56"
PRIMARY = {"EXACT_ORACLE", "ORACLE_BACKED_MINIMAX"}


def _score(row: dict) -> int:
    key = f"{SEED}:{row['packed_state_hex']}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big")


def split_rows(rows: list[dict], fraction: float = 0.15) -> tuple[list[dict], list[dict]]:
    bands: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        bands[str(row.get("band", "unknown"))].append(row)
    holdout, train = [], []
    for band, group in sorted(bands.items()):
        ordered = sorted(group, key=_score)
        n = max(1, round(len(group) * fraction))
        chosen = {id(row) for row in ordered[:n]}
        holdout.extend(row for row in group if id(row) in chosen)
        train.extend(row for row in group if id(row) not in chosen)
    holdout.sort(key=lambda r: str(r["packed_state_hex"]))
    train.sort(key=lambda r: str(r["packed_state_hex"]))
    return holdout, train


def audit(rows: list[dict], *, expected_parent: str = PARENT_SHA) -> dict:
    failures: list[str] = []
    seen: set[str] = set()
    for i, row in enumerate(rows):
        prefix = f"row[{i}]"
        if str(row.get("label_class")) not in PRIMARY:
            failures.append(f"{prefix}: rejected label class")
        if row.get("book_move_used") is not False:
            failures.append(f"{prefix}: book_move_used")
        if row.get("evaluation_only") is not False:
            failures.append(f"{prefix}: evaluation_only")
        if row.get("weights_sha256") != expected_parent:
            failures.append(f"{prefix}: parent sha")
        packed = str(row.get("packed_state_hex", ""))
        if not packed or packed in seen:
            failures.append(f"{prefix}: missing/duplicate packed state")
        seen.add(packed)
        if str(row.get("oracle_wdl", "")).upper() not in {"W", "D", "L"}:
            failures.append(f"{prefix}: invalid oracle_wdl")
    return {"status": "PASS" if not failures else "FAIL", "rows": len(rows), "failures": failures}


def build(source: Path, out_dir: Path) -> dict:
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    source_audit = audit(rows)
    if source_audit["status"] != "PASS":
        raise RuntimeError(json.dumps(source_audit))
    holdout, train = split_rows(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "holdout_labels.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in holdout), encoding="utf-8")
    (out_dir / "train_oracle.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in train), encoding="utf-8")
    epoch_size = max(1, round(len(train) * 3 / 0.10))
    oracle_slots = round(epoch_size * 0.10)
    manifest = {
        "schema": "oracle-horizon-continuation-v1",
        "immutable": True,
        "parent_epoch": 3,
        "parent_weights_sha256": PARENT_SHA,
        "mix": {"general": 0.80, "oracle_horizon": 0.10, "anchors": 0.10},
        "stream_epoch_size": epoch_size,
        "effective_counts": {
            "source_rows": len(rows), "holdout": len(holdout), "train_oracle_unique": len(train),
            "oracle_slots": oracle_slots, "general_slots": epoch_size - oracle_slots,
            "anchor_slots": round(epoch_size * 0.10),
        },
        "repeat_factor": oracle_slots / max(1, len(train)),
        "max_repeat": 3,
        "band_balance": {
            "source": dict(Counter(str(r.get("band", "unknown")) for r in rows)),
            "holdout": dict(Counter(str(r.get("band", "unknown")) for r in holdout)),
            "train_oracle": dict(Counter(str(r.get("band", "unknown")) for r in train)),
        },
        "exact_vs_backed_weights": {"EXACT_ORACLE": 1.0, "ORACLE_BACKED_MINIMAX": 0.85},
        "holdout_ids": [str(r["packed_state_hex"]) for r in holdout],
        "book_off": True,
        "SEARCH_ONLY_excluded": True,
        "seed": SEED,
        "source": str(source),
    }
    (out_dir / "TRAIN_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    pre = audit(train)
    pre.update({"holdout_rows": len(holdout), "manifest": str(out_dir / "TRAIN_MANIFEST.json")})
    (out_dir / "PRETRAIN_AUDIT.json").write_text(json.dumps(pre, indent=2) + "\n", encoding="utf-8")
    if pre["status"] != "PASS":
        raise RuntimeError("pre-train audit failed")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=Path("training/runs/oracle_horizon_pilot_v1/cycle1/labels_primary.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=Path("training/runs/oracle_horizon_pilot_v1/continuation_e3"))
    args = ap.parse_args()
    print(json.dumps(build(args.source, args.out_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
