"""Audit Cycle 1 oracle JSONL rows against the streaming trainer path."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "training/runs/oracle_horizon_pilot_v1/continuation_e3"
DEFAULT_OUT = RUN / "diagnostics/ORACLE_TARGET_AUDIT.json"
DEFAULT_MD = RUN / "diagnostics/ORACLE_TARGET_AUDIT.md"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", type=Path, default=RUN / "train_oracle.jsonl")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--md", type=Path, default=DEFAULT_MD)
    args = ap.parse_args()
    rows = [json.loads(line) for line in args.oracle.read_text().splitlines() if line.strip()]
    loader = (ROOT / "training/streaming_db_loader.py").read_text(encoding="utf-8")
    section = loader[loader.find("if self.oracle_jsonl is not None"):loader.find("finally:", loader.find("if self.oracle_jsonl is not None"))]
    classes = Counter(str(r.get("label_class", "")) for r in rows)
    wdls = Counter(str(r.get("oracle_wdl", "")).upper() for r in rows)
    weights = Counter(1.0 if r.get("label_class") == "EXACT_ORACLE" else 0.85 for r in rows)
    report = {
        "schema": "oracle-target-audit-v1",
        "source": str(args.oracle),
        "rows": len(rows),
        "unique_packed_states": len({r.get("packed_state_hex") for r in rows}),
        "label_class_counts": dict(classes),
        "wdl_counts": dict(wdls),
        "sample_weight_counts": {str(k): v for k, v in weights.items()},
        "state_supplied": "parent packed_state_hex; loader evaluates that row's packed state",
        "target_consumed_by_loss": "oracle_wdl W/D/L -> value_stm +1/0/-1 -> stm_to_target_prob -> scalar target probability",
        "best_move_consumed": False,
        "selected_move_consumed": False,
        "child_packed_states_consumed": False,
        "alternatives_supplied": False,
        "direct_preserving_vs_losing_move_signal": False,
        "search_only_excluded": all(r.get("label_class") != "SEARCH_ONLY" for r in rows),
        "sample_weight_rule": {"EXACT_ORACLE": 1.0, "other_oracle_rows": 0.85},
        "loader_evidence": {
            "oracle_wdl_present": "oracle_wdl" in section,
            "stm_to_target_prob_present": "stm_to_target_prob(value_stm)" in section,
            "record_to_fv_present": "record_to_fv(rec, target)" in section,
            "best_move_in_oracle_section": bool(re.search(r"best_move|selected_move", section)),
            "child_state_in_oracle_section": bool(re.search(r"child|alternative", section, re.I)),
            "search_only_guard_in_loader": "SEARCH_ONLY" in loader,
        },
        "conclusion": "Parent-state scalar WDL supervision does not teach move selection directly; action-contrast supervision is required.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md = f"""# Oracle target audit

- Rows: **{len(rows)}**; unique packed states: **{report['unique_packed_states']}**.
- State supplied: **parent** (`packed_state_hex`), not a child.
- Loss target: `oracle_wdl` maps W/D/L to +1/0/-1, then `stm_to_target_prob`, and `record_to_fv` supplies the scalar NNUE feature row.
- `best_move`, `selected_move`, child states, and alternatives: **not consumed** by this loss path.
- Direct preserving-versus-losing move signal: **no**; this is parent-state value supervision.
- Label classes: `{dict(classes)}`.
- WDL balance: `{dict(wdls)}`.
- Sample weights: exact = **1.0**, backed/other oracle = **0.85** (`{dict(weights)}`).
- `SEARCH_ONLY` excluded from this file: **{report['search_only_excluded']}**.

Evidence is the oracle branch in `training/streaming_db_loader.py`; the audit is intentionally read-only.
"""
    args.md.write_text(md, encoding="utf-8")
    print(json.dumps({"output": str(args.out), "markdown": str(args.md), "rows": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
