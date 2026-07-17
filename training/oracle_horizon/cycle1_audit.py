"""Audits for supervised Oracle Horizon Cycle 1 JSONL labels.

The audit is deliberately independent of the miner so that a malformed or
hand-edited primary file cannot silently become training data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping

RACE_PROOF_SCORE = 31_000
PRIMARY_CLASSES = {"EXACT_ORACLE", "ORACLE_BACKED_MINIMAX"}
REQUIRED_PROVENANCE = {
    "weights_sha256", "engine_sha256", "game_id", "lineage_id", "packed_state_hex",
    "book_move_used", "evaluation_only", "oracle_proven",
}


def race_proof(score: int | float | None, proven: bool) -> bool:
    """Return true for the near-terminal score band, even without proof."""
    return score is not None and abs(int(score)) >= RACE_PROOF_SCORE and not proven


def oracle_resolved(score: int | float | None, proven: bool) -> bool:
    """Only an explicit engine proof establishes an oracle entry.

    The near-terminal race band is useful diagnostics, but is intentionally
    not sufficient for EXACT or BACKED labels.
    """
    return bool(proven)


def classify_resolution(score: int | float | None, proven: bool) -> str:
    if proven:
        return "EXACT_ORACLE"
    if race_proof(score, False):
        return "SEARCH_ONLY"
    return "UNRESOLVED"


def _iter_rows(source: Iterable[Mapping] | Path | str) -> list[dict]:
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            return []
        return [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return [dict(row) for row in source]


def audit_rows(
    rows: Iterable[Mapping] | Path | str,
    *,
    parent_weights_sha256: str | None = None,
    parent_engine_sha256: str | None = None,
    max_positions: int = 10_000,
    min_primary: int = 50,
) -> dict:
    """Audit candidates or primary labels and return a serializable report."""
    values = _iter_rows(rows)
    failures: list[str] = []
    seen: set[str] = set()
    games: defaultdict[str, int] = defaultdict(int)
    lineages: defaultdict[str, int] = defaultdict(int)
    proven_wdl: dict[str, str] = {}
    primary_count = 0
    race_count = 0
    exact_count = 0
    backed_count = 0
    evaluation_only = 0
    for index, row in enumerate(values):
        prefix = f"row[{index}]"
        missing = sorted(REQUIRED_PROVENANCE - row.keys())
        if missing:
            failures.append(f"{prefix}: missing provenance {','.join(missing)}")
        packed = str(row.get("packed_state_hex", ""))
        if packed in seen:
            failures.append(f"{prefix}: duplicate packed_state_hex")
        if packed:
            seen.add(packed)
        if row.get("book_move_used") is not False:
            failures.append(f"{prefix}: book_move_used must be false")
        if row.get("evaluation_only") is not False:
            evaluation_only += 1
            failures.append(f"{prefix}: evaluation_only must be false")
        if parent_weights_sha256 and row.get("weights_sha256") != parent_weights_sha256:
            failures.append(f"{prefix}: weights sha mismatch")
        if parent_engine_sha256 and row.get("engine_sha256") != parent_engine_sha256:
            failures.append(f"{prefix}: engine sha mismatch")
        klass = str(row.get("label_class", row.get("proof_completeness_class", "")))
        proven = bool(row.get("oracle_proven", row.get("proven", False)))
        if abs(int(row.get("score", 0) or 0)) >= RACE_PROOF_SCORE and not proven:
            race_count += 1
            if klass in PRIMARY_CLASSES:
                failures.append(f"{prefix}: race-band score is not exact proof")
        if klass == "EXACT_ORACLE":
            exact_count += 1
            if not proven:
                failures.append(f"{prefix}: EXACT_ORACLE without proven=true")
        if klass == "ORACLE_BACKED_MINIMAX":
            backed_count += 1
            if not proven and not row.get("backed_proven"):
                failures.append(f"{prefix}: BACKED label lacks ladder proof")
        if klass not in PRIMARY_CLASSES:
            if row.get("primary") is True:
                failures.append(f"{prefix}: non-primary class marked primary")
        else:
            primary_count += 1
        game = str(row.get("game_id", ""))
        lineage = str(row.get("lineage_id", ""))
        games[game] += 1
        lineages[lineage] += 1
        if proven:
            wdl = str(row.get("oracle_wdl", ""))
            key = row.get("position_id", packed)
            if key in proven_wdl and proven_wdl[key] != wdl:
                failures.append(f"{prefix}: oracle WDL changed after proof")
            proven_wdl[str(key)] = wdl
    if len(values) > max_positions:
        failures.append(f"candidate cap exceeded: {len(values)} > {max_positions}")
    if primary_count < min_primary:
        status = "INSUFFICIENT_YIELD"
    elif failures:
        status = "FAIL"
    else:
        status = "PASS"
    return {
        "status": status,
        "audit_pass": status == "PASS",
        "rows": len(values),
        "primary_count": primary_count,
        "exact_count": exact_count,
        "backed_count": backed_count,
        "race_band_diagnostic_count": race_count,
        "evaluation_only_count": evaluation_only,
        "unique_packed_count": len(seen),
        "source_game_counts": dict(games),
        "source_lineage_counts": dict(lineages),
        "failures": failures,
        "primary_rejected_classes": ["PARTIAL", "ORACLE_SUPPORTED_PARTIAL", "SEARCH_ONLY"],
        "min_primary": min_primary,
    }


def audit_file(path: Path | str, **kwargs: object) -> dict:
    return audit_rows(Path(path), **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--weights-sha256")
    parser.add_argument("--engine-sha256")
    args = parser.parse_args()
    report = audit_file(
        args.labels,
        parent_weights_sha256=args.weights_sha256,
        parent_engine_sha256=args.engine_sha256,
    )
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["audit_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
