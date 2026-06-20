"""Dataset semantic parity: SQLite teacher reference vs candidate Parquet."""
from __future__ import annotations

import json
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from position_store_config import REPORT_DIR, ROOT, TEACHER_STORE_DB


@dataclass
class DatasetSemanticParity:
    passed: bool = False
    sqlite_positions: int = 0
    parquet_positions: int = 0
    sqlite_labels: int = 0
    parquet_labels: int = 0
    sqlite_observations: int = 0
    parquet_observations: int = 0
    sqlite_friend_with_policy: int = 0
    parquet_has_policy: int = 0
    parquet_no_policy: int = 0
    sqlite_no_policy_estimate: int = 0
    cohort_mismatches: int = 0
    mismatches: list[str] = field(default_factory=list)
    sqlite_cohort_top: dict[str, int] = field(default_factory=dict)
    parquet_cohort_top: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "sqlite_positions": self.sqlite_positions,
            "parquet_positions": self.parquet_positions,
            "sqlite_labels": self.sqlite_labels,
            "parquet_labels": self.parquet_labels,
            "sqlite_observations": self.sqlite_observations,
            "parquet_observations": self.parquet_observations,
            "sqlite_friend_with_policy": self.sqlite_friend_with_policy,
            "parquet_has_policy": self.parquet_has_policy,
            "parquet_no_policy": self.parquet_no_policy,
            "sqlite_no_policy_estimate": self.sqlite_no_policy_estimate,
            "cohort_mismatches": self.cohort_mismatches,
            "mismatches": self.mismatches,
            "sqlite_cohort_top": self.sqlite_cohort_top,
            "parquet_cohort_top": self.parquet_cohort_top,
        }


def audit_dataset_semantic_parity(
    candidate_dir: Path,
    *,
    teacher_db: Path = TEACHER_STORE_DB,
    root: Path = ROOT,
) -> DatasetSemanticParity:
    manifest = json.loads((candidate_dir / "manifest.json").read_text(encoding="utf-8"))
    counts = manifest.get("counts") or {}
    report = DatasetSemanticParity()

    conn = sqlite3.connect(f"file:{teacher_db}?mode=ro", uri=True)
    report.sqlite_positions = conn.execute("SELECT COUNT(*) FROM positions").fetchone()[0]
    report.sqlite_labels = conn.execute("SELECT COUNT(*) FROM labels").fetchone()[0]
    report.sqlite_observations = conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
    report.sqlite_friend_with_policy = conn.execute(
        "SELECT COUNT(*) FROM labels WHERE source LIKE 'friend_selfplay:%' "
        "AND payload_json LIKE '%policy_hash%'"
    ).fetchone()[0]
    sql_cohort: Counter = Counter()
    for source, n in conn.execute("SELECT source, COUNT(*) FROM labels GROUP BY source"):
        sql_cohort[str(source or "")] = int(n)
    conn.close()

    report.sqlite_no_policy_estimate = report.sqlite_labels - report.sqlite_friend_with_policy

    report.parquet_positions = pq.read_metadata(root / manifest["parts"]["positions"][0]).num_rows
    labels_table = pq.read_table(root / manifest["parts"]["labels"][0])
    report.parquet_labels = labels_table.num_rows
    report.parquet_observations = pq.read_metadata(root / manifest["parts"]["observations"][0]).num_rows
    has_pol = labels_table.column("has_policy")
    report.parquet_has_policy = sum(1 for i in range(labels_table.num_rows) if bool(has_pol[i].as_py()))
    report.parquet_no_policy = report.parquet_labels - report.parquet_has_policy

    pq_cohort = Counter()
    cohort_col = labels_table.column("source_cohort")
    for i in range(labels_table.num_rows):
        pq_cohort[str(cohort_col[i].as_py() or "")] += 1

    report.sqlite_cohort_top = dict(sql_cohort.most_common(25))
    report.parquet_cohort_top = dict(pq_cohort.most_common(25))

    for name, sql_n, manifest_n, pq_n in [
        ("positions", report.sqlite_positions, int(counts.get("positions", 0)), report.parquet_positions),
        ("labels", report.sqlite_labels, int(counts.get("labels", 0)), report.parquet_labels),
        ("observations", report.sqlite_observations, int(counts.get("observations", 0)), report.parquet_observations),
    ]:
        if sql_n != manifest_n:
            report.mismatches.append(f"{name}: sqlite={sql_n} manifest={manifest_n}")
        if pq_n != manifest_n:
            report.mismatches.append(f"{name}: parquet={pq_n} manifest={manifest_n}")

    expected_no = int(counts.get("labels", 0)) - int(counts.get("has_policy_labels", report.parquet_has_policy))
    if report.parquet_no_policy != expected_no and abs(report.parquet_no_policy - expected_no) > 0:
        report.mismatches.append(
            f"no_policy_labels: parquet={report.parquet_no_policy} manifest_expected={expected_no}"
        )

    for cohort in set(sql_cohort) | set(pq_cohort):
        if sql_cohort.get(cohort, 0) != pq_cohort.get(cohort, 0):
            report.cohort_mismatches += 1
    if report.cohort_mismatches:
        report.mismatches.append(f"cohort_distribution: {report.cohort_mismatches} cohorts differ")

    report.passed = len(report.mismatches) == 0
    return report


def write_semantic_parity_report(report: DatasetSemanticParity, *, out_dir: Path = REPORT_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"dataset_semantic_parity_{stamp}.json"
    path.write_text(
        json.dumps({"generated_at": datetime.now(timezone.utc).isoformat(), **report.to_dict()}, indent=2),
        encoding="utf-8",
    )
    return path
