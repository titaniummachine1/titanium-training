#!/usr/bin/env python3
"""FLYWHEEL_SPEC_V1 §0 label-certification harness (Phase A/B skeleton).

Runs bounded pilots and audits before any Gen-0 mass generation. On failure,
writes training/data/overnight_logs/TRAINING_PAUSED.json (same contract as
training_coordinator.py).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_TRAINING = Path(__file__).resolve().parents[1]
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

from titanium_training.paths import ENGINE_BIN, TRAINING_ROOT  # noqa: E402
from titanium_training.store.config import TEACHER_STORE_DB  # noqa: E402

REPORT_DIR = TRAINING_ROOT / "data" / "label_certification"
PAUSE_FILE = TRAINING_ROOT / "data" / "overnight_logs" / "TRAINING_PAUSED.json"
SPEC_THROUGHPUT_ROWS_PER_SEC = 9.25
SPEC_COST_CEILING_USD_PER_800K = 8.0
SPEC_BOX_HOURLY_USD = 0.35


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_pause(reason: str, *, details: dict[str, Any] | None = None) -> None:
    PAUSE_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "paused": True,
        "reason": reason,
        "updated_at": utc_now(),
    }
    if details:
        payload["details"] = details
    PAUSE_FILE.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


@dataclass
class CostPilotRoot:
    position_id: int
    node_budget: int
    wall_ms: float
    nodes: int | None
    bound: str | None
    exact: bool
    skipped: bool


@dataclass
class CostPilotReport:
    spec: str = "FLYWHEEL_SPEC_V1 §0 cost pilot"
    generated_at: str = field(default_factory=utc_now)
    engine_bin: str = ""
    engine_sha256: str = ""
    source_db: str = ""
    roots_requested: int = 10
    roots_labeled: int = 0
    roots_skipped: int = 0
    exact_bound_rate: float = 0.0
    total_wall_sec: float = 0.0
    throughput_rows_per_sec: float = 0.0
    estimated_usd_per_800k: float | None = None
    throughput_pass: bool = False
    cost_pass: bool = False
    exact_pass: bool = False
    roots: list[CostPilotRoot] = field(default_factory=list)
    kill_reasons: list[str] = field(default_factory=list)

    def evaluate_kills(self, *, pilot_only: bool = False) -> bool:
        self.kill_reasons.clear()
        if self.roots_labeled == 0:
            self.kill_reasons.append("no exact labels produced")
        if self.roots_labeled > 0 and self.exact_bound_rate < 1.0:
            self.kill_reasons.append(
                f"exact-bound rate {self.exact_bound_rate:.3f} < 1.0"
            )
        if pilot_only:
            self.exact_pass = self.roots_labeled > 0 and self.exact_bound_rate >= 1.0
            self.throughput_pass = True
            self.cost_pass = True
            return not self.kill_reasons
        if self.throughput_rows_per_sec < SPEC_THROUGHPUT_ROWS_PER_SEC:
            self.kill_reasons.append(
                f"throughput {self.throughput_rows_per_sec:.2f} < {SPEC_THROUGHPUT_ROWS_PER_SEC}"
            )
        if (
            self.estimated_usd_per_800k is not None
            and self.estimated_usd_per_800k > SPEC_COST_CEILING_USD_PER_800K
        ):
            self.kill_reasons.append(
                f"estimated ${self.estimated_usd_per_800k:.2f}/800k > ${SPEC_COST_CEILING_USD_PER_800K}"
            )
        self.exact_pass = self.roots_labeled > 0 and self.exact_bound_rate >= 1.0
        self.throughput_pass = self.throughput_rows_per_sec >= SPEC_THROUGHPUT_ROWS_PER_SEC
        self.cost_pass = (
            self.estimated_usd_per_800k is None
            or self.estimated_usd_per_800k <= SPEC_COST_CEILING_USD_PER_800K
        )
        return not self.kill_reasons


def _score_out_probe(engine_bin: Path, packed_hex: str, node_budget: int) -> tuple[dict[str, Any] | None, float]:
    t0 = time.perf_counter()
    completed = subprocess.run(
        [str(engine_bin), "score-out", "--nodes", str(node_budget), "--packed", packed_hex],
        capture_output=True,
        text=True,
        check=False,
    )
    wall_ms = (time.perf_counter() - t0) * 1000.0
    if completed.returncode != 0:
        return None, wall_ms
    try:
        decoder = json.JSONDecoder()
        start = len(completed.stdout) - len(completed.stdout.lstrip())
        parsed, end = decoder.raw_decode(completed.stdout, start)
    except (json.JSONDecodeError, TypeError):
        return None, wall_ms
    if completed.stdout[end:].strip():
        return None, wall_ms
    return parsed if isinstance(parsed, dict) else None, wall_ms


def run_cost_pilot(
    *,
    db_path: Path = TEACHER_STORE_DB,
    roots: int = 10,
    node_budget: int = 20_000,
    engine_bin: Path = ENGINE_BIN,
    out_path: Path | None = None,
) -> CostPilotReport:
    if not engine_bin.is_file():
        raise FileNotFoundError(engine_bin)
    if not db_path.is_file():
        raise FileNotFoundError(db_path)

    from score_out_labels import collect_labels

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    labels_path = REPORT_DIR / f".cost_pilot_{roots}_{node_budget}.jsonl"
    if labels_path.exists():
        labels_path.unlink()

    report = CostPilotReport(
        engine_bin=str(engine_bin),
        source_db=str(db_path),
        roots_requested=roots,
        engine_sha256=hashlib.sha256(engine_bin.read_bytes()).hexdigest(),
    )

    t0 = time.perf_counter()
    summary = collect_labels(
        db_path,
        labels_path,
        max_positions=roots,
        node_budget=node_budget,
        engine_bin=engine_bin,
    )
    total_wall_sec = time.perf_counter() - t0
    report.roots_labeled = int(summary.get("records", 0))
    report.roots_skipped = int(summary.get("skipped", 0))
    report.total_wall_sec = total_wall_sec

    exact_count = 0
    per_root_wall = (total_wall_sec * 1000.0 / max(report.roots_labeled, 1))
    for line in labels_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        bound = rec.get("bound")
        exact = bound == "exact"
        if exact:
            exact_count += 1
        report.roots.append(
            CostPilotRoot(
                position_id=int(rec["position_id"]),
                node_budget=node_budget,
                wall_ms=per_root_wall,
                nodes=rec.get("nodes"),
                bound=bound,
                exact=exact,
                skipped=False,
            )
        )

    if report.roots_labeled > 0:
        report.exact_bound_rate = exact_count / report.roots_labeled
        report.throughput_rows_per_sec = report.roots_labeled / report.total_wall_sec
        box_hours = report.total_wall_sec / 3600.0
        if box_hours > 0:
            usd = box_hours * SPEC_BOX_HOURLY_USD
            report.estimated_usd_per_800k = usd * (800_000 / report.roots_labeled)

    passed = report.evaluate_kills(pilot_only=roots < 100)
    out = out_path or (REPORT_DIR / "cost_pilot_report.json")
    out.write_text(json.dumps(asdict(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not passed:
        write_pause("flywheel_label_cert_cost_pilot_failed", details={"kill_reasons": report.kill_reasons})
    return report


def run_fidelity_calibration(
    *,
    engine_bin: Path = ENGINE_BIN,
    node_budgets: tuple[int, ...] = (5_000, 10_000, 20_000, 50_000, 100_000, 200_000),
    out_path: Path | None = None,
) -> dict[str, Any]:
    """Spike: wall-time vs node budget on startpos for 200ms/800ms mapping."""
    if not engine_bin.is_file():
        raise FileNotFoundError(engine_bin)
    from titanium_training.store.state import PositionState

    packed = PositionState.initial().packed_state().hex()
    rows: list[dict[str, Any]] = []
    for budget in node_budgets:
        result, wall_ms = _score_out_probe(engine_bin, packed, budget)
        rows.append(
            {
                "node_budget": budget,
                "wall_ms": round(wall_ms, 2),
                "bound": (result or {}).get("bound"),
                "nodes": (result or {}).get("nodes"),
                "depth": (result or {}).get("depth"),
            }
        )
    payload = {
        "spec": "FLYWHEEL_SPEC_V1 §2 fidelity calibration spike",
        "generated_at": utc_now(),
        "engine_bin": str(engine_bin),
        "packed": "startpos",
        "rows": rows,
        "notes": "Pick node budgets whose wall_ms is near 200 and 800 on this host.",
    }
    out = out_path or (REPORT_DIR / "fidelity_calibration.json")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


@dataclass
class CertificationSkeleton:
    semantic_reset_1000: str = "NOT_RUN"
    audit_partial_1800: str = "NOT_RUN"
    audit_exhaustive_450: str = "NOT_RUN"
    drift_canary_partial_180: str = "NOT_RUN"
    drift_canary_exhaustive_45: str = "NOT_RUN"


def run_certification_skeleton(*, out_path: Path | None = None) -> dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    skeleton = CertificationSkeleton()
    payload = {
        "spec": "FLYWHEEL_SPEC_V1 §0",
        "generated_at": utc_now(),
        "cost_pilot": json.loads((REPORT_DIR / "cost_pilot_report.json").read_text(encoding="utf-8"))
        if (REPORT_DIR / "cost_pilot_report.json").is_file()
        else None,
        "fidelity_calibration": json.loads((REPORT_DIR / "fidelity_calibration.json").read_text(encoding="utf-8"))
        if (REPORT_DIR / "fidelity_calibration.json").is_file()
        else None,
        "phase_b": asdict(skeleton),
        "mass_generation_allowed": False,
    }
    out = out_path or (REPORT_DIR / "label_certification_report.json")
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    pilot = sub.add_parser("cost-pilot", help="10-root §0 cost pilot")
    pilot.add_argument("--db", type=Path, default=TEACHER_STORE_DB)
    pilot.add_argument("--roots", type=int, default=10)
    pilot.add_argument("--node-budget", type=int, default=20_000)
    pilot.add_argument("--engine-bin", type=Path, default=ENGINE_BIN)
    pilot.add_argument("--out", type=Path, default=REPORT_DIR / "cost_pilot_report.json")

    cal = sub.add_parser("fidelity-calibration", help="startpos wall-time vs node budget")
    cal.add_argument("--engine-bin", type=Path, default=ENGINE_BIN)
    cal.add_argument("--out", type=Path, default=REPORT_DIR / "fidelity_calibration.json")

    skel = sub.add_parser("skeleton-report", help="merge pilots into §0 report skeleton")
    skel.add_argument("--out", type=Path, default=REPORT_DIR / "label_certification_report.json")

    args = parser.parse_args()
    if args.cmd == "cost-pilot":
        report = run_cost_pilot(
            db_path=args.db,
            roots=args.roots,
            node_budget=args.node_budget,
            engine_bin=args.engine_bin,
            out_path=args.out,
        )
        print(json.dumps({"passed": not report.kill_reasons, "kill_reasons": report.kill_reasons}, indent=2))
        return 0 if not report.kill_reasons else 1
    if args.cmd == "fidelity-calibration":
        run_fidelity_calibration(engine_bin=args.engine_bin, out_path=args.out)
        return 0
    if args.cmd == "skeleton-report":
        run_certification_skeleton(out_path=args.out)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
