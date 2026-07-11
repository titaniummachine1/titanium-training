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

from score_out_labels import PROTOCOL, _read_rows, _value_i16, collect_labels  # noqa: E402
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


def _label_fingerprint(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "bound": result.get("bound"),
        "value_i16": _value_i16(result),
        "score": result.get("score"),
        "nodes": result.get("nodes"),
        "depth": result.get("depth"),
        "selected_move": result.get("selected_move"),
        "schema": result.get("schema"),
    }


def run_semantic_reset_equivalence(
    *,
    db_path: Path = TEACHER_STORE_DB,
    roots: int = 1000,
    node_budget: int = 20_000,
    engine_bin: Path = ENGINE_BIN,
    out_path: Path | None = None,
) -> dict[str, Any]:
    """Fresh-vs-fresh score-out equivalence: two independent subprocess labels must match."""
    if not engine_bin.is_file():
        raise FileNotFoundError(engine_bin)
    rows = _read_rows(db_path, roots)
    mismatches: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for position_id, _canonical, packed, side_to_move in rows:
        packed_hex = packed.hex()
        a, _ = _score_out_probe(engine_bin, packed_hex, node_budget)
        b, _ = _score_out_probe(engine_bin, packed_hex, node_budget)
        if a is None or b is None:
            mismatches.append({"position_id": position_id, "reason": "score-out failed"})
            continue
        if a.get("side_to_move") != side_to_move or b.get("side_to_move") != side_to_move:
            mismatches.append({"position_id": position_id, "reason": "side_to_move mismatch"})
            continue
        fa, fb = _label_fingerprint(a), _label_fingerprint(b)
        if fa != fb:
            mismatches.append(
                {"position_id": position_id, "reason": "semantic mismatch", "first": fa, "second": fb}
            )
    wall_sec = time.perf_counter() - t0
    passed = not mismatches
    payload = {
        "spec": "FLYWHEEL_SPEC_V1 §0 semantic-reset equivalence",
        "generated_at": utc_now(),
        "engine_bin": str(engine_bin),
        "source_db": str(db_path),
        "roots_requested": roots,
        "roots_checked": len(rows),
        "node_budget": node_budget,
        "mismatch_count": len(mismatches),
        "wall_sec": round(wall_sec, 2),
        "passed": passed,
        "first_mismatches": mismatches[:5],
    }
    out = out_path or (REPORT_DIR / "semantic_reset_1000.json")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not passed:
        write_pause("flywheel_semantic_reset_failed", details={"mismatch_count": len(mismatches)})
    return payload


def run_label_audit(
    *,
    db_path: Path = TEACHER_STORE_DB,
    count: int,
    node_budget: int,
    tag: str,
    engine_bin: Path = ENGINE_BIN,
    out_path: Path | None = None,
) -> dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    labels_path = REPORT_DIR / f".audit_{tag}_{count}_{node_budget}.jsonl"
    if labels_path.exists():
        labels_path.unlink()
    t0 = time.perf_counter()
    summary = collect_labels(
        db_path,
        labels_path,
        max_positions=count,
        node_budget=node_budget,
        engine_bin=engine_bin,
    )
    wall_sec = time.perf_counter() - t0
    records = summary.get("records", 0)
    selected = summary.get("selected", count)
    exact_rate = records / selected if selected else 0.0
    throughput = records / wall_sec if wall_sec > 0 else 0.0
    passed = records > 0 and exact_rate >= 0.99
    payload = {
        "spec": f"FLYWHEEL_SPEC_V1 §0 label audit ({tag})",
        "generated_at": utc_now(),
        "tag": tag,
        "node_budget": node_budget,
        "selected": selected,
        "records": records,
        "skipped": summary.get("skipped", 0),
        "exact_rate": round(exact_rate, 4),
        "wall_sec": round(wall_sec, 2),
        "throughput_rows_per_sec": round(throughput, 4),
        "labels_path": str(labels_path),
        "passed": passed,
    }
    out = out_path or (REPORT_DIR / f"audit_{tag}.json")
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not passed:
        write_pause(f"flywheel_audit_{tag}_failed", details=payload)
    return payload


def run_drift_canary(
    *,
    labels_path: Path,
    sample: int,
    tag: str,
    node_budget: int,
    engine_bin: Path = ENGINE_BIN,
    out_path: Path | None = None,
) -> dict[str, Any]:
    if not labels_path.is_file():
        raise FileNotFoundError(labels_path)
    lines = [ln for ln in labels_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    picks = lines[:sample]
    mismatches: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for line in picks:
        rec = json.loads(line)
        packed_hex = rec["packed_state_hex"]
        side_to_move = rec["side_to_move"]
        node_budget_row = rec.get("protocol", {}).get("node_budget", node_budget)
        fresh, _ = _score_out_probe(engine_bin, packed_hex, int(node_budget_row))
        if fresh is None:
            mismatches.append({"position_id": rec.get("position_id"), "reason": "score-out failed"})
            continue
        if fresh.get("schema") != PROTOCOL:
            mismatches.append({"position_id": rec.get("position_id"), "reason": "schema drift"})
            continue
        stored_fp = {
            "bound": rec.get("bound"),
            "value_i16": rec.get("value_i16"),
            "score": rec.get("score"),
            "nodes": rec.get("nodes"),
            "depth": rec.get("depth"),
            "selected_move": rec.get("selected_move"),
            "schema": PROTOCOL,
        }
        fresh_fp = _label_fingerprint(fresh)
        if fresh.get("side_to_move") != side_to_move or stored_fp != fresh_fp:
            mismatches.append(
                {
                    "position_id": rec.get("position_id"),
                    "reason": "drift",
                    "stored": stored_fp,
                    "fresh": fresh_fp,
                }
            )
    wall_sec = time.perf_counter() - t0
    passed = not mismatches
    payload = {
        "spec": f"FLYWHEEL_SPEC_V1 §0 drift canary ({tag})",
        "generated_at": utc_now(),
        "tag": tag,
        "sample": len(picks),
        "node_budget": node_budget,
        "mismatch_count": len(mismatches),
        "wall_sec": round(wall_sec, 2),
        "passed": passed,
        "first_mismatches": mismatches[:5],
    }
    out = out_path or (REPORT_DIR / f"drift_canary_{tag}.json")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not passed:
        write_pause(f"flywheel_drift_canary_{tag}_failed", details={"mismatch_count": len(mismatches)})
    return payload


def run_gen0_pilot(
    *,
    db_path: Path = TEACHER_STORE_DB,
    roots: int = 15_000,
    node_budget: int = 20_000,
    engine_bin: Path = ENGINE_BIN,
    out_path: Path | None = None,
) -> dict[str, Any]:
    """Bounded Gen-0 pilot corpus (not 200k) once §0 gates pass."""
    flywheel_dir = TRAINING_ROOT / "data" / "flywheel"
    flywheel_dir.mkdir(parents=True, exist_ok=True)
    labels_path = out_path or (flywheel_dir / f"gen0_pilot_{roots}_{node_budget}.jsonl")
    if labels_path.exists():
        raise FileExistsError(f"refusing to overwrite pilot corpus: {labels_path}")
    t0 = time.perf_counter()
    summary = collect_labels(
        db_path,
        labels_path,
        max_positions=roots,
        node_budget=node_budget,
        engine_bin=engine_bin,
    )
    wall_sec = time.perf_counter() - t0
    records = int(summary.get("records", 0))
    selected = int(summary.get("selected", roots))
    payload = {
        "spec": "FLYWHEEL_SPEC_V1 Gen-0 pilot (bounded)",
        "generated_at": utc_now(),
        "roots_requested": roots,
        "selected": selected,
        "records": records,
        "skipped": summary.get("skipped", 0),
        "node_budget": node_budget,
        "wall_sec": round(wall_sec, 2),
        "labels_path": str(labels_path),
        "passed": records >= roots * 0.95,
    }
    manifest = flywheel_dir / "gen0_pilot_manifest.json"
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _phase_b_status(path: Path) -> str:
    if not path.is_file():
        return "NOT_RUN"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return "ERROR"
    return "PASSED" if data.get("passed") else "FAILED"


def run_phase_b_all(
    *,
    db_path: Path = TEACHER_STORE_DB,
    engine_bin: Path = ENGINE_BIN,
    semantic_roots: int = 1000,
    partial_count: int = 1800,
    exhaustive_count: int = 450,
    partial_budget: int = 20_000,
    exhaustive_budget: int = 200_000,
    start_gen0_pilot: bool = True,
    gen0_roots: int = 15_000,
) -> dict[str, Any]:
    """Run full §0 Phase B sequence and refresh certification report."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    semantic = run_semantic_reset_equivalence(
        db_path=db_path, roots=semantic_roots, node_budget=partial_budget, engine_bin=engine_bin
    )
    partial = run_label_audit(
        db_path=db_path,
        count=partial_count,
        node_budget=partial_budget,
        tag="partial",
        engine_bin=engine_bin,
    )
    exhaustive = run_label_audit(
        db_path=db_path,
        count=exhaustive_count,
        node_budget=exhaustive_budget,
        tag="exhaustive",
        engine_bin=engine_bin,
    )
    drift_p = run_drift_canary(
        labels_path=Path(partial["labels_path"]),
        sample=180,
        tag="partial_180",
        node_budget=partial_budget,
        engine_bin=engine_bin,
    )
    drift_e = run_drift_canary(
        labels_path=Path(exhaustive["labels_path"]),
        sample=45,
        tag="exhaustive_45",
        node_budget=exhaustive_budget,
        engine_bin=engine_bin,
    )
    phase_b = {
        "semantic_reset_1000": "PASSED" if semantic["passed"] else "FAILED",
        "audit_partial_1800": "PASSED" if partial["passed"] else "FAILED",
        "audit_exhaustive_450": "PASSED" if exhaustive["passed"] else "FAILED",
        "drift_canary_partial_180": "PASSED" if drift_p["passed"] else "FAILED",
        "drift_canary_exhaustive_45": "PASSED" if drift_e["passed"] else "FAILED",
    }
    all_pass = all(v == "PASSED" for v in phase_b.values())
    cost_pilot = None
    cost_path = REPORT_DIR / "cost_pilot_report.json"
    if cost_path.is_file():
        cost_pilot = json.loads(cost_path.read_text(encoding="utf-8"))
    cost_ok = bool(cost_pilot and cost_pilot.get("exact_pass"))
    mass_allowed = all_pass and cost_ok
    gen0_manifest = None
    if mass_allowed and start_gen0_pilot:
        gen0_manifest = run_gen0_pilot(
            db_path=db_path, roots=gen0_roots, node_budget=partial_budget, engine_bin=engine_bin
        )
    payload = {
        "spec": "FLYWHEEL_SPEC_V1 §0",
        "generated_at": utc_now(),
        "cost_pilot": cost_pilot,
        "fidelity_calibration": json.loads((REPORT_DIR / "fidelity_calibration.json").read_text(encoding="utf-8"))
        if (REPORT_DIR / "fidelity_calibration.json").is_file()
        else None,
        "phase_b": phase_b,
        "mass_generation_allowed": mass_allowed,
        "gen0_pilot": gen0_manifest,
    }
    out = REPORT_DIR / "label_certification_report.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not all_pass:
        write_pause("flywheel_phase_b_failed", details={"phase_b": phase_b})
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
    phase_b = {
        "semantic_reset_1000": _phase_b_status(REPORT_DIR / "semantic_reset_1000.json"),
        "audit_partial_1800": _phase_b_status(REPORT_DIR / "audit_partial.json"),
        "audit_exhaustive_450": _phase_b_status(REPORT_DIR / "audit_exhaustive.json"),
        "drift_canary_partial_180": _phase_b_status(REPORT_DIR / "drift_canary_partial_180.json"),
        "drift_canary_exhaustive_45": _phase_b_status(REPORT_DIR / "drift_canary_exhaustive_45.json"),
    }
    cost_pilot = None
    cost_path = REPORT_DIR / "cost_pilot_report.json"
    if cost_path.is_file():
        cost_pilot = json.loads(cost_path.read_text(encoding="utf-8"))
    cost_ok = bool(cost_pilot and cost_pilot.get("exact_pass"))
    mass_allowed = cost_ok and all(v == "PASSED" for v in phase_b.values())
    payload = {
        "spec": "FLYWHEEL_SPEC_V1 §0",
        "generated_at": utc_now(),
        "cost_pilot": json.loads((REPORT_DIR / "cost_pilot_report.json").read_text(encoding="utf-8"))
        if (REPORT_DIR / "cost_pilot_report.json").is_file()
        else None,
        "fidelity_calibration": json.loads((REPORT_DIR / "fidelity_calibration.json").read_text(encoding="utf-8"))
        if (REPORT_DIR / "fidelity_calibration.json").is_file()
        else None,
        "phase_b": phase_b,
        "mass_generation_allowed": mass_allowed,
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

    sem = sub.add_parser("semantic-reset", help="1000-root fresh-vs-fresh equivalence")
    sem.add_argument("--db", type=Path, default=TEACHER_STORE_DB)
    sem.add_argument("--roots", type=int, default=1000)
    sem.add_argument("--node-budget", type=int, default=20_000)
    sem.add_argument("--engine-bin", type=Path, default=ENGINE_BIN)

    aud = sub.add_parser("label-audit", help="partial or exhaustive label audit")
    aud.add_argument("--db", type=Path, default=TEACHER_STORE_DB)
    aud.add_argument("--count", type=int, required=True)
    aud.add_argument("--node-budget", type=int, required=True)
    aud.add_argument("--tag", type=str, required=True, choices=("partial", "exhaustive"))
    aud.add_argument("--engine-bin", type=Path, default=ENGINE_BIN)

    drift = sub.add_parser("drift-canary", help="re-label sample from audit corpus")
    drift.add_argument("--labels", type=Path, required=True)
    drift.add_argument("--sample", type=int, required=True)
    drift.add_argument("--tag", type=str, required=True)
    drift.add_argument("--node-budget", type=int, default=20_000)
    drift.add_argument("--engine-bin", type=Path, default=ENGINE_BIN)

    pball = sub.add_parser("phase-b-all", help="run full §0 Phase B; start Gen-0 pilot if gates pass")
    pball.add_argument("--db", type=Path, default=TEACHER_STORE_DB)
    pball.add_argument("--engine-bin", type=Path, default=ENGINE_BIN)
    pball.add_argument("--no-gen0-pilot", action="store_true")

    gen0 = sub.add_parser("gen0-pilot", help="bounded Gen-0 pilot labeling")
    gen0.add_argument("--db", type=Path, default=TEACHER_STORE_DB)
    gen0.add_argument("--roots", type=int, default=15_000)
    gen0.add_argument("--node-budget", type=int, default=20_000)
    gen0.add_argument("--engine-bin", type=Path, default=ENGINE_BIN)
    gen0.add_argument("--out", type=Path, default=None)

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
    if args.cmd == "semantic-reset":
        result = run_semantic_reset_equivalence(
            db_path=args.db,
            roots=args.roots,
            node_budget=args.node_budget,
            engine_bin=args.engine_bin,
        )
        print(json.dumps({"passed": result["passed"], "mismatch_count": result["mismatch_count"]}, indent=2))
        return 0 if result["passed"] else 1
    if args.cmd == "label-audit":
        result = run_label_audit(
            db_path=args.db,
            count=args.count,
            node_budget=args.node_budget,
            tag=args.tag,
            engine_bin=args.engine_bin,
        )
        print(json.dumps(result, indent=2))
        return 0 if result["passed"] else 1
    if args.cmd == "drift-canary":
        result = run_drift_canary(
            labels_path=args.labels,
            sample=args.sample,
            tag=args.tag,
            node_budget=args.node_budget,
            engine_bin=args.engine_bin,
        )
        print(json.dumps({"passed": result["passed"], "mismatch_count": result["mismatch_count"]}, indent=2))
        return 0 if result["passed"] else 1
    if args.cmd == "phase-b-all":
        result = run_phase_b_all(
            db_path=args.db,
            engine_bin=args.engine_bin,
            start_gen0_pilot=not args.no_gen0_pilot,
        )
        print(json.dumps({"mass_generation_allowed": result["mass_generation_allowed"], "phase_b": result["phase_b"]}, indent=2))
        return 0 if result["mass_generation_allowed"] else 1
    if args.cmd == "gen0-pilot":
        result = run_gen0_pilot(
            db_path=args.db,
            roots=args.roots,
            node_budget=args.node_budget,
            engine_bin=args.engine_bin,
            out_path=args.out,
        )
        print(json.dumps(result, indent=2))
        return 0 if result["passed"] else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
