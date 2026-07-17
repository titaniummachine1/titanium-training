#!/usr/bin/env python3
"""Benchmark Ace epoch15000 forward latency and compare Titanium variants.

Phase 1 (always): JS forward bench via ace_harness.mjs
Phase 2 (optional): engine_variant_pipeline smoke for Titanium A/B

Usage:
  python training/tools/ka_teacher/ace_benchmark.py --forward-repeats 30
  python training/tools/ka_teacher/ace_benchmark.py --run-titanium-smoke
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_TRAINING = _REPO / "training"
_EXTRACT = _TRAINING / "tools" / "ka_teacher" / "extract_ace_runtime.js"
_HARNESS = _TRAINING / "tools" / "ka_teacher" / "ace_harness.mjs"
_PIPELINE = _REPO / "tools" / "binary_match" / "engine_variant_pipeline.py"
_OUT = _TRAINING / "data" / "ka_teacher_benchmark"


def run_node(script: Path, *args: str) -> dict:
    proc = subprocess.run(
        ["node", str(script), *args],
        cwd=str(_REPO),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout or f"node failed: {script}")
    return json.loads(proc.stdout)


def main() -> int:
    ap = argparse.ArgumentParser(description="Ace forward + optional Titanium smoke benchmark")
    ap.add_argument("--forward-repeats", type=int, default=20)
    ap.add_argument("--run-titanium-smoke", action="store_true")
    ap.add_argument("--engine-a", default="titanium-v17-route-touch")
    ap.add_argument("--engine-b", default="titanium-v17")
    ap.add_argument("--games", type=int, default=10)
    args = ap.parse_args()
    _OUT.mkdir(parents=True, exist_ok=True)

    if not (_REPO / "reference" / "ace.html").is_file():
        print("reference/ace.html missing", file=sys.stderr)
        return 2

    subprocess.run(["node", str(_EXTRACT)], cwd=str(_REPO), check=True)
    forward = run_node(_HARNESS, "--bench", "--repeats", str(args.forward_repeats))
    forward_path = _OUT / "ace_forward_bench.json"
    forward_path.write_text(json.dumps(forward, indent=2), encoding="utf-8")
    print(f"Ace JS forward: {forward.get('per_call_ms', 0):.2f} ms/call ({forward.get('repeats')} repeats)")

    report: dict = {"forward": forward, "titanium_smoke": None}
    if args.run_titanium_smoke:
        smoke_dir = _OUT / f"smoke_{args.engine_a}_vs_{args.engine_b}"
        proc = subprocess.run(
            [
                sys.executable,
                str(_PIPELINE),
                "--engine-a",
                args.engine_a,
                "--engine-b",
                args.engine_b,
                "--games",
                str(args.games),
                "--threads",
                "4",
                "--time",
                "0.5",
                "--out-dir",
                str(smoke_dir),
            ],
            cwd=str(_REPO),
            check=False,
        )
        summary_path = smoke_dir / "summary.json"
        if summary_path.is_file():
            report["titanium_smoke"] = json.loads(summary_path.read_text(encoding="utf-8"))
        report["titanium_smoke_exit"] = proc.returncode

    report_path = _OUT / "ace_benchmark_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
