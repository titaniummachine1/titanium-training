#!/usr/bin/env python3
"""Preflight + overnight status report for unattended operation."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_TRAINING = _REPO / "training"
_DATA = _TRAINING / "data"
_LOGS = _DATA / "overnight_logs"


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def run_cat_tests() -> dict:
    env = dict(**{k: v for k, v in __import__("os").environ.items()})
    env["RUSTFLAGS"] = "-C target-cpu=native"
    proc = subprocess.run(
        ["cargo", "test", "-p", "titanium", "cat_", "--", "--nocapture"],
        cwd=_REPO / "engine",
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )
    return {
        "passed": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-2000:],
    }


def cat_nps_benchmark() -> dict:
    """Quick bench before CAT deploy — compare to baseline if present."""
    baseline_path = _LOGS / "cat_nps_baseline.json"
    env = dict(**{k: v for k, v in __import__("os").environ.items()})
    env["RUSTFLAGS"] = "-C target-cpu=native"
    proc = subprocess.run(
        ["cargo", "run", "--release", "-p", "titanium", "--", "bench", "4", "3"],
        cwd=_REPO / "engine",
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    nps = None
    for line in proc.stdout.splitlines():
        if "nps=" in line:
            try:
                nps = float(line.split("nps=")[-1].strip())
            except ValueError:
                pass
    result = {"nps": nps, "ok": proc.returncode == 0}
    if baseline_path.is_file():
        base = read_json(baseline_path).get("nps")
        if base and nps:
            result["overhead_pct"] = round((base - nps) / base * 100.0, 2)
    else:
        if nps:
            baseline_path.write_text(json.dumps({"nps": nps}, indent=2) + "\n", encoding="utf-8")
            result["baseline_saved"] = True
    return result


def local_promotion_hashes() -> dict:
    current = _TRAINING / "runs" / "value_oracle" / "net_weights_best.bin"
    previous = _TRAINING / "runs" / "value_oracle" / "net_weights_previous.bin"
    cur = sha256_file(current)
    prev = sha256_file(previous)
    return {
        "current_sha256": cur,
        "previous_sha256": prev,
        "distinct": cur != prev if cur and prev else None,
        "current_path": str(current),
        "previous_path": str(previous),
    }


def cache_rebuild_state() -> dict:
    report = read_json(_LOGS / "safe_rebuild_report.json")
    watcher = read_json(_LOGS / "safe_rebuild_watcher_state.json")
    pause = (_LOGS / "pause_training_epochs.json").is_file()
    opening = (_DATA / "opening_exploration_enabled.json").is_file()
    live_rows = None
    manifest = _DATA / "feature_cache" / "manifest.json"
    if manifest.is_file():
        live_rows = read_json(manifest).get("row_count")
    return {
        "safe_rebuild_report": report,
        "watcher_state": watcher,
        "training_paused": pause,
        "opening_exploration_enabled": opening,
        "live_cache_rows": live_rows,
    }


def opening_metrics() -> dict:
    metrics_path = _LOGS / "opening_exploration_metrics.jsonl"
    if not metrics_path.is_file():
        return {"novelty_rate": None, "median_novelty_exit_ply": None}
    novel = 0
    total = 0
    exit_plies: list[int] = []
    for line in metrics_path.read_text(encoding="utf-8").splitlines()[-5000:]:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        total += 1
        if row.get("novelty_found"):
            novel += 1
        if row.get("novelty_exit_ply") is not None:
            exit_plies.append(int(row["novelty_exit_ply"]))
    exit_plies.sort()
    median = exit_plies[len(exit_plies) // 2] if exit_plies else None
    return {
        "novelty_rate": (novel / total) if total else None,
        "median_novelty_exit_ply": median,
        "sample_games": total,
    }


def ingestion_rate() -> dict:
    importer_log = _LOGS / "continuous_pool.log"
    if not importer_log.is_file():
        return {"note": "continuous_pool.log not found"}
    tail = importer_log.read_text(encoding="utf-8", errors="replace").splitlines()[-50:]
    return {"log_tail": tail}


def build_report(*, run_cat: bool, run_bench: bool) -> dict:
    weights = local_promotion_hashes()
    report: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "promoted_weights": weights,
        "weight_upload": {"note": "Run push_oracle_generation.ps1; uploads skipped when remote hash matches"},
        "oracle_workers": {
            "expected": 13,
            "note": "Verify on VM: curl localhost:8765/status or supervisor_status.json",
        },
        "game_mixture": {
            "target_selfplay_pct": 70,
            "target_mixed_pct": 30,
            "note": "Supervisor uses choose_matchup(game_id) per game",
        },
        "search_budget": {"nodes_per_move": 200_000, "move_time_sec": 5.0, "timeout_is_fallback": True},
        "cache_rebuild": cache_rebuild_state(),
        "opening_metrics": opening_metrics(),
        "ingestion": ingestion_rate(),
        "rollback_command": (
            "Copy-Item training\\data\\feature_cache_pre_v2_backup -Recurse "
            "training\\data\\feature_cache -Force; "
            "Remove-Item training\\data\\overnight_logs\\pause_training_epochs.json -ErrorAction SilentlyContinue"
        ),
        "oracle_emergency_stop": (
            ".\\stop_oracle_worker.ps1 -Host <ORACLE_HOST>"
        ),
        "cat_deploy_gate": {
            "replace_production_binary": False,
            "requires": [
                "cat unit tests pass",
                "deterministic search",
                "legal move count unchanged",
                "NPS regression measured",
                "20-game smoke test",
            ],
        },
    }
    if run_cat:
        report["cat_tests"] = run_cat_tests()
    if run_bench:
        report["cat_nps"] = cat_nps_benchmark()
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-cat-tests", action="store_true")
    ap.add_argument("--run-cat-bench", action="store_true")
    ap.add_argument(
        "--out",
        type=Path,
        default=_LOGS / "overnight_status_report.json",
    )
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = build_report(run_cat=args.run_cat_tests, run_bench=args.run_cat_bench)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report.get("cat_tests") and not report["cat_tests"].get("passed"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
