#!/usr/bin/env python3
"""Overnight classical HCE improvement loop.

1. d6 node-cost bench (search_bench + TITANIUM_BENCH_ENGINE)
2. 200-game A/B: 4 local workers (shard 0-3) + 13 Oracle shards (4-16)
3. Unified pool: local shards 0–3 + Oracle 4–16 → one merged status.json
4. PROMOTE if score > 0.5 after target games (any positive Elo); DOUBLE if inconclusive (max 1600)

Does not touch NN training. Pauses flywheel labeling during active HCE matches.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
ENGINE = REPO / "engine"
BENCH_BIN = ENGINE / "target" / "release" / "search_bench.exe"
TITANIUM_BIN = ENGINE / "target" / "release" / "titanium.exe"
MATCH_PY = REPO / "tools" / "binary_match" / "parallel_engine_match.py"
ORACLE_LAUNCH = REPO / "tools" / "binary_match" / "launch_oracle_shards.ps1"
LOG_DIR = REPO / "training" / "data" / "overnight_logs"
STATE_FILE = LOG_DIR / "engine_improve_state.json"
PAUSE_FILE = LOG_DIR / "TRAINING_PAUSED.json"
FLYWHEEL_PAUSE_REASON = "engine_hce_match_running"
RUNS_ROOT = REPO / "tools" / "binary_match" / "runs" / "overnight_engine"

BENCH_POSITIONS = (
    "startpos",
    "c3h-midgame",
    "wall-maze",
    "low-wall",
    "endgame-c5",
    "dense-maze",
)
BASELINE_ENGINE = "titanium-v17"
INITIAL_GAMES = 200
MAX_GAMES = 1600
PROMOTE_SCORE = 0.5  # score strictly above → promote (even ~+3 Elo)
REJECT_SCORE = 0.48  # clearly worse
INCONCLUSIVE_LOW = 0.48  # double when score in [0.48, 0.5]
NEUTRAL_LB_FLOOR = 0.45  # efficiency-promote floor: not proven worse
MIN_NODE_FACTOR = 1.03  # ≥3% fewer nodes @ fixed d6
MIN_NPS_FACTOR = 1.02  # ≥2% higher NPS @ fixed wall time
MIN_DEPTH_GAIN = 1  # ≥1 ply deeper at same wall time
ORACLE_KEY = Path(os.environ.get("TITANIUM_ORACLE_SSH_KEY", Path.home() / ".ssh" / "oracle_titanium.key"))


@dataclass
class Candidate:
    candidate_id: str
    engine_a: str
    engine_b: str = BASELINE_ENGINE
    note: str = ""
    bench_only_skip_match: bool = False


DEFAULT_CANDIDATES: list[Candidate] = [
    Candidate("probcut", "titanium-v17-probcut", note="ProbCut verification"),
    Candidate("route-touch", "titanium-v17-route-touch", note="route-touch wall ordering"),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def pause_flywheel_labeling() -> dict[str, Any]:
    """Stop flywheel cert/labeling so titanium.exe is free for clean HCE matches."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ps = (
        "$killed = @(); "
        "Get-CimInstance Win32_Process -Filter \"name='python.exe'\" -ErrorAction SilentlyContinue | "
        "Where-Object { $_.CommandLine -match 'flywheel_label_cert' } | "
        "ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue; $killed += $_.ProcessId }; "
        "Get-Process titanium -ErrorAction SilentlyContinue | "
        "ForEach-Object { Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue; $killed += $_.Id }; "
        "$killed | ConvertTo-Json -Compress"
    )
    proc = subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        killed = json.loads(proc.stdout.strip() or "[]")
    except json.JSONDecodeError:
        killed = []
    payload = {
        "paused": True,
        "reason": FLYWHEEL_PAUSE_REASON,
        "updated_at": utc_now(),
        "note": "Paused for classical HCE A/B match; resume after engine_improve poll completes",
        "killed_pids": killed,
    }
    PAUSE_FILE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def resume_flywheel_labeling() -> None:
    if not PAUSE_FILE.is_file():
        return
    try:
        row = json.loads(PAUSE_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    if row.get("reason") == FLYWHEEL_PAUSE_REASON:
        PAUSE_FILE.unlink(missing_ok=True)


def wilson_interval(successes: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    p = successes / n
    z2 = z * z
    denom = 1 + z2 / n
    center = p + z2 / (2 * n)
    margin = z * math.sqrt((p * (1 - p) + z2 / (4 * n)) / n)
    return (center - margin) / denom, (center + margin) / denom


def estimate_elo_delta(score_a: float) -> float:
    """Logistic Elo delta for A from match score (no draws)."""
    if score_a <= 0.0 or score_a >= 1.0:
        return 0.0
    return -400.0 * math.log10(1.0 / score_a - 1.0)


def shard_status_paths(run_dir: Path) -> list[Path]:
    paths: list[Path] = []
    local_status = run_dir / "local" / "status.json"
    if local_status.is_file():
        paths.append(local_status)
    oracle_dir = run_dir / "oracle"
    if oracle_dir.is_dir():
        paths.extend(sorted(oracle_dir.glob("status_shard_*.json")))
    return paths


def merge_status_files(paths: list[Path]) -> dict[str, Any]:
    a_wins = b_wins = draws = errors = completed = 0
    for path in paths:
        if not path.is_file():
            continue
        row = json.loads(path.read_text(encoding="utf-8"))
        a_wins += int(row.get("a_wins", 0))
        b_wins += int(row.get("b_wins", 0))
        draws += int(row.get("draws", 0))
        errors += int(row.get("errors", 0))
        completed += int(row.get("completed_games", 0))
    n = a_wins + b_wins + draws
    score_a = (a_wins + 0.5 * draws) / n if n else 0.0
    lb, ub = wilson_interval(a_wins + 0.5 * draws, n)
    return {
        "a_wins": a_wins,
        "b_wins": b_wins,
        "draws": draws,
        "errors": errors,
        "completed_games": completed,
        "decisive_games": n,
        "score_a": round(score_a, 4),
        "wilson_lb_a": round(lb, 4),
        "wilson_ub_a": round(ub, 4),
    }


def assigned_game_count(total_games: int, shard: dict[str, Any] | None) -> int | None:
    """Return how many stable game IDs belong to one shard or shard span."""
    if not shard:
        return None
    count = int(shard.get("count", 0))
    offset = int(shard.get("offset", 0))
    span = int(shard.get("span", 0))
    if total_games < 0 or count <= 0 or offset < 0 or span <= 0 or offset + span > count:
        return None
    assigned = 0
    for residue in range(offset, offset + span):
        if residue < total_games:
            assigned += (total_games - 1 - residue) // count + 1
    return assigned


def run_time_bench(
    engine_flag: str,
    *,
    position: str = "startpos",
    sec: int = 10,
    runs: int = 3,
    extra_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    env = os.environ.copy()
    env["TITANIUM_BENCH_ENGINE"] = engine_flag
    env["RUSTFLAGS"] = "-C target-cpu=native"
    if extra_env:
        env.update(extra_env)
    proc = subprocess.run(
        [str(BENCH_BIN), "time", "--sec", str(sec), "--runs", str(runs), "--threads", "1", "--position", position],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"time bench failed for {engine_flag} {position}: {proc.stderr[-500:]}")
    row = json.loads(proc.stdout.strip().splitlines()[-1])
    return {
        "position": position,
        "median_nps": float(row.get("median_nps", row.get("nps", 0))),
        "median_depth": int(row.get("median_depth", row.get("depth", 0))),
        "median_nodes": int(row.get("median_nodes", row.get("nodes", 0))),
        "wall_sec": sec,
        "runs": runs,
    }


def efficiency_signals(bench_d6: dict[str, Any] | None, bench_time: dict[str, Any] | None) -> list[str]:
    signals: list[str] = []
    if bench_d6 and float(bench_d6.get("node_factor_vs_baseline", 0)) >= MIN_NODE_FACTOR:
        signals.append(f"d6_nodes_x{bench_d6['node_factor_vs_baseline']:.3f}")
    if bench_time:
        for pos, row in bench_time.get("per_position", {}).items():
            base = row.get("baseline") or {}
            cand = row.get("candidate") or {}
            base_nps = float(base.get("median_nps", 0))
            cand_nps = float(cand.get("median_nps", 0))
            base_depth = int(base.get("median_depth", 0))
            cand_depth = int(cand.get("median_depth", 0))
            if base_nps > 0 and cand_nps >= base_nps * MIN_NPS_FACTOR:
                signals.append(f"nps_{pos}_x{cand_nps / base_nps:.3f}")
            if cand_depth >= base_depth + MIN_DEPTH_GAIN and cand_nps >= base_nps * 0.98:
                signals.append(f"depth_{pos}_+{cand_depth - base_depth}")
    return signals


def match_verdict(
    summary: dict[str, Any],
    *,
    target_games: int,
    bench_d6: dict[str, Any] | None = None,
    bench_time: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Return {action, reason}. Promote on any positive score; double when inconclusive."""
    n = int(summary.get("decisive_games", 0))
    if n < target_games:
        return {"action": "RUNNING", "reason": "insufficient_games"}
    score = float(summary["score_a"])
    lb = float(summary["wilson_lb_a"])
    eff = efficiency_signals(bench_d6, bench_time)
    if score > PROMOTE_SCORE:
        return {"action": "PROMOTE", "reason": "positive_score"}
    if score < REJECT_SCORE:
        return {"action": "REJECT", "reason": "proven_weaker"}
    if lb >= NEUTRAL_LB_FLOOR and eff:
        return {"action": "PROMOTE", "reason": "efficiency_" + "+".join(eff)}
    if INCONCLUSIVE_LOW <= score <= PROMOTE_SCORE:
        return {"action": "DOUBLE", "reason": "inconclusive_score"}
    if n >= target_games:
        return {"action": "REJECT", "reason": "no_gain"}
    return {"action": "RUNNING", "reason": "in_progress"}


def build_native(*, allow_existing: bool = True) -> None:
    env = os.environ.copy()
    env["RUSTFLAGS"] = "-C target-cpu=native"
    try:
        subprocess.run(
            ["cargo", "build", "--release", "-p", "titanium", "--bin", "search_bench", "--bin", "titanium"],
            cwd=str(ENGINE),
            env=env,
            check=True,
        )
    except subprocess.CalledProcessError:
        if allow_existing and BENCH_BIN.is_file() and TITANIUM_BIN.is_file():
            return
        raise


def run_d6_bench(engine_flag: str, *, extra_env: dict[str, str] | None = None) -> dict[str, int]:
    env = os.environ.copy()
    env["TITANIUM_BENCH_ENGINE"] = engine_flag
    env["RUSTFLAGS"] = "-C target-cpu=native"
    if extra_env:
        env.update(extra_env)
    nodes_by_pos: dict[str, int] = {}
    for pos in BENCH_POSITIONS:
        proc = subprocess.run(
            [str(BENCH_BIN), "depth", "--depth", "6", "--position", pos, "--threads", "1"],
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"search_bench failed for {engine_flag} {pos}: {proc.stderr[-500:]}")
        row = json.loads(proc.stdout.strip().splitlines()[-1])
        nodes_by_pos[pos] = int(row["nodes"])
    return nodes_by_pos


def bench_compare(
    candidate: Candidate,
    *,
    extra_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    base_nodes = run_d6_bench(candidate.engine_b)
    cand_nodes = run_d6_bench(candidate.engine_a, extra_env=extra_env)
    total_base = sum(base_nodes.values())
    total_cand = sum(cand_nodes.values())
    factor = total_base / total_cand if total_cand else 0.0
    time_positions = ("startpos", "c3h-midgame")
    time_rows: dict[str, dict[str, Any]] = {}
    for pos in time_positions:
        time_rows[pos] = {
            "baseline": run_time_bench(candidate.engine_b, position=pos),
            "candidate": run_time_bench(candidate.engine_a, position=pos, extra_env=extra_env),
        }
    bench_time = {"per_position": time_rows}
    return {
        "baseline": candidate.engine_b,
        "candidate": candidate.engine_a,
        "baseline_total_nodes_d6": total_base,
        "candidate_total_nodes_d6": total_cand,
        "node_factor_vs_baseline": round(factor, 4),
        "per_position": {
            "baseline": base_nodes,
            "candidate": cand_nodes,
        },
        "time_bench": bench_time,
        "efficiency_signals": efficiency_signals(
            {"node_factor_vs_baseline": factor},
            bench_time,
        ),
    }


def _shard_contribution(path: Path) -> dict[str, Any]:
    row = json.loads(path.read_text(encoding="utf-8"))
    completed = int(row.get("completed_games", 0))
    assigned = assigned_game_count(int(row.get("target_games", 0)), row.get("shard"))
    running = bool(row.get("running", False))
    if assigned is not None and completed >= assigned:
        running = False
    return {
        "path": str(path),
        "completed_games": completed,
        "a_wins": int(row.get("a_wins", 0)),
        "b_wins": int(row.get("b_wins", 0)),
        "draws": int(row.get("draws", 0)),
        "running": running,
        "assigned_games": assigned,
        "shard": row.get("shard"),
    }


def collect_match_status(run_dir: Path, *, target_games: int | None = None) -> dict[str, Any]:
    paths = shard_status_paths(run_dir)
    merged = merge_status_files(paths)
    contributions = [_shard_contribution(p) for p in paths]
    local_paths = [p for p in paths if p.parent.name == "local"]
    oracle_paths = [p for p in paths if p.parent.name == "oracle"]
    local_part = merge_status_files(local_paths) if local_paths else {}
    oracle_part = merge_status_files(oracle_paths) if oracle_paths else {}
    workers_reporting_running = any(c.get("running") for c in contributions)
    if target_games is None:
        for p in paths:
            try:
                row = json.loads(p.read_text(encoding="utf-8"))
                target_games = int(row.get("target_games", 0)) or target_games
            except (json.JSONDecodeError, OSError):
                pass
    n = int(merged.get("decisive_games", 0))
    score = float(merged.get("score_a", 0.0))
    incomplete = bool(target_games and int(merged.get("completed_games", 0)) < target_games)
    pooled: dict[str, Any] = {
        **merged,
        "pool": "unified",
        "target_games": target_games,
        "running": incomplete or workers_reporting_running,
        "workers_reporting_running": workers_reporting_running,
        "shard_contributions": contributions,
        "local_pool": {
            "decisive_games": local_part.get("decisive_games", 0),
            "a_wins": local_part.get("a_wins", 0),
            "b_wins": local_part.get("b_wins", 0),
            "score_a": local_part.get("score_a", 0.0),
        },
        "oracle_pool": {
            "decisive_games": oracle_part.get("decisive_games", 0),
            "a_wins": oracle_part.get("a_wins", 0),
            "b_wins": oracle_part.get("b_wins", 0),
            "score_a": oracle_part.get("score_a", 0.0),
        },
        "elo_delta_a": round(estimate_elo_delta(score), 1) if n else 0.0,
        "status_files": [str(p) for p in paths],
        "updated_at": utc_now(),
    }
    (run_dir / "status.json").write_text(json.dumps(pooled, indent=2) + "\n", encoding="utf-8")
    (run_dir / "merged_summary.json").write_text(json.dumps(pooled, indent=2) + "\n", encoding="utf-8")
    return pooled


def launch_local_shard(
    *,
    engine_a: str,
    engine_b: str,
    games: int,
    out_dir: Path,
    seed: int,
) -> subprocess.Popen[Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["TITANIUM_ENGINE_BIN"] = str(TITANIUM_BIN)
    cmd = [
        sys.executable,
        str(MATCH_PY),
        "--engine-a",
        engine_a,
        "--engine-b",
        engine_b,
        "--games",
        str(games),
        "--clock-sec",
        "60",
        "--workers",
        "4",
        "--shard-count",
        "17",
        "--shard-offset",
        "0",
        "--shard-span",
        "4",
        "--seed",
        str(seed),
        "--out-dir",
        str(out_dir / "local"),
    ]
    log = out_dir / "local_launch.log"
    handle = open(log, "a", encoding="utf-8")
    return subprocess.Popen(cmd, cwd=str(REPO), env=env, stdout=handle, stderr=subprocess.STDOUT)


def launch_oracle_shards(
    *,
    engine_a: str,
    engine_b: str,
    games: int,
    out_dir: Path,
    run_id: str,
    engine_git_commit: str,
) -> bool:
    if not ORACLE_KEY.is_file():
        return False
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(ORACLE_LAUNCH),
        "-SshKeyPath",
        str(ORACLE_KEY),
        "-EngineA",
        engine_a,
        "-EngineB",
        engine_b,
        "-Games",
        str(games),
        "-RunId",
        run_id,
        "-EngineGitRef",
        engine_git_commit,
        "-LocalRunDir",
        str(out_dir / "oracle"),
        "-Mode",
        "launch",
    ]
    log = out_dir / "oracle_launch.log"
    with open(log, "a", encoding="utf-8") as handle:
        proc = subprocess.run(cmd, cwd=str(REPO), stdout=handle, stderr=subprocess.STDOUT, check=False)
    return proc.returncode == 0


def pull_oracle_results(run_dir: Path, run_id: str) -> None:
    if not ORACLE_KEY.is_file():
        return
    cmd = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(ORACLE_LAUNCH),
        "-SshKeyPath",
        str(ORACLE_KEY),
        "-RunId",
        run_id,
        "-LocalRunDir",
        str(run_dir / "oracle"),
        "-Mode",
        "pull",
    ]
    subprocess.run(cmd, cwd=str(REPO), check=False)


@dataclass
class LoopState:
    updated_at: str = field(default_factory=utc_now)
    baseline_engine: str = BASELINE_ENGINE
    active_run: dict[str, Any] | None = None
    queue: list[dict[str, Any]] = field(default_factory=list)
    completed: list[dict[str, Any]] = field(default_factory=list)
    promoted: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)


def recover_active_run_from_disk(state: LoopState) -> dict[str, Any] | None:
    """Reattach to the newest overnight run that still has shard status on disk."""
    promoted_ids = {row.get("id") for row in state.promoted}
    rejected_ids = set(state.rejected)
    completed_ids = {row.get("candidate_id") for row in state.completed}
    skip_ids = promoted_ids | rejected_ids | completed_ids
    if not RUNS_ROOT.is_dir():
        return None
    candidates: list[tuple[float, Path]] = []
    for run_dir in RUNS_ROOT.iterdir():
        if not run_dir.is_dir():
            continue
        name = run_dir.name
        cid = name[len("overnight_") :].split("_g", 1)[0] if name.startswith("overnight_") else ""
        if cid in skip_ids:
            continue
        if not shard_status_paths(run_dir):
            continue
        candidates.append((run_dir.stat().st_mtime, run_dir))
    if not candidates:
        return None
    run_dir = sorted(candidates, key=lambda x: x[0])[-1][1]
    name = run_dir.name
    games = 200
    if "_g" in name:
        try:
            games = int(name.split("_g", 1)[1].split("_", 1)[0])
        except ValueError:
            pass
    cid = "rfp-ace"
    if name.startswith("overnight_"):
        cid = name[len("overnight_") :].split("_g", 1)[0]
    engine_a = f"titanium-v17-{cid}" if cid != "rfp-ace" else "titanium-v17-rfp-ace"
    bench_d6 = None
    bench_time = None
    bench_path = run_dir / "bench_d6.json"
    if bench_path.is_file():
        bench = json.loads(bench_path.read_text(encoding="utf-8"))
        bench_d6 = {
            "node_factor_vs_baseline": bench.get("node_factor_vs_baseline"),
            "baseline_total_nodes_d6": bench.get("baseline_total_nodes_d6"),
            "candidate_total_nodes_d6": bench.get("candidate_total_nodes_d6"),
        }
        bench_time = bench.get("time_bench")
    oracle_launched = (run_dir / "oracle" / "launcher_manifest.json").is_file()
    return {
        "candidate_id": cid,
        "engine_a": engine_a,
        "engine_b": BASELINE_ENGINE,
        "games": games,
        "run_id": name,
        "run_dir": str(run_dir),
        "oracle_launched": oracle_launched,
        "started_at": utc_now(),
        "bench_d6": bench_d6,
        "bench_time": bench_time,
        "recovered": True,
    }


def load_state() -> LoopState:
    state = _load_state_raw()
    if state.active_run is None:
        recovered = recover_active_run_from_disk(state)
        if recovered:
            state.active_run = recovered
            save_state(state)
    return state


def _load_state_raw() -> LoopState:
    if not STATE_FILE.is_file():
        return LoopState(queue=[asdict(c) for c in DEFAULT_CANDIDATES])
    raw = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return LoopState(
        updated_at=raw.get("updated_at", utc_now()),
        baseline_engine=raw.get("baseline_engine", BASELINE_ENGINE),
        active_run=raw.get("active_run"),
        queue=raw.get("queue", []),
        completed=raw.get("completed", []),
        promoted=raw.get("promoted", []),
        rejected=raw.get("rejected", []),
    )


def save_state(state: LoopState) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    state.updated_at = utc_now()
    STATE_FILE.write_text(json.dumps(asdict(state), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def current_engine_commit() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD^{commit}"],
        cwd=str(ENGINE),
        capture_output=True,
        text=True,
        check=True,
    )
    commit = proc.stdout.strip()
    if len(commit) != 40 or any(ch not in "0123456789abcdef" for ch in commit.lower()):
        raise RuntimeError(f"unexpected engine commit identity: {commit!r}")
    return commit


def start_candidate_match(state: LoopState, cand: dict[str, Any], *, games: int) -> dict[str, Any]:
    pause_flywheel_labeling()
    time.sleep(2)
    build_native()
    engine_git_commit = current_engine_commit()
    cid = cand["candidate_id"]
    run_id = f"overnight_{cid}_g{games}_{int(time.time())}"
    run_dir = RUNS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    bench = bench_compare(Candidate(**{k: cand[k] for k in ("candidate_id", "engine_a", "engine_b", "note") if k in cand}))
    (run_dir / "bench_d6.json").write_text(json.dumps(bench, indent=2) + "\n", encoding="utf-8")
    bench_d6 = {
        "node_factor_vs_baseline": bench.get("node_factor_vs_baseline"),
        "baseline_total_nodes_d6": bench.get("baseline_total_nodes_d6"),
        "candidate_total_nodes_d6": bench.get("candidate_total_nodes_d6"),
    }
    bench_time = bench.get("time_bench")
    local_proc = launch_local_shard(
        engine_a=cand["engine_a"],
        engine_b=cand["engine_b"],
        games=games,
        out_dir=run_dir,
        seed=1337,
    )
    oracle_ok = launch_oracle_shards(
        engine_a=cand["engine_a"],
        engine_b=cand["engine_b"],
        games=games,
        out_dir=run_dir,
        run_id=run_id,
        engine_git_commit=engine_git_commit,
    )
    active = {
        "candidate_id": cid,
        "engine_a": cand["engine_a"],
        "engine_b": cand["engine_b"],
        "engine_git_commit": engine_git_commit,
        "games": games,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "local_pid": local_proc.pid,
        "oracle_launched": oracle_ok,
        "started_at": utc_now(),
        "bench_d6": bench_d6,
        "bench_time": bench_time,
        "efficiency_signals": bench.get("efficiency_signals", []),
    }
    state.active_run = active
    save_state(state)
    return active


def poll_active(state: LoopState) -> dict[str, Any] | None:
    active = state.active_run
    if not active:
        return None
    run_dir = Path(active["run_dir"])
    if active.get("oracle_launched"):
        pull_oracle_results(run_dir, active["run_id"])
    summary = collect_match_status(run_dir, target_games=int(active["games"]))
    verdict = match_verdict(
        summary,
        target_games=int(active["games"]),
        bench_d6=active.get("bench_d6"),
        bench_time=active.get("bench_time"),
    )
    report = {**summary, **verdict, "target_games": active["games"]}
    action = verdict["action"]
    if action == "RUNNING":
        save_state(state)
        return report
    if action != "DOUBLE":
        resume_flywheel_labeling()
    record = {**active, "final": report}
    state.completed.append(record)
    if action == "PROMOTE":
        state.promoted.append({"id": active["candidate_id"], "reason": verdict["reason"]})
    elif action == "REJECT":
        state.rejected.append(active["candidate_id"])
    elif action == "DOUBLE":
        next_games = min(int(active["games"]) * 2, MAX_GAMES)
        if next_games > int(active["games"]):
            state.active_run = None
            save_state(state)
            start_candidate_match(state, active, games=next_games)
            report["doubled_to"] = next_games
            return report
    state.active_run = None
    if state.queue and state.queue[0].get("candidate_id") == active["candidate_id"]:
        state.queue.pop(0)
    save_state(state)
    return report


def tick(state: LoopState) -> dict[str, Any]:
    if state.active_run:
        return {"action": "poll", "result": poll_active(state)} or {}
    if not state.queue:
        return {"action": "idle", "message": "queue empty"}
    cand = state.queue[0]
    if cand.get("bench_only_skip_match"):
        state.queue.pop(0)
        save_state(state)
        return {"action": "skipped", "candidate": cand}
    active = start_candidate_match(state, cand, games=INITIAL_GAMES)
    return {"action": "started", "active": active}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build", help="native build titanium + search_bench")
    bench = sub.add_parser("bench-baseline", help="d6 node totals for baseline v17")
    bench.add_argument("--engine", default=BASELINE_ENGINE)
    sub.add_parser("tick", help="poll active match or start next candidate")
    sub.add_parser("status", help="print loop state")
    poll = sub.add_parser("poll", help="poll active run only")
    start = sub.add_parser("start-rfp", help="pause flywheel and run rfp-ace vs v17 match now")
    start.add_argument("--games", type=int, default=INITIAL_GAMES)
    args = parser.parse_args()
    if args.cmd == "build":
        build_native()
        return 0
    state = load_state()
    if args.cmd == "bench-baseline":
        build_native()
        print(json.dumps(run_d6_bench(args.engine), indent=2))
        return 0
    if args.cmd == "status":
        print(json.dumps(asdict(state), indent=2))
        if state.active_run:
            print(json.dumps(collect_match_status(Path(state.active_run["run_dir"]), target_games=int(state.active_run["games"])), indent=2))
        return 0
    if args.cmd == "poll":
        print(json.dumps(poll_active(state) or {"verdict": "no_active_run"}, indent=2))
        return 0
    if args.cmd == "start-rfp":
        cand = asdict(Candidate("rfp-ace", "titanium-v17-rfp-ace", note="ACE RFP depth<=3 margin"))
        state.active_run = None
        state.queue = [cand] + [q for q in state.queue if q.get("candidate_id") != "rfp-ace"]
        active = start_candidate_match(state, cand, games=args.games)
        print(json.dumps(active, indent=2))
        return 0
    if args.cmd == "start-next":
        if not state.queue:
            print("queue empty", file=sys.stderr)
            return 1
        build_native()
        active = start_candidate_match(state, state.queue[0], games=args.games)
        print(json.dumps(active, indent=2))
        return 0
    if args.cmd == "tick":
        build_native()
        print(json.dumps(tick(state), indent=2))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
