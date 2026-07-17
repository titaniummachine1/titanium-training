#!/usr/bin/env python3
"""Bounded, supervised Oracle Horizon Cycle 1 runner.

This executable never starts the coordinator, producer, unattended loop, or
promotion machinery.  Training is intentionally skipped when a safe 10%
curriculum injection cannot be proven.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TRAINING = ROOT / "training"
if str(TRAINING) not in sys.path:
    sys.path.insert(0, str(TRAINING))

from titanium_training.store.state import replay_game
from oracle_horizon.bands import assign_band
from oracle_horizon.cycle1_audit import audit_file, classify_resolution, oracle_resolved
from oracle_horizon.horizon_rows import build_horizon_row, initial_target_move, ply_move
from oracle_horizon.needs_learning import needs_learning
from engine_session import EngineSession

SEED = 2_026_0717
DEPLOYMENT_NODES = 50_000
LADDER_NODES = (50_000, 200_000, 800_000, 3_200_000)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_openings(path: Path) -> list[list[str]]:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        rows = []
        for group in doc.get("nodesByPly", {}).values():
            if isinstance(group, list):
                rows.extend(
                    list(row["prefix"]) for row in group
                    if isinstance(row, dict) and row.get("prefix")
                )
        if rows:
            return sorted({tuple(map(str, row)) for row in rows}, key=lambda row: (len(row), row))
    except (OSError, TypeError, ValueError, KeyError):
        pass
    return [["e2", "e8", "e3", "e7", "e4", "e6"]]


def _winner(moves: list[str]) -> int | None:
    if not moves or moves[-1].endswith(("h", "v")):
        return None
    last = moves[-1][-1]
    mover = (len(moves) - 1) % 2
    if mover == 0 and last == "9":
        return 0
    if mover == 1 and last == "1":
        return 1
    return None


def generate_game(
    game_id: str,
    opening: list[str],
    *,
    engine: Path,
    weights: Path,
    time_sec: float,
    weights_sha256: str,
    engine_sha256: str,
) -> dict:
    moves = list(opening)
    sessions = (
        EngineSession("titanium-v17", weights, engine_bin=engine),
        EngineSession("titanium-v17", weights, engine_bin=engine),
    )
    try:
        for _ in range(128 - len(moves)):
            side = sessions[len(moves) % 2]
            if not side.sync(moves):
                raise RuntimeError("generation protocol error: position sync failed")
            move = side.go(time_sec)
            if not move:
                raise RuntimeError("generation protocol error: no bestmove")
            moves.append(move)
            if _winner(moves) is not None:
                break
    finally:
        for session in sessions:
            session.close()
    winner = _winner(moves)
    return {
        "game_id": game_id,
        "lineage_id": f"cycle1-{game_id}",
        "moves": moves,
        "winner": winner,
        "plies": len(moves),
        "opening": opening,
        "weights_sha256": weights_sha256,
        "engine_sha256": engine_sha256,
        "book_move_used": False,
        "protocol_error": False,
    }


def score_out(
    packed: bytes,
    *,
    engine: Path,
    weights: Path,
    nodes: int,
    budget: dict,
    deadline: float | None = None,
) -> dict:
    env = os.environ.copy()
    env["TITANIUM_BOOK_MODE"] = "off"
    env["TITANIUM_NET_WEIGHTS_PATH"] = str(weights.resolve())
    started = time.monotonic()
    timeout = max(30.0, min(600.0, nodes / 5000.0 + 30.0))
    if deadline is not None:
        remaining = deadline - started
        if remaining <= 0:
            raise TimeoutError("score-out wall-time budget exhausted")
        timeout = min(timeout, remaining)
    import subprocess
    proc = subprocess.run(
        [str(engine), "score-out", "--nodes", str(nodes), "--packed", packed.hex()],
        cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=timeout,
    )
    budget["score_out_calls"] = budget.get("score_out_calls", 0) + 1
    budget["score_out_seconds"] = budget.get("score_out_seconds", 0.0) + (time.monotonic() - started)
    if proc.returncode != 0:
        raise RuntimeError(f"score-out failed: {proc.stderr[-500:]}")
    frames = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
    if len(frames) != 1 or not isinstance(frames[0], dict):
        raise RuntimeError("score-out protocol error")
    return frames[0]


def mine_game(
    game: dict,
    *,
    engine: Path,
    weights: Path,
    weights_sha256: str,
    engine_sha256: str,
    budget: dict,
    max_positions: int,
    deadline: float,
) -> list[dict]:
    states = replay_game(game["moves"])
    scan_start = max(0, len(states) - max(16, 10))
    oracle: dict | None = None
    oracle_index: int | None = None
    for index in range(len(states) - 1, scan_start - 1, -1):
        if time.monotonic() >= deadline or budget["candidate_positions"] >= max_positions:
            break
        result = score_out(states[index].packed_state(), engine=engine, weights=weights,
                           nodes=DEPLOYMENT_NODES, budget=budget, deadline=deadline)
        if oracle_resolved(result.get("score"), bool(result.get("proven"))):
            oracle_index, oracle = index, result
            break
    if oracle is None:
        # A late terminal position occasionally needs a deeper proof, but do
        # not turn this fallback into a whole-game score-out sweep.
        for index in range(len(states) - 1, max(-1, len(states) - 9), -1):
            if time.monotonic() >= deadline or budget["candidate_positions"] >= max_positions:
                break
            result = score_out(states[index].packed_state(), engine=engine, weights=weights,
                               nodes=200_000, budget=budget, deadline=deadline)
            if oracle_resolved(result.get("score"), bool(result.get("proven"))):
                oracle_index, oracle = index, result
                break
    if oracle is None or oracle_index is None:
        budget["no_oracle_entry"] = budget.get("no_oracle_entry", 0) + 1
        budget["last_oracle_index"] = None
        return []
    budget["last_oracle_index"] = oracle_index
    oracle_wdl = "W" if int(oracle.get("score", 0)) > 0 else "L" if int(oracle.get("score", 0)) < 0 else "D"
    rows: list[dict] = []
    for index in range(oracle_index, max(-1, oracle_index - 9), -1):
        if time.monotonic() >= deadline or budget["candidate_positions"] >= max_positions:
            budget["budget_paused"] = True
            break
        deploy = score_out(states[index].packed_state(), engine=engine, weights=weights,
                           nodes=DEPLOYMENT_NODES, budget=budget, deadline=deadline)
        budget["candidate_positions"] += 1
        deploy_wdl = "W" if int(deploy.get("score", 0)) > 0 else "L" if int(deploy.get("score", 0)) < 0 else "D"
        ref = {"wdl": deploy_wdl, "best_move": deploy.get("selected_move"),
               "nodes_ratio": 1, "eval": deploy.get("score", 0), "move_flip": False}
        target = {"wdl": oracle_wdl, "best_move": initial_target_move(
                   index=index, oracle_index=oracle_index,
                   oracle_move=oracle.get("selected_move")),
                  "decisive": oracle_wdl != "D", "only_defense": False}
        needed, reasons = needs_learning(ref, target)
        backed_proven = False
        proven_move: str | None = None
        ladder: list[dict] = []
        if needed:
            for nodes in LADDER_NODES[1:]:
                if time.monotonic() >= deadline:
                    budget["budget_paused"] = True
                    break
                proof = score_out(states[index].packed_state(), engine=engine, weights=weights,
                                  nodes=nodes, budget=budget, deadline=deadline)
                ladder.append({"nodes": nodes, **proof})
                proof_wdl = "W" if int(proof.get("score", 0)) > 0 else "L" if int(proof.get("score", 0)) < 0 else "D"
                if proof.get("proven") and proof_wdl == oracle_wdl:
                    backed_proven = True
                    proven_move = proof.get("selected_move")
                    break
        best_move = ply_move(
            index=index,
            oracle_index=oracle_index,
            oracle_move=oracle.get("selected_move"),
            deploy_move=deploy.get("selected_move"),
            proven_move=proven_move,
            exact=not needed,
        )
        if backed_proven:
            target = {**target, "best_move": proven_move}
            needed, reasons = needs_learning(ref, target)
        elif not needed:
            target = {**target, "best_move": deploy.get("selected_move")}
        klass = "EXACT_ORACLE" if not needed and oracle.get("proven") else (
            "ORACLE_BACKED_MINIMAX" if backed_proven else "SEARCH_ONLY"
        )
        rows.append(build_horizon_row(
            position_id=f"{game['game_id']}:{index}",
            packed_state_hex=states[index].packed_state().hex(),
            game_id=game["game_id"],
            lineage_id=game["lineage_id"],
            index=index,
            oracle_index=oracle_index,
            band=assign_band(oracle_index - index),
            label_class=klass,
            primary=klass in {"EXACT_ORACLE", "ORACLE_BACKED_MINIMAX"},
            oracle_wdl=oracle_wdl,
            oracle_proven=bool(oracle.get("proven")) or backed_proven,
            backed_proven=backed_proven,
            deploy=deploy,
            best_move=best_move,
            needs_learning_value=needed,
            needs_learning_reasons=reasons,
            ladder=ladder,
            weights_sha256=weights_sha256,
            engine_sha256=engine_sha256,
        ))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def run(args: argparse.Namespace) -> dict:
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    weights_sha = sha256_file(args.weights)
    engine_sha = sha256_file(args.engine)
    openings = load_openings(args.opening_book)
    games: list[dict] = []
    errors = 0
    game_tasks = []
    for index in range(args.games):
        rng = random.Random(SEED + index)
        opening = list(openings[rng.randrange(len(openings))])
        game_tasks.append((index, opening))
    with (out / "games.jsonl").open("w", encoding="utf-8") as game_file:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {
                pool.submit(
                    generate_game, f"cycle1-{index:04d}", opening,
                    engine=args.engine, weights=args.weights, time_sec=args.time_sec,
                    weights_sha256=weights_sha, engine_sha256=engine_sha,
                ): index
                for index, opening in game_tasks
            }
            completed = 0
            for future in as_completed(futures):
                try:
                    game = future.result()
                except (OSError, RuntimeError, subprocess.SubprocessError):
                    errors += 1
                    for pending in futures:
                        pending.cancel()
                    break
                games.append(game)
                completed += 1
                game_file.write(json.dumps(game, sort_keys=True) + "\n")
                game_file.flush()
                print(f"GENERATED {completed}/{args.games} plies={game['plies']}", flush=True)
    games.sort(key=lambda game: game["game_id"])
    if errors:
        raise RuntimeError("protocol_errors must be zero; generation stopped")
    budget = {"candidate_positions": 0, "score_out_calls": 0, "score_out_seconds": 0.0,
              "no_oracle_entry": 0, "budget_paused": False}
    deadline = time.monotonic() + args.max_cpu_hours * 3600.0
    candidates: list[dict] = []
    for game_index, game in enumerate(games, start=1):
        if time.monotonic() >= deadline or budget["candidate_positions"] >= args.max_positions:
            budget["budget_paused"] = True
            break
        before = budget["score_out_seconds"]
        budget["last_oracle_index"] = None
        try:
            candidates.extend(mine_game(
                game, engine=args.engine, weights=args.weights,
                weights_sha256=weights_sha, engine_sha256=engine_sha,
                budget=budget, max_positions=args.max_positions, deadline=deadline,
            ))
        except (TimeoutError, subprocess.TimeoutExpired):
            budget["budget_paused"] = True
        print(
            f"MINE game {game_index} oracle_ply={budget.get('last_oracle_index', 'none')} "
            f"candidates={sum(row['game_id'] == game['game_id'] for row in candidates)} "
            f"cpu_s={budget['score_out_seconds'] - before:.3f}",
            flush=True,
        )
    _write_jsonl(out / "labels_candidates.jsonl", candidates)
    primary = [row for row in candidates if row["primary"]]
    _write_jsonl(out / "labels_primary.jsonl", primary)
    _write_jsonl(out / "oracle_horizon_labels.jsonl", primary)
    (out / "mine_stats.json").write_text(json.dumps({
        **budget, "games_generated": len(games), "protocol_errors": errors,
        "candidate_count": len(candidates), "primary_count": len(primary),
    }, indent=2) + "\n", encoding="utf-8")
    audit = audit_file(out / "labels_primary.jsonl", parent_weights_sha256=weights_sha,
                       parent_engine_sha256=engine_sha, max_positions=args.max_positions)
    (out / "CYCLE1_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    report = {
        "cycle": "cycle1", "recommendation": "unattended_safe=false",
        "audit_status": audit["status"], "primary_count": len(primary),
        "games_generated": len(games), "protocol_errors": errors,
        "proof_horizon_mean": (
            sum(row["plies_to_oracle_entry"] for row in primary) / len(primary)
            if primary else None
        ),
        "missed_defense_count": sum("missed_only_defense" in row.get("needs_learning_reasons", []) for row in candidates),
        "training_status": "SKIPPED" if args.skip_train or len(primary) < 50 or audit["status"] != "PASS" else "INSUFFICIENT_PIPELINE",
        "training_reason": "safe 10% curriculum DB injection is not wired for Cycle 1",
    }
    (out / "CYCLE1_REPORT.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if report["training_status"] == "SKIPPED":
        (out / "TRAINING_SKIPPED.json").write_text(json.dumps({
            "reason": (
                "skip_train requested"
                if args.skip_train
                else "insufficient audited primary yield or audit failure"
            ),
            "curriculum_injection": "deferred_file_lane",
        }, indent=2) + "\n", encoding="utf-8")
    else:
        (out / "TRAINING_SKIPPED.json").write_text(json.dumps({
            "reason": "INSUFFICIENT_PIPELINE: curriculum_injection=deferred_file_lane"
        }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True), flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--max-positions", type=int, default=10_000)
    parser.add_argument("--max-cpu-hours", type=float, default=4.0)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--opening-book", type=Path, default=TRAINING / "data/opening_book/non_titanium_10ply.json")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--time-sec", type=float, default=0.9)
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()
    args.weights = args.weights.resolve()
    args.engine = args.engine.resolve()
    return 0 if run(args) else 1


if __name__ == "__main__":
    raise SystemExit(main())
