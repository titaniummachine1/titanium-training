#!/usr/bin/env python3
"""Continuously add Ka epoch-15000 ``ka_nn`` labels to the canonical DB.

The Node worker is persistent: weights are loaded once and every request uses
the verified native ONNX batch evaluator (or the WASM parity fallback).
Positions come from canonical game prefixes, so no second Quoridor rules
implementation is needed in Python.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import random
import sqlite3
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_TRAINING = _REPO / "training"
WORKER = Path(__file__).with_name("ka_nn_batch_worker.mjs")
GAMES_DB = _TRAINING / "data" / "canonical" / "games.db"
LABELS_DB = _TRAINING / "data" / "canonical" / "labels.db"
LOG_DIR = _TRAINING / "data" / "overnight_logs"
DEFAULT_OUT = _TRAINING / "data" / "ka_teacher_quarantine" / "ka_nn_labels.jsonl"
STATE_PATH = LOG_DIR / "ka_nn_labeling_state.json"
COORDINATOR_STATE_PATH = LOG_DIR / "training_coordinator_state.json"
SOURCE = "ka_nn"


def ensure_rejection_table(labels: sqlite3.Connection) -> None:
    labels.execute(
        """
        CREATE TABLE IF NOT EXISTS ka_nn_rejections (
            pos_key TEXT PRIMARY KEY,
            reason TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    labels.commit()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    line = f"[{utc_now()}] {message}"
    print(line, flush=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with (LOG_DIR / "ka_nn_labeling.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def save_state(payload: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(STATE_PATH)


def coordinator_is_training() -> bool:
    try:
        state = json.loads(COORDINATOR_STATE_PATH.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return False
    return state.get("state") == "TRAINING"


@dataclass(frozen=True)
class Candidate:
    game_id: str
    pos_key: str
    moves: list[str]
    phase: str


def _phase_range(max_ply: int, desired: str) -> tuple[int, int] | None:
    if desired == "opening":
        lo, hi = 0, min(8, max_ply)
    elif desired == "midgame":
        lo, hi = 9, min(24, max_ply)
    else:
        lo, hi = 25, max_ply
    return (lo, hi) if lo <= hi else None


def sample_candidates(
    games: sqlite3.Connection,
    labels: sqlite3.Connection,
    *,
    limit: int,
    seed: int,
) -> list[Candidate]:
    """Sample balanced unlabeled prefixes without rescanning one cut per game.

    The old sampler picked only one random ply from each game.  Because the
    canonical games and label stores overlap sparsely, a 4k request produced
    only ~350 candidates and starved a parallel inference pool.  Joining the
    indexed stores first makes every returned row useful; randomized rowid
    windows keep permanently-invalid imported prefixes from blocking progress.
    """
    rng = random.Random(seed)
    ensure_rejection_table(labels)
    attached = {str(row[1]) for row in games.execute("PRAGMA database_list")}
    if "label_store" not in attached:
        label_path = next(
            str(row[2]) for row in labels.execute("PRAGMA database_list") if row[1] == "main"
        )
        games.execute("ATTACH DATABASE ? AS label_store", (label_path,))
    max_rowid = int(games.execute("SELECT COALESCE(MAX(rowid), 0) FROM game_moves").fetchone()[0])
    if max_rowid <= 0:
        return []

    phases = (
        ("opening", "gm.move_num BETWEEN 0 AND 8"),
        ("midgame", "gm.move_num BETWEEN 9 AND 24"),
        ("endgame", "gm.move_num >= 25"),
    )
    out: list[Candidate] = []
    seen: set[str] = set()
    base_target, remainder = divmod(limit, len(phases))
    for phase_index, (phase, phase_sql) in enumerate(phases):
        target = base_target + (1 if phase_index < remainder else 0)
        start = rng.randint(1, max_rowid)
        fetch_limit = max(64, target * 3)
        query = f"""
            SELECT gm.rowid, gm.game_id, gm.move_num, gm.pos_key
            FROM game_moves gm
            JOIN label_store.positions p ON p.pos_key=gm.pos_key
            LEFT JOIN label_store.labels l
              ON l.pos_key=gm.pos_key AND l.source=?
            LEFT JOIN label_store.ka_nn_rejections r ON r.pos_key=gm.pos_key
            WHERE {{row_window}} AND {phase_sql}
              AND l.pos_key IS NULL AND r.pos_key IS NULL
            ORDER BY gm.rowid
            LIMIT ?
        """
        rows = games.execute(
            query.format(row_window="gm.rowid >= ?"),
            (SOURCE, start, fetch_limit),
        ).fetchall()
        if len(rows) < fetch_limit:
            rows.extend(
                games.execute(
                    query.format(row_window="gm.rowid < ?"),
                    (SOURCE, start, fetch_limit - len(rows)),
                ).fetchall()
            )
        rng.shuffle(rows)
        phase_count = 0
        for _rowid, game_id, cut_raw, pos_key_raw in rows:
            if phase_count >= target:
                break
            cut = int(cut_raw)
            pos_key = str(pos_key_raw)
            if pos_key in seen:
                continue
            move_rows = games.execute(
                "SELECT move_alg FROM game_moves WHERE game_id=? AND move_num < ? ORDER BY move_num",
                (game_id, cut),
            ).fetchall()
            moves = [str(row[0]) for row in move_rows]
            if len(moves) != cut:
                continue
            seen.add(pos_key)
            out.append(Candidate(str(game_id), pos_key, moves, phase))
            phase_count += 1
    # Canonical openings transpose heavily, so there may not be enough unique
    # unlabeled opening keys for an exact third. Keep the pool full with later
    # positions instead of returning an undersized batch.
    remaining = limit - len(out)
    if remaining > 0:
        start = rng.randint(1, max_rowid)
        rows = games.execute(
            """
            SELECT gm.rowid, gm.game_id, gm.move_num, gm.pos_key
            FROM game_moves gm
            JOIN label_store.positions p ON p.pos_key=gm.pos_key
            LEFT JOIN label_store.labels l
              ON l.pos_key=gm.pos_key AND l.source=?
            LEFT JOIN label_store.ka_nn_rejections r ON r.pos_key=gm.pos_key
            WHERE gm.rowid >= ? AND gm.move_num >= 9
              AND l.pos_key IS NULL AND r.pos_key IS NULL
            ORDER BY gm.rowid
            LIMIT ?
            """,
            (SOURCE, start, remaining * 4),
        ).fetchall()
        rng.shuffle(rows)
        for _rowid, game_id, cut_raw, pos_key_raw in rows:
            if len(out) >= limit:
                break
            cut = int(cut_raw)
            pos_key = str(pos_key_raw)
            if pos_key in seen:
                continue
            move_rows = games.execute(
                "SELECT move_alg FROM game_moves WHERE game_id=? AND move_num < ? ORDER BY move_num",
                (game_id, cut),
            ).fetchall()
            moves = [str(row[0]) for row in move_rows]
            if len(moves) != cut:
                continue
            seen.add(pos_key)
            phase = "midgame" if cut <= 24 else "endgame"
            out.append(Candidate(str(game_id), pos_key, moves, phase))
    rng.shuffle(out)
    return out


class KaWorker:
    def __init__(
        self,
        *,
        backend: str,
        batch_max: int,
        model_batch: int,
        device_id: int,
        threads: int = 1,
        command_prefix: list[str] | None = None,
        name: str = "local",
    ) -> None:
        cmd = [
            *(command_prefix or ["node", str(WORKER)]),
            "--backend",
            backend,
            "--batch-max",
            str(batch_max),
            "--model-batch",
            str(model_batch),
            "--device-id",
            str(device_id),
            "--threads",
            str(threads),
        ]
        self.proc = subprocess.Popen(
            cmd,
            cwd=str(_REPO),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        assert self.proc.stderr is not None
        ready_lines: list[str] = []
        for _ in range(20):
            line = self.proc.stderr.readline().strip()
            if line:
                ready_lines.append(line)
            if "ready backend=" in line or self.proc.poll() is not None:
                break
        ready = next((line for line in ready_lines if "ready backend=" in line), "")
        if not ready:
            self.close()
            raise RuntimeError(f"Ka worker {name} failed to initialize: {' | '.join(ready_lines)}")
        self.ready = ready
        self.name = name
        self.request_id = 0

    def evaluate(self, candidates: list[Candidate]) -> dict:
        if not candidates:
            return {"ok": True, "rows": []}
        self.request_id += 1
        request = {
            "id": self.request_id,
            "positions": [
                {"id": candidate.pos_key, "moves": candidate.moves}
                for candidate in candidates
            ],
        }
        assert self.proc.stdin is not None and self.proc.stdout is not None
        self.proc.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            stderr = self.proc.stderr.read() if self.proc.stderr else ""
            raise RuntimeError(f"Ka worker exited unexpectedly: {stderr}")
        response = json.loads(line)
        if not response.get("ok"):
            raise RuntimeError(f"Ka worker request failed: {response.get('error')}")
        if response.get("id") != self.request_id:
            raise RuntimeError("Ka worker response id mismatch")
        return response

    def close(self) -> None:
        if self.proc.stdin is not None:
            self.proc.stdin.close()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.terminate()
            self.proc.wait(timeout=10)

    def __enter__(self) -> "KaWorker":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class KaWorkerPool:
    """Keep independent inference lanes busy from one shared chunk queue."""

    def __init__(self, workers: list[KaWorker], *, chunk_size: int) -> None:
        if not workers:
            raise ValueError("worker pool must not be empty")
        self.workers = workers
        self.chunk_size = max(1, chunk_size)

    def evaluate(self, candidates: list[Candidate]) -> dict:
        chunks = [
            candidates[offset : offset + self.chunk_size]
            for offset in range(0, len(candidates), self.chunk_size)
        ]
        next_chunk = 0
        lock = threading.Lock()

        def run_lane(worker: KaWorker) -> list[dict]:
            nonlocal next_chunk
            lane_results: list[dict] = []
            while True:
                with lock:
                    if next_chunk >= len(chunks):
                        return lane_results
                    chunk = chunks[next_chunk]
                    next_chunk += 1
                lane_results.append(worker.evaluate(chunk))

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.workers)) as executor:
            futures = [executor.submit(run_lane, worker) for worker in self.workers]
            responses = [response for future in futures for response in future.result()]

        rows = [row for response in responses for row in response.get("rows", [])]
        rejected = [row for response in responses for row in response.get("rejected", [])]
        timing_keys = ("prepare", "inference", "summarize")
        timings = {
            key: sum(float(response.get("timings_ms", {}).get(key, 0.0)) for response in responses)
            for key in timing_keys
        }
        first = responses[0] if responses else {}
        return {
            "ok": True,
            "rows": rows,
            "rejected": rejected,
            "backend": "+".join(worker.name for worker in self.workers),
            "checkpoint": first.get("checkpoint"),
            "weights_sha256": first.get("weights_sha256"),
            "timings_ms": timings,
            "chunks": len(chunks),
        }

    def close(self) -> None:
        for worker in self.workers:
            worker.close()

    def __enter__(self) -> "KaWorkerPool":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def store_labels(
    labels: sqlite3.Connection,
    candidates: list[Candidate],
    response: dict,
    out_path: Path,
) -> int:
    by_key = {candidate.pos_key: candidate for candidate in candidates}
    accepted: list[tuple[str, float, dict, Candidate]] = []
    for row in response.get("rows", []):
        pos_key = str(row.get("id", ""))
        candidate = by_key.get(pos_key)
        value_stm = float(row.get("value_stm", float("nan")))
        if candidate is None or not math.isfinite(value_stm) or not -1.0 <= value_stm <= 1.0:
            continue
        accepted.append((pos_key, value_stm, row, candidate))
    rejected = [
        (str(row.get("id", "")), str(row.get("error", "unknown replay error")))
        for row in response.get("rejected", [])
        if str(row.get("id", "")) in by_key
    ]
    if not accepted and not rejected:
        return 0

    labels.execute("BEGIN IMMEDIATE")
    try:
        now = utc_now()
        labels.executemany(
            """
            INSERT INTO ka_nn_rejections(pos_key, reason, created_at)
            VALUES (?, ?, ?)
            ON CONFLICT(pos_key) DO NOTHING
            """,
            [(pos_key, reason, now) for pos_key, reason in rejected],
        )
        written = 0
        written_keys: list[str] = []
        for pos_key, value_stm, _row, _candidate in accepted:
            if labels.execute("SELECT 1 FROM positions WHERE pos_key=?", (pos_key,)).fetchone() is None:
                continue
            cur = labels.execute(
                """
                INSERT INTO labels(pos_key, source, value_stm, n_samples)
                VALUES (?, ?, ?, 1)
                ON CONFLICT(pos_key, source) DO NOTHING
                """,
                (pos_key, SOURCE, value_stm),
            )
            if int(cur.rowcount) > 0:
                written += 1
                written_keys.append(pos_key)
        if written_keys:
            from position_usage_db import increment_new_eligible, upsert_positions

            new_eligible = upsert_positions(labels, written_keys)
            if new_eligible:
                increment_new_eligible(labels, new_eligible)
        labels.commit()
    except Exception:
        labels.rollback()
        raise

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as handle:
        for pos_key, value_stm, row, candidate in accepted:
            audit = {
                "schema": "ka-nn-label-v1",
                "created_at": utc_now(),
                "source": SOURCE,
                "game_id": candidate.game_id,
                "pos_key": pos_key,
                "moves": candidate.moves,
                "phase": candidate.phase,
                "value_stm": value_stm,
                "teacher": row,
                "backend": response.get("backend"),
                "checkpoint": response.get("checkpoint"),
                "weights_sha256": response.get("weights_sha256"),
            }
            handle.write(json.dumps(audit, separators=(",", ":")) + "\n")
    return written


def main() -> int:
    from prep_guard import guard_real_work

    guard_real_work("labeling", detail="ka_nn_collect_labels.py")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=512, help="positions per batch")
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--seed", type=int, default=15000)
    parser.add_argument(
        "--backend",
        choices=("directml", "cpu", "wasm", "js", "auto"),
        default="directml",
    )
    parser.add_argument("--batch-max", type=int, default=64)
    parser.add_argument("--model-batch", type=int, default=64)
    parser.add_argument("--device-id", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--local-cpu-workers", type=int, default=0)
    parser.add_argument("--oracle-workers", type=int, default=0)
    parser.add_argument("--oracle-host", default="92.5.77.92")
    parser.add_argument("--oracle-user", default="ubuntu")
    parser.add_argument(
        "--oracle-key",
        type=Path,
        default=Path.home() / ".ssh" / "oracle_titanium.key",
    )
    parser.add_argument("--oracle-root", default="/home/ubuntu/ka-teacher")
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument(
        "--pause-during-training",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="leave teacher lanes warm but idle while the local trainer/validation cycle runs",
    )
    parser.add_argument("--sleep-sec", type=float, default=1.0)
    parser.add_argument("--games-db", type=Path, default=GAMES_DB)
    parser.add_argument("--labels-db", type=Path, default=LABELS_DB)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")
    if not args.games_db.is_file() or not args.labels_db.is_file():
        parser.error("canonical games.db and labels.db must exist")

    games = sqlite3.connect(args.games_db, timeout=120)
    labels = sqlite3.connect(args.labels_db, timeout=120)
    labels.execute("PRAGMA journal_mode=WAL")
    ensure_rejection_table(labels)
    total = 0
    batch_index = 0
    try:
        workers = [
            KaWorker(
                backend=args.backend,
                batch_max=args.batch_max,
                model_batch=args.model_batch,
                device_id=args.device_id,
                threads=args.threads,
                name=f"{args.backend}-local",
            )
        ]
        workers.extend(
            KaWorker(
                backend="cpu",
                batch_max=args.batch_max,
                model_batch=args.model_batch,
                device_id=0,
                threads=1,
                name=f"cpu-local-{index}",
            )
            for index in range(args.local_cpu_workers)
        )
        remote_worker = f"{args.oracle_root}/training/tools/ka_teacher/ka_nn_batch_worker.mjs"
        workers.extend(
            KaWorker(
                backend="cpu",
                batch_max=args.batch_max,
                model_batch=args.model_batch,
                device_id=0,
                threads=1,
                command_prefix=[
                    "ssh",
                    "-i",
                    str(args.oracle_key),
                    "-o",
                    "BatchMode=yes",
                    f"{args.oracle_user}@{args.oracle_host}",
                    "nice",
                    "-n",
                    "5",
                    "node",
                    remote_worker,
                ],
                name=f"cpu-oracle-{index}",
            )
            for index in range(args.oracle_workers)
        )
        with KaWorkerPool(workers, chunk_size=args.chunk_size) as worker:
            for lane in workers:
                log(f"lane {lane.name}: {lane.ready}")
            pause_logged = False
            while True:
                if args.pause_during_training and coordinator_is_training():
                    if not pause_logged:
                        log("coordinator is TRAINING; teacher pool paused to avoid local contention")
                        pause_logged = True
                    time.sleep(15.0)
                    continue
                if pause_logged:
                    log("coordinator left TRAINING; teacher pool resumed")
                    pause_logged = False
                batch_index += 1
                candidates = sample_candidates(
                    games,
                    labels,
                    limit=args.limit,
                    seed=args.seed + batch_index,
                )
                if not candidates:
                    log("no unlabeled canonical candidates found")
                    break
                started = time.perf_counter()
                response = worker.evaluate(candidates)
                elapsed = time.perf_counter() - started
                written = store_labels(labels, candidates, response, args.out)
                total += written
                phase_counts = {
                    phase: sum(candidate.phase == phase for candidate in candidates)
                    for phase in ("opening", "midgame", "endgame")
                }
                state = {
                    "updated_at": utc_now(),
                    "batch_index": batch_index,
                    "last_sampled": len(candidates),
                    "last_written": written,
                    "total_written": total,
                    "elapsed_sec": elapsed,
                    "positions_per_sec": len(candidates) / max(elapsed, 1e-9),
                    "phase_counts": phase_counts,
                    "rejected_prefixes": len(response.get("rejected", [])),
                    "backend": response.get("backend"),
                    "checkpoint": response.get("checkpoint"),
                    "weights_sha256": response.get("weights_sha256"),
                    "worker_count": len(workers),
                    "chunks": response.get("chunks", 1),
                    "stage_cpu_ms": response.get("timings_ms", {}),
                    "out": str(args.out),
                }
                save_state(state)
                log(
                    f"batch {batch_index}: sampled={len(candidates)} written={written} "
                    f"rejected={state['rejected_prefixes']} "
                    f"rate={state['positions_per_sec']:.1f}/s phases={phase_counts} total={total}"
                )
                if not args.continuous:
                    break
                time.sleep(max(0.0, args.sleep_sec))
    finally:
        games.close()
        labels.close()
    return 0 if total > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
