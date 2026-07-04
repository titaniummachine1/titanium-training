"""Oracle game-factory supervisor."""
from __future__ import annotations

import json
import os
import queue
import resource
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .generation import GenerationStore
from .matchup import choose_matchup
from .protocol import read_json, utc_now
from .spool import DurableSpool, SpoolConfig
from .worker import RuntimeConfig, play_matchup_game, preflight_engine


@dataclass
class SupervisorConfig:
    data_dir: Path = Path("/var/lib/titanium-game-factory")
    engine_bin: Path = Path("/opt/titanium-game-factory/engine/target/release/titanium")
    workers: int = 13
    move_time: float = 5.0
    node_budget: int = 200_000
    generation_seed: int = 0
    max_worker_failures: int = 20


class GameSupervisor:
    def __init__(self, cfg: SupervisorConfig):
        self.cfg = cfg
        self.spool = DurableSpool(SpoolConfig(cfg.data_dir / "spool"))
        self.generations = GenerationStore(cfg.data_dir / "generations")
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._pause.clear()
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()
        self._failures = 0
        self._completed = 0
        self._started = time.time()
        self._schedule_index = 0
        self._matchup_counts: dict[str, int] = {"generation_selfplay": 0, "generation_mixed": 0}
        self._status_path = cfg.data_dir / "supervisor_status.json"

    def nproc(self) -> int:
        try:
            return int(subprocess.check_output(["nproc"], text=True).strip())
        except Exception:
            return os.cpu_count() or 1

    def status(self) -> dict[str, Any]:
        bp = self.spool.backpressure()
        elapsed = max(time.time() - self._started, 1.0)
        with self._lock:
            completed = self._completed
            failures = self._failures
            schedule_index = self._schedule_index
            counts = dict(self._matchup_counts)
        return {
            "ok": not self._stop.is_set(),
            "paused": self._pause.is_set(),
            "workers_configured": self.cfg.workers,
            "workers_expected": 13,
            "nproc": self.nproc(),
            "completed": completed,
            "games_per_hour": completed * 3600.0 / elapsed,
            "worker_failures": failures,
            "schedule_index": schedule_index,
            "matchup_counts": counts,
            "schedule_counts": counts,
            "spool": bp,
            "search_config": {
                "move_time_sec": self.cfg.move_time,
                "node_budget": self.cfg.node_budget if self.cfg.node_budget > 0 else None,
            },
            "active_generation": self.generations.active_manifest(),
            "updated_at": utc_now(),
        }

    def _write_status(self) -> None:
        self._status_path.parent.mkdir(parents=True, exist_ok=True)
        self._status_path.write_text(json.dumps(self.status(), indent=2), encoding="utf-8")

    def _next_job(self) -> dict[str, Any] | None:
        # Caller (`_worker_loop`) always holds `self._lock` around this call —
        # re-acquiring it here deadlocked every worker thread the instant
        # `active_manifest()` returned truthy (threading.Lock is not
        # reentrant). `_schedule_index` is already protected by the caller's
        # lock; do not lock again.
        manifest = self.generations.active_manifest()
        if not manifest:
            return None
        idx = self._schedule_index
        self._schedule_index += 1
        current = str(manifest["current_deployed_hash"]).lower()
        prior = manifest.get("prior_deployed_hash")
        prior_distinct = bool(manifest.get("prior_is_distinct")) and prior
        prior_s = str(prior).lower() if prior_distinct else None
        game_id = (
            f"oracle-{manifest['generation_id']}-w{idx:06d}-"
            f"{int(time.time())}-{idx}"
        )
        matchup = choose_matchup(game_id, current, prior_s)
        return {"game_id": game_id, "game_index": idx, "matchup": matchup}

    def _worker_loop(self, worker_id: int) -> None:
        backoff = 1.0
        runtime = RuntimeConfig(
            engine_bin=self.cfg.engine_bin,
            data_dir=self.cfg.data_dir,
            move_time=self.cfg.move_time,
            node_budget=self.cfg.node_budget,
        )
        while not self._stop.is_set():
            if self._pause.is_set() or self.spool.backpressure()["stop"]:
                time.sleep(1.0)
                continue
            with self._lock:
                job = self._next_job()
                active_dir = self.generations.active_dir()
                active_manifest = self.generations.active_manifest()
            if not job or not active_dir or not active_manifest:
                time.sleep(2.0)
                continue
            generation = {"path": str(active_dir), "manifest": active_manifest}
            try:
                payload = play_matchup_game(
                    cfg=runtime,
                    generation=generation,
                    matchup=job["matchup"],
                    game_id=job["game_id"],
                    worker_id=worker_id,
                    game_index=job["game_index"],
                )
                self.spool.write_game(payload)
                with self._lock:
                    self._completed += 1
                    kind = job["matchup"].kind
                    self._matchup_counts[kind] = self._matchup_counts.get(kind, 0) + 1
                backoff = 1.0
            except Exception as exc:
                with self._lock:
                    self._failures += 1
                    failures = self._failures
                if failures >= self.cfg.max_worker_failures:
                    self._stop.set()
                    break
                time.sleep(backoff)
                backoff = min(backoff * 2.0, 60.0)
            finally:
                self._write_status()

    def start(self) -> None:
        if self.nproc() < max(1, self.cfg.workers // 2):
            raise RuntimeError(f"nproc={self.nproc()} unexpectedly low for workers={self.cfg.workers}")
        active = self.generations.active_manifest()
        if active:
            preflight_engine(
                RuntimeConfig(self.cfg.engine_bin, self.cfg.data_dir, self.cfg.move_time, node_budget=self.cfg.node_budget),
                {"path": str(self.generations.active_dir()), "manifest": active},
            )
        for worker_id in range(self.cfg.workers):
            t = threading.Thread(target=self._worker_loop, args=(worker_id,), daemon=True, name=f"oracle-worker-{worker_id}")
            t.start()
            self._threads.append(t)

    def stop(self, *, drain: bool = False, timeout: float = 120.0) -> None:
        self._pause.set()
        if not drain:
            self._stop.set()
        deadline = time.time() + timeout
        while time.time() < deadline:
            alive = [t for t in self._threads if t.is_alive()]
            if not alive:
                return
            if not drain:
                time.sleep(0.5)
            else:
                self._stop.set()
                time.sleep(0.5)

    def pause(self) -> None:
        self._pause.set()

    def resume(self) -> None:
        self._pause.clear()

