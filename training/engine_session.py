"""Warm engine session: one persistent titanium process per game side.

Self-play used to spawn a brand-new `titanium genmove` process for every
single ply -- cold transposition table, cold dist-topology LRU, cold NNUE
weight load, every move. `titanium session --engine ...` is a long-lived REPL
that keeps one warm TitaniumSearch (TT, killers, history, countermove tables,
dist LRU) for the life of the process. This wrapper drives that REPL so a
whole game reuses one process per side instead of one process per ply.

Wire protocol (see engine/src/titanium/session.rs):
  position MOVES...   -> "ready N"      (cheap incremental apply if it extends)
  go TIME_SEC          -> "info json {...}" then "bestmove MOVE"
  quit                 -> process exits

Weights are loaded once at process start (env TITANIUM_NET_WEIGHTS_PATH) and
never touched again for the life of the session -- consistent with freezing
one side's weights for the whole game (see freeze_worker_game_weights in
streaming_checkpoint_chain.py).

Timing: `go TIME_SEC` is sent to an already-synced, already-warm process, so
the wall-clock time from sending "go" to reading "bestmove" is search time
only. Process spawn and weight load happen once at session start, outside
any per-move timing.
"""
from __future__ import annotations

import os
import queue
import subprocess
import json
import sys
import threading
from pathlib import Path
from typing import Optional

try:
    from titanium_training.paths import ENGINE_BIN, REPO_ROOT
except ModuleNotFoundError:
    # Standalone match bundles only need this wrapper plus an explicit binary.
    # Keep them independent of the full training package.
    REPO_ROOT = Path(
        os.environ.get("TITANIUM_GAME_FACTORY_ROOT", Path(__file__).resolve().parent.parent)
    )
    ENGINE_BIN = Path(
        os.environ.get(
            "TITANIUM_ENGINE_BIN",
            REPO_ROOT / "engine" / "target" / "release" / "titanium",
        )
    )


def _apply_engine_process_priority(proc: subprocess.Popen) -> None:
    """Optional RealTime/High + fixed affinity from env (picked once by the harness).

    TITANIUM_PROCESS_PRIORITY=realtime|high
    TITANIUM_AFFINITY_MASK=0x... or decimal (Windows/Linux bit mask of logical CPUs)
    """
    prio = (os.environ.get("TITANIUM_PROCESS_PRIORITY") or "").strip().lower()
    mask_raw = (os.environ.get("TITANIUM_AFFINITY_MASK") or "").strip()
    if not prio and not mask_raw:
        return
    try:
        if sys.platform == "win32":
            import ctypes

            handle = int(proc._handle)  # type: ignore[attr-defined]
            kernel32 = ctypes.windll.kernel32
            if prio in ("realtime", "real_time", "rt"):
                # REALTIME_PRIORITY_CLASS
                kernel32.SetPriorityClass(handle, 0x00000100)
            elif prio == "high":
                kernel32.SetPriorityClass(handle, 0x00000080)
            if mask_raw:
                mask = int(mask_raw, 0)
                kernel32.SetProcessAffinityMask(handle, mask)
        else:
            if hasattr(os, "sched_setaffinity") and mask_raw:
                mask = int(mask_raw, 0)
                cpus = [i for i in range(mask.bit_length()) if mask & (1 << i)]
                if cpus:
                    os.sched_setaffinity(proc.pid, cpus)
            if prio in ("realtime", "real_time", "rt", "high"):
                try:
                    if hasattr(os, "setpriority") and hasattr(os, "PRIO_PROCESS"):
                        os.setpriority(os.PRIO_PROCESS, proc.pid, -20)
                except OSError:
                    pass
                try:
                    import ctypes
                    import ctypes.util

                    libname = ctypes.util.find_library("c")
                    if not libname:
                        return
                    libc = ctypes.CDLL(libname, use_errno=True)

                    class SchedParam(ctypes.Structure):
                        _fields_ = [("sched_priority", ctypes.c_int)]

                    max_prio = 50
                    if hasattr(os, "sched_get_priority_max"):
                        max_prio = int(os.sched_get_priority_max(1))
                    param = SchedParam(max_prio)
                    libc.sched_setscheduler(int(proc.pid), 1, ctypes.byref(param))
                except Exception:
                    pass
    except Exception:
        # Never fail session start because of priority/affinity.
        pass


class EngineSession:
    def __init__(
        self,
        engine: str,
        weights: Path | None,
        threads: int = 1,
        engine_bin: Path | None = None,
    ):
        env = os.environ.copy()
        env["TITANIUM_BOOK_MODE"] = "off"
        if weights is not None and Path(weights).is_file():
            env["TITANIUM_NET_WEIGHTS_PATH"] = str(Path(weights).resolve())
        else:
            env.pop("TITANIUM_NET_WEIGHTS_PATH", None)
        binary = Path(engine_bin) if engine_bin is not None else ENGINE_BIN
        cmd = [str(binary), "session", "--engine", engine]
        if threads > 1:
            cmd += ["--threads", str(threads)]
        self.engine = engine
        self._applied: list[str] = []
        self._proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        _apply_engine_process_priority(self._proc)
        self._q: "queue.Queue[str]" = queue.Queue()
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _read_loop(self) -> None:
        try:
            stdout = self._proc.stdout
            if stdout is None:
                return
            for line in stdout:
                self._q.put(line.rstrip("\n"))
        except Exception:
            pass

    def _readline(self, timeout: float) -> Optional[str]:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def _send(self, cmd: str) -> bool:
        try:
            assert self._proc.stdin is not None
            self._proc.stdin.write(cmd + "\n")
            self._proc.stdin.flush()
            return True
        except Exception:
            return False

    def alive(self) -> bool:
        return self._proc.poll() is None

    def sync(self, moves: list[str], *, timeout: float = 30.0) -> bool:
        """Push the full current move list. Cheap incremental apply when it
        only extends the previously synced list -- the warm state (TT, dist
        LRU, killers) is preserved across the call either way."""
        if not self.alive():
            return False
        if not self._send("position " + " ".join(moves)):
            return False
        line = self._readline(timeout)
        if line and line.startswith("ready"):
            self._applied = list(moves)
            return True
        return False

    def go(self, time_sec: float, *, overhead_sec: float = 20.0) -> Optional[str]:
        """Search from the already-synced position. Returns the chosen move,
        or None on timeout/crash/no-move. Time budget is pure search time --
        the process is already warm and already at the right position."""
        if not self.alive():
            return None
        if not self._send(f"go {time_sec}"):
            return None
        deadline = max(time_sec + overhead_sec, 10.0)
        while True:
            line = self._readline(deadline)
            if line is None:
                return None
            if line.startswith("bestmove "):
                tok = line.split()[1]
                return None if tok == "(none)" else tok
            if line.startswith("error"):
                return None
            # "info json ..." or other diagnostics -- keep waiting for bestmove

    def go_detailed(self, time_sec: float, *, overhead_sec: float = 20.0) -> dict:
        """Return the final info JSON plus bestmove without changing go()."""
        result: dict = {"bestmove": None, "info": {}, "raw_info": []}
        if not self.alive() or not self._send(f"go {time_sec}"):
            return result
        deadline = max(time_sec + overhead_sec, 10.0)
        while True:
            line = self._readline(deadline)
            if line is None:
                return result
            if line.startswith("info json "):
                raw = line[len("info json "):]
                result["raw_info"].append(raw)
                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, dict):
                        result["info"] = parsed
                except json.JSONDecodeError:
                    pass
            elif line.startswith("bestmove "):
                tok = line.split()[1]
                result["bestmove"] = None if tok == "(none)" else tok
                return result
            elif line.startswith("error"):
                return result

    def close(self, *, timeout: float = 5.0) -> None:
        try:
            if self.alive():
                self._send("quit")
        except Exception:
            pass
        try:
            self._proc.wait(timeout=timeout)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass
