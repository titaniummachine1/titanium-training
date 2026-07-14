#!/usr/bin/env python3
"""Audit TRAINING_PREP_ONLY freeze — PASS / BLOCK / INVALID."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

_TRAINING = Path(__file__).resolve().parents[1]
_REPO = _TRAINING.parent
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

from db_import import GAMES_DB_PATH, LABELS_DB_PATH
from prep_guard import prep_only_enabled
from real_work_registry import REAL_WORK_ENTRY_POINTS, UNGUARDED_FORBIDDEN

LOG_DIR = _TRAINING / "data" / "overnight_logs"
REPORT_PATH = LOG_DIR / "training_freeze_audit.json"

WORKER_PATTERNS = (
    "training_coordinator.py",
    "local_game_pool.py",
    "continuous_pool.py",
    "oracle_importer.py",
    "self_play_overnight.py",
    "ka_nn_collect_labels.py",
    "oracle_game_factory",
)

MUTABLE_PATHS = (
    GAMES_DB_PATH,
    LABELS_DB_PATH,
    _TRAINING / "runs",
    LOG_DIR / "training_coordinator_state.json",
    LOG_DIR / "continuous_pool_state.json",
)


class AuditStatus(str, Enum):
    PASS = "PASS"
    BLOCK = "BLOCK"
    INVALID = "INVALID"


@dataclass(frozen=True)
class LiveProcess:
    pid: int
    command_line: str
    role: str
    prep_guard_at_start: bool | None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _detect_live_processes() -> list[LiveProcess]:
    if sys.platform != "win32":
        return []
    try:
        out = subprocess.check_output(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                "Select-Object ProcessId, CommandLine | ConvertTo-Json -Compress",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    if not out:
        return []
    try:
        rows = json.loads(out)
    except json.JSONDecodeError:
        return []
    if isinstance(rows, dict):
        rows = [rows]
    live: list[LiveProcess] = []
    for row in rows:
        cmd = str(row.get("CommandLine") or "")
        pid = int(row.get("ProcessId") or 0)
        if not pid or not cmd:
            continue
        if not any(p in cmd for p in WORKER_PATTERNS):
            continue
        role = "unknown"
        for pat in WORKER_PATTERNS:
            if pat in cmd:
                role = pat
                break
        # Processes started before guard wiring cannot be verified as frozen.
        prep_guard_at_start = None
        if "training_coordinator" in cmd or "local_game_pool" in cmd:
            prep_guard_at_start = False
        live.append(
            LiveProcess(
                pid=pid,
                command_line=cmd,
                role=role,
                prep_guard_at_start=prep_guard_at_start,
            )
        )
    return live


def _read_pid_file(name: str) -> int | None:
    path = LOG_DIR / name
    if not path.is_file():
        return None
    try:
        return int(path.read_text(encoding="ascii").strip())
    except ValueError:
        return None


def _path_snapshot() -> dict[str, float | None]:
    snap: dict[str, float | None] = {}
    for path in MUTABLE_PATHS:
        if path.is_file():
            snap[str(path)] = path.stat().st_mtime
        elif path.is_dir():
            snap[str(path)] = path.stat().st_mtime
        else:
            snap[str(path)] = None
    return snap


def audit_freeze() -> dict[str, Any]:
    blockers: list[str] = []
    invalid: list[str] = []

    if not prep_only_enabled():
        blockers.append("TRAINING_PREP_ONLY is not enabled in this shell")

    if UNGUARDED_FORBIDDEN:
        blockers.append(f"unguarded entry points: {[e.path for e in UNGUARDED_FORBIDDEN]}")

    live = _detect_live_processes()
    coordinator_pid = _read_pid_file("training_coordinator.pid")
    for proc in live:
        if proc.role == "training_coordinator.py" or proc.pid == coordinator_pid:
            blockers.append(
                f"live training_coordinator pid={proc.pid} may mutate labels.db, "
                f"games.db, checkpoints (started before freeze guard; not auto-stopped)"
            )
        elif proc.role in (
            "local_game_pool.py",
            "continuous_pool.py",
            "oracle_importer.py",
            "self_play_overnight.py",
        ):
            blockers.append(f"live worker pid={proc.pid} role={proc.role} may perform real work")

    try:
        snapshot = _path_snapshot()
    except OSError as exc:
        invalid.append(f"could not snapshot mutable paths: {exc}")
        snapshot = {}

    if invalid:
        status = AuditStatus.INVALID
    elif blockers:
        status = AuditStatus.BLOCK
    else:
        status = AuditStatus.PASS

    manual_stop = (
        f"Stop-Process -Id {live[0].pid} -Force"
        if live
        else "No live workers detected"
    )
    if coordinator_pid:
        manual_stop = (
            f"Stop-Process -Id {coordinator_pid} -Force  # training_coordinator.pid"
        )

    return {
        "status": status.value,
        "audited_at": _utc_now(),
        "prep_only_env": os.environ.get("TRAINING_PREP_ONLY", "<unset>"),
        "prep_only_enabled": prep_only_enabled(),
        "live_processes": [
            {
                "pid": p.pid,
                "role": p.role,
                "prep_guard_at_start": p.prep_guard_at_start,
                "command_line": p.command_line[:240],
            }
            for p in live
        ],
        "coordinator_pid_file": coordinator_pid,
        "pid_24892_note": (
            "PID 24892 was reported in prior session; re-run audit for current PIDs"
        ),
        "mutable_path_snapshot": snapshot,
        "guarded_entry_points": len([e for e in REAL_WORK_ENTRY_POINTS if e.guarded]),
        "blockers": blockers,
        "invalid_reasons": invalid,
        "manual_stop_command": manual_stop,
        "generic_stop_script": (
            "See training/tools/AUDIT_FREEZE_README.md for pid-file based stop commands"
        ),
    }


def main() -> int:
    report = audit_freeze()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    status = report["status"]
    if status == AuditStatus.PASS.value:
        return 0
    if status == AuditStatus.BLOCK.value:
        return 2
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
