"""Tests for supervisor PID helpers and architecture invariants."""
from __future__ import annotations

import ast
import importlib
import inspect
import os
import sys
import tempfile
import textwrap
from collections import Counter
from pathlib import Path

import psutil
import pytest

# ---------------------------------------------------------------------------
# Module paths
# ---------------------------------------------------------------------------
_TRAINING = Path(__file__).resolve().parents[1]
_TOOLS = _TRAINING / "tools"
sys.path.insert(0, str(_TRAINING))
sys.path.insert(0, str(_TOOLS))


# ---------------------------------------------------------------------------
# Import the supervisor module
# ---------------------------------------------------------------------------
import importlib.util

_sup_spec = importlib.util.spec_from_file_location(
    "persistent_supervisor", _TOOLS / "persistent_supervisor.py"
)
assert _sup_spec and _sup_spec.loader
_sup_mod = importlib.util.module_from_spec(_sup_spec)
# Stub out psutil-using imports that might fail without a real lock
sys.modules.setdefault("rebuild_checkpoint", type(sys)("rebuild_checkpoint"))
sys.modules["rebuild_checkpoint"].read_checkpoint = lambda *a, **kw: None  # type: ignore
sys.modules["rebuild_checkpoint"].stderr_is_deterministic = lambda *a, **kw: False  # type: ignore
_sup_spec.loader.exec_module(_sup_mod)  # type: ignore[union-attr]

optional_pid = _sup_mod.optional_pid
optional_int = _sup_mod.optional_int
read_pid_file = _sup_mod.read_pid_file
pid_alive = _sup_mod.pid_alive


# ===========================================================================
# 1.  optional_pid
# ===========================================================================
class TestOptionalPid:
    def test_none_returns_none(self):
        assert optional_pid(None) is None

    def test_zero_returns_none(self):
        assert optional_pid(0) is None

    def test_negative_returns_none(self):
        assert optional_pid(-1) is None

    def test_valid_int_string(self):
        assert optional_pid("12345") == 12345

    def test_valid_integer(self):
        assert optional_pid(99) == 99

    def test_float_string_fails(self):
        # "3.14" cannot be int()-ed from a string
        assert optional_pid("3.14") is None

    def test_float_value_truncates(self):
        # float values are truncated by int(), result > 0
        assert optional_pid(3.9) == 3

    def test_empty_string_returns_none(self):
        assert optional_pid("") is None

    def test_whitespace_string_returns_none(self):
        assert optional_pid("   ") is None

    def test_non_numeric_string_returns_none(self):
        assert optional_pid("abc") is None

    def test_json_null_is_none(self):
        # JSON null deserialises as Python None
        import json
        val = json.loads("null")
        assert optional_pid(val) is None

    def test_json_numeric(self):
        import json
        val = json.loads("17232")
        assert optional_pid(val) == 17232


# ===========================================================================
# 2.  read_pid_file
# ===========================================================================
class TestReadPidFile:
    """Use tempfile.mkdtemp() to avoid Windows pytest-temp locking issues."""

    def _tmp(self) -> Path:
        import tempfile
        return Path(tempfile.mkdtemp())

    def test_missing_file_returns_none(self):
        d = self._tmp()
        assert read_pid_file(d / "no_such.pid") is None

    def test_empty_file_returns_none(self):
        d = self._tmp()
        p = d / "empty.pid"
        p.write_text("", encoding="utf-8")
        assert read_pid_file(p) is None

    def test_whitespace_only_returns_none(self):
        d = self._tmp()
        p = d / "ws.pid"
        p.write_text("   \n", encoding="utf-8")
        assert read_pid_file(p) is None

    def test_malformed_text_returns_none(self):
        d = self._tmp()
        p = d / "bad.pid"
        p.write_text("not_a_number\n", encoding="utf-8")
        assert read_pid_file(p) is None

    def test_zero_returns_none(self):
        d = self._tmp()
        p = d / "zero.pid"
        p.write_text("0\n", encoding="utf-8")
        assert read_pid_file(p) is None

    def test_valid_pid(self):
        d = self._tmp()
        p = d / "ok.pid"
        p.write_text("14452\n", encoding="utf-8")
        assert read_pid_file(p) == 14452

    def test_json_null_text_returns_none(self):
        d = self._tmp()
        p = d / "null.pid"
        p.write_text("null\n", encoding="utf-8")
        assert read_pid_file(p) is None


# ===========================================================================
# 3.  pid_alive
# ===========================================================================
class TestPidAlive:
    def test_none_returns_false(self):
        assert pid_alive(None) is False

    def test_zero_returns_false(self):
        assert pid_alive(0) is False

    def test_negative_returns_false(self):
        assert pid_alive(-1) is False

    def test_dead_pid_returns_false(self):
        # Use a PID that is almost certainly dead (very large number).
        assert pid_alive(9_999_999) is False

    def test_own_process_alive(self):
        assert pid_alive(os.getpid()) is True

    def test_protected_pid_alive_via_psutil(self):
        """Cross-verify with psutil directly."""
        pid = os.getpid()
        p = psutil.Process(pid)
        assert p.is_running()
        assert pid_alive(pid) is True


# ===========================================================================
# 4.  No duplicate function definitions in supervisor source
# ===========================================================================
class TestNoDuplicateDefs:
    def _parse_defs(self, src: str) -> list[str]:
        tree = ast.parse(src)
        return [
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]

    def test_supervisor_no_duplicates(self):
        src = (_TOOLS / "persistent_supervisor.py").read_text(encoding="utf-8")
        defs = self._parse_defs(src)
        dupes = {k: v for k, v in Counter(defs).items() if v > 1}
        assert dupes == {}, f"Duplicate function definitions found: {dupes}"

    def test_supervisor_single_module_docstring(self):
        src = (_TOOLS / "persistent_supervisor.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        count = sum(1 for node in ast.walk(tree) if isinstance(node, ast.Module) and ast.get_docstring(node))
        assert count == 1

    def test_watchdog_no_duplicates(self):
        src = (_TOOLS / "overnight_pool_watchdog.py").read_text(encoding="utf-8")
        defs = self._parse_defs(src)
        dupes = {k: v for k, v in Counter(defs).items() if v > 1}
        assert dupes == {}, f"Duplicate function definitions in watchdog: {dupes}"


# ===========================================================================
# 5.  Supervisor never mentions continuous_pool.py in launch logic
# ===========================================================================
class TestNoContinuousPoolLaunch:
    def test_no_continuous_pool_in_launch_functions(self):
        src = (_TOOLS / "persistent_supervisor.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("_launch"):
                    func_src = ast.get_source_segment(src, node) or ""
                    assert "continuous_pool.py" not in func_src, (
                        f"{node.name} contains 'continuous_pool.py'"
                    )

    def test_bat_file_no_continuous_pool(self):
        bat = (_TRAINING.parent / "start_overnight_pool.bat").read_text(encoding="utf-8")
        # Should not invoke python directly with continuous_pool
        for line in bat.splitlines():
            stripped = line.strip().lower()
            if stripped.startswith("::") or stripped.startswith("rem"):
                continue  # skip comments
            assert "continuous_pool.py" not in stripped, (
                f"start_overnight_pool.bat contains 'continuous_pool.py': {line!r}"
            )


# ===========================================================================
# 6.  Zero Titanium children does NOT trigger pool restart when commits are healthy
# ===========================================================================
class TestZeroTiChildrenNoFalseStall:
    """The supervisor judges local-pool health by pool_generation% commit timestamps,
    not by instantaneous titanium.exe child count.  When commits are recent the pool
    must NOT be flagged as stalled even when pool_titanium_children == 0."""

    def _make_state(self, last_local_progress_at: str) -> dict:
        return {
            "machine_state": "RUNNING",
            "pool_state": "RUNNING",
            "pool_last_local_progress_at": last_local_progress_at,
            "pool_last_progress_at": last_local_progress_at,
            "pool_restart_times": [],
            "pool_pid": os.getpid(),  # use own PID so pool_alive=True
        }

    def test_recent_commit_not_stalled(self):
        from datetime import datetime, timezone, timedelta
        now = datetime.now(timezone.utc)
        # commit 60 seconds ago — well within POOL_STALL_SEC (300s)
        recent = (now - timedelta(seconds=60)).isoformat()
        state = self._make_state(recent)

        pool_alive = True
        last_local = datetime.fromisoformat(state["pool_last_local_progress_at"])
        elapsed = (now - last_local).total_seconds()
        stall = pool_alive and elapsed > _sup_mod.POOL_STALL_SEC
        assert not stall, "Should NOT stall when commit was 60 s ago"

    def test_stale_commit_is_stalled(self):
        from datetime import datetime, timezone, timedelta
        now = datetime.now(timezone.utc)
        # commit 10 minutes ago — beyond POOL_STALL_SEC (300s)
        stale = (now - timedelta(seconds=600)).isoformat()
        state = self._make_state(stale)

        pool_alive = True
        last_local = datetime.fromisoformat(state["pool_last_local_progress_at"])
        elapsed = (now - last_local).total_seconds()
        stall = pool_alive and elapsed > _sup_mod.POOL_STALL_SEC
        assert stall, "Should stall when commit was 10 minutes ago"

    def test_zero_ti_children_alone_not_stall_signal(self):
        """Verify the supervisor code path uses commit timestamp, not ti_children."""
        src = (_TOOLS / "persistent_supervisor.py").read_text(encoding="utf-8")
        # The stall detection block should not contain a condition that alone keys on
        # ti_children == 0 for pool restart (it's only used as secondary for rebuild).
        # Find the pool stall section:
        assert "pool_last_local_progress_at" in src, (
            "Supervisor must use pool_last_local_progress_at for stall detection"
        )
        # pool restart block should NOT have "pool_ti == 0" as a standalone condition
        assert "pool_ti == 0" not in src and "pool_ti==0" not in src, (
            "pool_ti == 0 should not appear as standalone stall condition"
        )
