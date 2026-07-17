"""Smoke tests for the bounded Ka-AB teacher adapter."""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "ka_ab_teacher.mjs"
ACE = Path(os.environ.get("KA_ACE", ""))
if not ACE.is_file():
    ACE = HERE.parents[2] / "reference" / "ace.html"
NODE = shutil.which("node")


class KaAbTeacherSmoke(unittest.TestCase):
    def setUp(self) -> None:
        if not NODE:
            self.skipTest("Node unavailable")

    def run_adapter(self, *args: str) -> tuple[subprocess.CompletedProcess[str], dict]:
        proc = subprocess.run(
            [NODE, str(SCRIPT), *args],
            cwd=str(HERE.parents[2]),
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        row = json.loads(proc.stdout) if proc.stdout else {}
        return proc, row

    def test_bounded_schema_and_repeatability(self) -> None:
        if not ACE.is_file():
            self.skipTest(f"supplied Ace bundle unavailable: {ACE}")
        first_proc, first = self.run_adapter("--nodes", "4")
        second_proc, second = self.run_adapter("--nodes", "4")
        self.assertEqual(first_proc.returncode, 0, first_proc.stderr)
        self.assertEqual(second_proc.returncode, 0, second_proc.stderr)
        self.assertEqual(first_proc.stdout.count("\n"), 1)
        self.assertEqual(second_proc.stdout.count("\n"), 1)
        self.assertEqual(first["schema"], "ace-ka-ab-teacher-v1")
        self.assertEqual(first["engine"]["mode"], "ab")
        self.assertTrue(first["engine"]["beta_ab_used"])
        self.assertIn(first["engine"]["backend"], {"wasm-simd", "js"})
        self.assertEqual(first["engine"]["backend_requested"], "auto")
        self.assertEqual(first["budget"]["requested_evals"], 4)
        self.assertEqual(first["budget"]["requested_time_ms"], 0)
        self.assertLessEqual(first["budget"]["actual_evals"], 4)
        self.assertIsInstance(first["teacher"]["best_move_official"], str)
        self.assertTrue(-1 <= first["teacher"]["value_stm"] <= 1)
        stable = ("source", "engine", "position", "teacher")
        for key in stable:
            if key == "engine":
                self.assertEqual(first[key]["config"], second[key]["config"])
            elif key == "source":
                self.assertEqual(first[key], second[key])
            else:
                self.assertEqual(first[key], second[key])

    def test_js_backend_and_benchmark_output(self) -> None:
        if not ACE.is_file():
            self.skipTest(f"supplied Ace bundle unavailable: {ACE}")
        proc, row = self.run_adapter("--backend", "js", "--nodes", "2", "--bench", "2")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(row["engine"]["backend"], "js")
        self.assertEqual(row["engine"]["backend_requested"], "js")
        self.assertEqual(row["benchmark"]["repeats"], 2)
        self.assertEqual(row["benchmark"]["nodes_per_run"], 2)

    def test_default_batch_chunk_in_provenance(self) -> None:
        if not ACE.is_file():
            self.skipTest(f"supplied Ace bundle unavailable: {ACE}")
        proc, row = self.run_adapter("--nodes", "4")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(row["engine"]["config"]["batchChunk"], 8)

    def test_explicit_batch_chunk_in_provenance(self) -> None:
        if not ACE.is_file():
            self.skipTest(f"supplied Ace bundle unavailable: {ACE}")
        proc, row = self.run_adapter("--batch-chunk", "3", "--nodes", "4")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(row["engine"]["config"]["batchChunk"], 3)

    def test_batch_chunk_out_of_range_fails_closed(self) -> None:
        for bad_value in ("0", "33", "-1", "abc"):
            proc, row = self.run_adapter("--batch-chunk", bad_value, "--nodes", "4")
            self.assertNotEqual(proc.returncode, 0, f"batch-chunk={bad_value} should be rejected")
            self.assertEqual(proc.stdout, "")
            self.assertEqual(row, {})
            self.assertIn("batch-chunk", proc.stderr)

    def test_benchmark_reports_backend_batch_chunk_and_safe_throughput(self) -> None:
        if not ACE.is_file():
            self.skipTest(f"supplied Ace bundle unavailable: {ACE}")
        proc, row = self.run_adapter(
            "--backend", "js", "--nodes", "2", "--bench", "3", "--batch-chunk", "4",
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        benchmark = row["benchmark"]
        self.assertEqual(benchmark["backend"], "js")
        self.assertEqual(benchmark["batch_chunk"], 4)
        self.assertGreaterEqual(benchmark["actual_evals"], 0)
        self.assertGreaterEqual(benchmark["total_ms"], 0)
        self.assertGreaterEqual(benchmark["per_run_ms"], 0)
        # Throughput must be a finite, non-negative number even when the run
        # is fast enough that elapsed time rounds to (near) zero: no NaN/Inf
        # from a hidden divide-by-zero.
        throughput = benchmark["throughput_evals_per_sec"]
        self.assertTrue(math.isfinite(throughput))
        self.assertGreaterEqual(throughput, 0)
        # Backend is WASM-SIMD or plain JS on CPU; never claim GPU.
        self.assertNotIn("gpu", json.dumps(benchmark).lower())

    def test_wasm_backend_is_real_or_fails_closed(self) -> None:
        if not ACE.is_file():
            self.skipTest(f"supplied Ace bundle unavailable: {ACE}")
        proc, row = self.run_adapter("--backend", "wasm", "--nodes", "8")
        if proc.returncode == 0:
            self.assertEqual(row["engine"]["backend"], "wasm-simd")
        else:
            self.assertEqual(row, {})
            self.assertIn("ka_backend", proc.stderr)
    def test_missing_ace_path_fails_closed(self) -> None:
        missing = HERE / "does-not-exist-ace.html"
        proc = subprocess.run(
            [NODE, str(SCRIPT), "--ace", str(missing), "--nodes", "4"],
            cwd=str(HERE.parents[2]),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertEqual(proc.stdout, "")


if __name__ == "__main__":
    unittest.main()
