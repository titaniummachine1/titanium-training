#!/usr/bin/env python3
"""Deterministic smoke test for the bounded Ace MCTS teacher adapter."""
from __future__ import annotations

import json
import os
import subprocess
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "ace_mcts_teacher.mjs"
ACE = Path(os.environ.get("USERPROFILE", "C:/Users/Terminatort8000")) / "Downloads" / "ace.html"


class AceMctsTeacherSmoke(unittest.TestCase):
    def test_default_mcts_smoke(self) -> None:
        if not ACE.is_file():
            self.fail(f"supplied Ace bundle missing: {ACE}")
        proc = subprocess.run(
            ["node", str(SCRIPT), "--nodes", "4"],
            cwd=str(HERE.parents[2]),
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        row = json.loads(proc.stdout)
        self.assertEqual(row["schema"], "ace-mcts-teacher-v1")
        self.assertEqual(row["engine"]["mode"], "mcts")
        self.assertTrue(row["engine"]["certified_default"])
        self.assertFalse(row["engine"]["beta_ab_used"])
        self.assertEqual(len(row["source"]["ace_bundle_sha256"]), 64)
        self.assertEqual(row["budget"]["requested_nodes"], 4)
        self.assertLessEqual(row["budget"]["actual_nodes"], 4)
        self.assertTrue(row["legal_policy"])
        self.assertTrue(all("official" in move and "visits" in move for move in row["legal_policy"]))


if __name__ == "__main__":
    unittest.main()
