from __future__ import annotations

import unittest
from unittest.mock import patch

from tools.operations import run_nnue_cycle


class RunNnueCycleTests(unittest.TestCase):
    @patch.object(run_nnue_cycle, "_log")
    @patch.object(run_nnue_cycle, "mark_game_trained")
    @patch.object(run_nnue_cycle, "pretrain_sanity_ok", return_value=(False, "bad stamp"))
    def test_blocked_game_stays_pending(self, _sanity, mark_trained, _log):
        self.assertEqual(run_nnue_cycle.run_on_game(42), 1)
        mark_trained.assert_not_called()

    @patch.object(run_nnue_cycle, "_log")
    @patch.object(run_nnue_cycle, "run_on_game", side_effect=[1, 0])
    @patch.object(run_nnue_cycle, "untrained_game_ids", return_value=[10, 11])
    def test_catch_up_stops_before_skipping_failed_row(self, _pending, run_game, _log):
        with patch(
            "titanium_training.training.guards.load_guard_state",
            return_value={"last_trained_game_id": 9},
        ):
            self.assertEqual(run_nnue_cycle.run_catch_up(), 1)
        run_game.assert_called_once_with(10, dry_run=False)

    @patch.object(run_nnue_cycle, "_log")
    @patch.object(run_nnue_cycle, "record_elo_sample")
    @patch.object(run_nnue_cycle, "snapshot_weights")
    @patch.object(run_nnue_cycle, "load_manifest", return_value={})
    @patch.object(run_nnue_cycle, "game_source_tag", return_value="adaptive-ka")
    @patch.object(run_nnue_cycle, "enforce_artifact_cap", return_value=(True, "cap ok"))
    @patch.object(run_nnue_cycle, "pretrain_sanity_ok", return_value=(True, "ready"))
    @patch.object(run_nnue_cycle, "mark_game_trained")
    def test_dry_run_does_not_advance_cursor(
        self,
        mark_trained,
        _sanity,
        _cap,
        _source,
        _manifest,
        _snapshot,
        _elo,
        _log,
    ):
        with patch(
            "titanium_training.training.guards.load_guard_state",
            return_value={"games_trained": 1},
        ):
            self.assertEqual(run_nnue_cycle.run_on_game(43, dry_run=True), 0)
        mark_trained.assert_not_called()


if __name__ == "__main__":
    unittest.main()
