from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tools.operations.opponent_curriculum import (
    MAX_VISITS,
    claim_game,
    load_state,
    preferred_adaptive_opponent,
    record_result,
)


class CurriculumTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.state = root / "state.json"
        self.events = root / "events.jsonl"

    def tearDown(self):
        self.tmp.cleanup()

    def test_claims_alternate_colors_and_persist(self):
        claims = [claim_game("ka", self.state) for _ in range(4)]
        self.assertEqual([c["our_is_p1"] for c in claims], [True, False, True, False])
        self.assertEqual([c["opponent_visits"] for c in claims], [1, 1, 1, 1])

    def test_window_advances_by_twenty_at_target(self):
        for i in range(16):
            result = record_result(
                "ka", our_win=i < 8, game_id=str(i), visits=1,
                state_path=self.state, events_path=self.events,
            )
        self.assertEqual(result["visits"], 21)
        self.assertEqual(result["window_games"], 0)
        event = json.loads(self.events.read_text(encoding="utf-8").strip())
        self.assertEqual(event["old_visits"], 1)
        self.assertEqual(event["new_visits"], 21)

    def test_losing_window_never_decreases(self):
        for i in range(16):
            result = record_result(
                "zero", our_win=False, game_id=str(i), visits=1,
                state_path=self.state, events_path=self.events,
            )
        self.assertEqual(result["visits"], 1)

    def test_cap_is_enforced(self):
        state = load_state(self.state)
        state["opponents"]["ka"]["visits"] = MAX_VISITS - 10
        from tools.operations.opponent_curriculum import save_state
        save_state(state, self.state)
        for i in range(16):
            result = record_result(
                "ka", our_win=True, game_id=str(i), visits=MAX_VISITS - 10,
                state_path=self.state, events_path=self.events,
            )
        self.assertEqual(result["visits"], MAX_VISITS)

    def test_zero_is_preferred_until_a_crushing_complete_window(self):
        state = load_state(self.state)
        self.assertEqual(preferred_adaptive_opponent(state), "zero")
        for i in range(16):
            record_result(
                "zero", our_win=i < 4, game_id=str(i), visits=1,
                state_path=self.state, events_path=self.events,
            )
        self.assertEqual(preferred_adaptive_opponent(load_state(self.state)), "ka")

    def test_zero_remains_preferred_when_minimum_budget_is_playable(self):
        for i in range(16):
            record_result(
                "zero", our_win=i < 5, game_id=str(i), visits=1,
                state_path=self.state, events_path=self.events,
            )
        self.assertEqual(preferred_adaptive_opponent(load_state(self.state)), "zero")


if __name__ == "__main__":
    unittest.main()
