from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

from titanium_training.store.move_codec import algebraic_to_ace
from zero_teacher.client import (
    START_STATE,
    ace_to_zero_move,
    paired_search_pressure,
    zero_move_text,
    zero_to_ace_move,
)
from zero_teacher.collect_budget import consume_until_budget


class ZeroBridgeTests(unittest.TestCase):
    def test_real_start_state(self):
        self.assertEqual(START_STATE["player0Cell"], 4)
        self.assertEqual(START_STATE["player1Cell"], 76)

    def test_move_round_trip(self):
        for text in ("e1", "e2", "e8", "e9", "a1h", "h8h", "a1v", "h8v"):
            ace = algebraic_to_ace(text)
            move = ace_to_zero_move(ace)
            self.assertEqual(zero_to_ace_move(move), ace)
            self.assertEqual(zero_move_text(move), text)

    def test_paired_pressure_detects_best_move_change(self):
        def row(target, fraction, value=0.0):
            return {"move": {"kind": "pawn", "target": target}, "visitFraction": fraction, "q": value}
        stable = {"rootValue": 0.1, "moves": [row(13, 0.9), row(5, 0.1)]}
        changed = {"rootValue": 0.4, "moves": [row(5, 0.8), row(13, 0.2)]}
        result = paired_search_pressure(stable, changed)
        self.assertTrue(result["best_move_changed"])
        self.assertGreater(result["search_pressure"], 0.0)

    def test_budget_stream_stops_at_first_deep_snapshot(self):
        closed = []

        def chunks():
            try:
                yield {"totalVisits": 60, "moves": [1]}
                yield {"totalVisits": 430, "moves": [1]}
                yield {"totalVisits": 8_000, "moves": [1]}
            finally:
                closed.append(True)

        consumed = consume_until_budget(chunks(), deep_visits=400, max_chunks=32)
        self.assertEqual([row["totalVisits"] for row in consumed], [60, 430])
        self.assertEqual(closed, [True])


if __name__ == "__main__":
    unittest.main()
