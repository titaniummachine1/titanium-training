from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments.features.color_rotation import claim_local_game


class ColorRotationTests(unittest.TestCase):
    def test_colors_and_openings_alternate_persistently(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rotation.json"
            self.assertEqual(claim_local_game("self", path), {"game_index": 0, "our_is_p1": True})
            self.assertEqual(claim_local_game("self", path), {"game_index": 1, "our_is_p1": False})
            self.assertEqual(claim_local_game("self", path), {"game_index": 2, "our_is_p1": True})
            self.assertEqual(claim_local_game("pure", path), {"game_index": 0, "our_is_p1": True})


if __name__ == "__main__":
    unittest.main()
