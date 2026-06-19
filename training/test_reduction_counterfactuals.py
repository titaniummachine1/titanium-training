from __future__ import annotations

import unittest

from training.reduction_counterfactual_schema import (
    SCHEMA,
    bound_class,
    classify_pair,
    stable_partition,
    validate_row,
    wilson_lower,
)


def event(**overrides):
    row = {
        "ordinal": 7,
        "parent_hash": "aa",
        "child_hash": "bb",
        "move": "c4h",
        "depth": 6,
        "ply": 2,
        "alpha": 10,
        "beta": 11,
        "move_index": 12,
        "base_reduction": 2,
        "extra_reduction": False,
        "verification_triggered": False,
        "score": 0,
        "nodes": 100,
        "hidden": [0.0] * 32,
    }
    row.update(overrides)
    return row


class ReductionCounterfactualTests(unittest.TestCase):
    def test_fail_low_numeric_difference_preserves_decision(self):
        baseline = event(score=0, nodes=100)
        counterfactual = event(score=-20, nodes=70, extra_reduction=True)
        result = classify_pair(
            baseline, counterfactual, minimum_nodes_saved=10, minimum_savings_ratio=0.1
        )
        self.assertEqual(result["sample_status"], "SAFE")
        self.assertTrue(result["activate_plus_one"])
        self.assertEqual(result["net_nodes_saved"], 30)

    def test_changed_bound_is_unsafe_even_when_it_saves_nodes(self):
        baseline = event(score=0, nodes=100)
        counterfactual = event(score=20, nodes=5, extra_reduction=True)
        result = classify_pair(
            baseline, counterfactual, minimum_nodes_saved=1, minimum_savings_ratio=0.0
        )
        self.assertEqual(result["sample_status"], "UNSAFE")
        self.assertFalse(result["activate_plus_one"])

    def test_context_mismatch_is_unknown_not_unsafe(self):
        result = classify_pair(
            event(), event(move="d4h", extra_reduction=True),
            minimum_nodes_saved=1, minimum_savings_ratio=0.0,
        )
        self.assertEqual(result["sample_status"], "UNKNOWN")

    def test_safe_but_unprofitable_is_not_activation_positive(self):
        result = classify_pair(
            event(nodes=100), event(nodes=99, extra_reduction=True),
            minimum_nodes_saved=8, minimum_savings_ratio=0.05,
        )
        self.assertTrue(result["safe_plus_one_reduction"])
        self.assertFalse(result["activate_plus_one"])

    def test_partition_is_stable_and_game_disjoint(self):
        self.assertEqual(stable_partition("game-a", 7), stable_partition("game-a", 7))
        self.assertIn(stable_partition("game-b", 7), {"train", "calibration", "final_test"})

    def test_validation_rejects_activation_without_safety(self):
        with self.assertRaises(ValueError):
            validate_row({
                "schema": SCHEMA,
                "sample_status": "UNKNOWN",
                "safe_plus_one_reduction": False,
                "activate_plus_one": True,
            })

    def test_wilson_is_conservative_for_small_samples(self):
        self.assertLess(wilson_lower(3, 3), 0.5)
        self.assertGreater(wilson_lower(1000, 1000), 0.99)

    def test_bound_classes(self):
        self.assertEqual(bound_class(0, 0, 1), "FAIL_LOW")
        self.assertEqual(bound_class(1, 0, 1), "FAIL_HIGH")
        self.assertEqual(bound_class(5, 0, 10), "EXACT")


if __name__ == "__main__":
    unittest.main()

