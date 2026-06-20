from __future__ import annotations

import unittest

from experiments.lmr.collect_search_importance import search_pressure_target, target_components
from experiments.evaluation.compare_pressure_sources import pearson, ranks
from experiments.lmr.train_search_importance import grouped_split, row_is_trainable


def probe(best: str, score: int, depth: int, nodes: int, depth_log: list[dict]) -> dict:
    return {
        "best": best,
        "score": score,
        "depth": depth,
        "nodes": nodes,
        "depth_log": depth_log,
    }


class SearchImportanceTests(unittest.TestCase):
    def test_correlation_helpers_handle_ties(self):
        self.assertAlmostEqual(pearson([1, 2, 3], [2, 4, 6]), 1.0)
        self.assertEqual(ranks([5, 1, 5]), [1.5, 0.0, 1.5])

    def test_stable_chain_has_low_pressure(self):
        shallow = probe("e5", 100, 2, 100, [])
        deep = probe("e5", 105, 5, 800, [
            {"depth": 2, "score": 100, "pv": "e5"},
            {"depth": 3, "score": 103, "pv": "e5"},
            {"depth": 4, "score": 102, "pv": "e5"},
            {"depth": 5, "score": 105, "pv": "e5"},
        ])
        self.assertLess(search_pressure_target(target_components(shallow, deep)), -0.8)

    def test_move_and_iteration_flips_have_high_pressure(self):
        shallow = probe("e5", 100, 2, 100, [])
        deep = probe("c4h", -700, 5, 12_000, [
            {"depth": 2, "score": 100, "pv": "e5"},
            {"depth": 3, "score": -200, "pv": "d4v"},
            {"depth": 4, "score": 250, "pv": "e5"},
            {"depth": 5, "score": -700, "pv": "c4h"},
        ])
        components = target_components(shallow, deep)
        self.assertEqual(components["iteration_flip_rate"], 1.0)
        self.assertGreater(search_pressure_target(components), 0.6)

    def test_terminal_and_forced_native_rows_are_not_trainable(self):
        self.assertFalse(row_is_trainable({
            "teacher": "titanium-native",
            "shallow": {"best": "(none)", "score": 0},
            "deep": {"best": "(none)", "score": 0},
        }))
        self.assertFalse(row_is_trainable({
            "teacher": "titanium-native",
            "shallow": {"best": "e5", "score": 100},
            "deep": {"best": "e5", "score": 31_998},
        }))
        self.assertTrue(row_is_trainable({
            "teacher": "zero-ink",
            "search_pressure": 0.5,
        }))

    def test_grouped_split_never_leaks_a_game_across_teachers(self):
        rows = [
            {"teacher": "titanium-native", "source_game_key": "shared", "moves_bin": "a"},
            {"teacher": "quoridor-zero.ink", "source_game_key": "shared", "moves_bin": "b"},
            {"teacher": "titanium-native", "source_game_key": "native-2", "moves_bin": "c"},
            {"teacher": "quoridor-zero.ink", "source_game_key": "zero-2", "moves_bin": "d"},
        ]
        train, val = grouped_split(rows, seed=7, val_fraction=0.5)
        train_keys = {row["source_game_key"] for row in train}
        val_keys = {row["source_game_key"] for row in val}
        self.assertTrue(train_keys.isdisjoint(val_keys))

    def test_grouped_split_keeps_existing_games_fixed_when_data_is_appended(self):
        rows = [
            {"teacher": "titanium-native", "source_game_key": f"game-{i}", "moves_bin": str(i)}
            for i in range(40)
        ]
        train, val = grouped_split(rows, seed=1337)
        assignment = {
            row["source_game_key"]: "train" for row in train
        } | {
            row["source_game_key"]: "val" for row in val
        }
        extended = rows + [
            {"teacher": "titanium-native", "source_game_key": f"new-{i}", "moves_bin": f"n{i}"}
            for i in range(20)
        ]
        train2, val2 = grouped_split(extended, seed=1337)
        assignment2 = {
            row["source_game_key"]: "train" for row in train2
        } | {
            row["source_game_key"]: "val" for row in val2
        }
        self.assertTrue(all(assignment[key] == assignment2[key] for key in assignment))


if __name__ == "__main__":
    unittest.main()
