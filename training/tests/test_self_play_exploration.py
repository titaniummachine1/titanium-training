from __future__ import annotations

import random
from unittest.mock import patch

from self_play_overnight import ExplorationConfig, choose_exploration_move


def test_exploration_keeps_best_before_start_ply():
    cfg = ExplorationConfig(start_ply=8, chance=1.0)

    with patch("self_play_overnight.legal_moves") as legal:
        mv = choose_exploration_move(["e2", "e8"], "e3", None, cfg, random.Random(1))

    assert mv == "e3"
    legal.assert_not_called()


def test_exploration_can_pick_close_alternative_and_rejects_blunder():
    cfg = ExplorationConfig(
        start_ply=2,
        chance=1.0,
        max_loss_cp=80,
        candidate_count=4,
        top_n=3,
        temperature_cp=10.0,
        wall_bonus_cp=0,
    )

    with patch(
        "self_play_overnight.legal_moves",
        return_value=["best", "close", "bad", "other"],
    ), patch(
        "self_play_overnight.child_eval_scores",
        return_value={"best": 100, "close": 95, "bad": -200, "other": 20},
    ):
        seen = {
            choose_exploration_move(["e2", "e8"], "best", None, cfg, random.Random(seed))
            for seed in range(50)
        }

    assert seen <= {"best", "close"}
    assert "close" in seen
