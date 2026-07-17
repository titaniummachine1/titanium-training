from __future__ import annotations

import random
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

from opening_prefix_index import OpeningPrefixIndex
from oracle_game_factory.worker import opening_temperature_for_move as oracle_opening_temperature_for_move
from oracle_game_factory.worker import choose_eval_move_by_temperature
from self_play_overnight import (
    OpeningExplorationConfig,
    opening_temperature_for_move,
    play_one_game,
)


def test_plies_1_to_4_return_zero():
    for ply in (1, 2, 3, 4):
        temp, novelty = opening_temperature_for_move(ply, False, True)
        assert temp == 0.0
        assert novelty is False


def test_known_prefix_ply_5_returns_one():
    temp, novelty = opening_temperature_for_move(5, False, True)
    assert temp == 1.0
    assert novelty is False


def test_known_prefix_ply_6_returns_095():
    temp, novelty = opening_temperature_for_move(6, False, True)
    assert abs(temp - 0.95) < 1e-9
    assert novelty is False


def test_unknown_prefix_sets_novelty_permanently():
    temp, novelty = opening_temperature_for_move(5, False, False)
    assert temp == 0.0
    assert novelty is True
    later_temp, later_novelty = opening_temperature_for_move(6, novelty, True)
    assert later_temp == 0.0
    assert later_novelty is True


def test_after_novelty_all_later_moves_zero():
    for ply in (7, 8, 9, 10):
        temp, novelty = opening_temperature_for_move(ply, True, True)
        assert temp == 0.0
        assert novelty is True


def test_mixed_games_get_temperature_on_known_prefixes():
    tmp = Path(tempfile.mkdtemp())
    idx = OpeningPrefixIndex(tmp / "p.db")
    try:
        idx.register_game(["e2", "e8", "e3", "e7"], 1, source="seed", max_ply=16)
        cfg = OpeningExplorationConfig(enabled=True)
        captured: list[float] = []

        def fake_engine(moves, time_sec, weights, nodes=None, temperature=0.0, engine=None):
            captured.append(temperature)
            seq = ["e2", "e8", "e3", "e7", "e4", "e6"]
            return seq[len(moves)] if len(moves) < len(seq) else None

        with patch("self_play_overnight.engine_move_budget", side_effect=fake_engine), patch(
            "self_play_overnight.choose_temperature_move",
            side_effect=lambda _moves, best, *_args, **_kwargs: (best, False, False),
        ):
            r = play_one_game(
                "g1",
                0.01,
                None,
                None,
                mixed=True,
                current_is_p0=True,
                opening_exploration=cfg,
                prefix_index=idx,
                rng=random.Random(1),
            )
        assert r["explored_moves"] > 0
        assert captured[:4] == [0.0, 0.0, 0.0, 0.0]
        assert captured[4] > 0.0
    finally:
        idx.close()
        shutil.rmtree(tmp, ignore_errors=True)


def test_local_and_oracle_share_temperature_calculation():
    for ply, known in ((4, True), (5, True), (6, True), (5, False)):
        local = opening_temperature_for_move(ply, False, known)
        oracle = oracle_opening_temperature_for_move(ply, False, known)
        assert local == oracle


def test_no_move_aborts_instead_of_draw():
    def fake_engine(moves, time_sec, weights, nodes=None, temperature=0.0, engine=None):
        return None

    with patch("self_play_overnight.engine_move_budget", side_effect=fake_engine):
        r = play_one_game(
            "g_abort",
            0.01,
            None,
            None,
            mixed=True,
            current_is_p0=True,
            rng=random.Random(3),
        )
    assert r["aborted"] is True
    assert r["abort_reason"] == "max_plies_or_no_move"
    assert r["outcome_p0"] is None
    assert r["current_won"] is None


def test_repeated_position_sixth_occurrence_is_draw():
    cycle = ["e2", "e8", "e1", "e9"]

    def fake_engine(moves, time_sec, weights, nodes=None, temperature=0.0, engine=None):
        return cycle[len(moves) % len(cycle)]

    with patch("self_play_overnight.engine_move_budget", side_effect=fake_engine):
        r = play_one_game(
            "g_repeat",
            0.01,
            None,
            None,
            mixed=True,
            current_is_p0=True,
            rng=random.Random(4),
        )
    assert r.get("aborted") is False
    assert r["draw_reason"] == "repetition"
    assert r["outcome_p0"] == 0
    assert r["current_won"] is None
    assert r["plies"] == 20


def test_novel_exit_after_unknown_resulting_prefix():
    tmp = Path(tempfile.mkdtemp())
    idx = OpeningPrefixIndex(tmp / "p.db")
    try:
        idx.register_game(["e2", "e8", "e3", "e7"], 1, source="seed", max_ply=16)
        cfg = OpeningExplorationConfig(enabled=True)

        def fake_engine(moves, time_sec, weights, nodes=None, temperature=0.0, engine=None):
            seq = ["e2", "e8", "e3", "e7", "d3", "e6"]
            return seq[len(moves)] if len(moves) < len(seq) else None

        with patch("self_play_overnight.engine_move_budget", side_effect=fake_engine), patch(
            "self_play_overnight.choose_temperature_move",
            side_effect=lambda _moves, best, *_args, **_kwargs: (best, False, False),
        ):
            r = play_one_game(
                "g2",
                0.01,
                None,
                None,
                mixed=False,
                current_is_p0=True,
                opening_exploration=cfg,
                prefix_index=idx,
                rng=random.Random(2),
            )
        assert r.get("novel_prefix") is True
        assert r.get("novel_exit_ply") == 5
        assert r["move_temperatures"][0] == 1.0
    finally:
        idx.close()
        shutil.rmtree(tmp, ignore_errors=True)


def test_opening_temperature_changes_played_move_when_sampler_picks_alternative():
    tmp = Path(tempfile.mkdtemp())
    idx = OpeningPrefixIndex(tmp / "p.db")
    try:
        idx.register_game(["e2", "e8", "e3", "e7"], 1, source="seed", max_ply=16)
        cfg = OpeningExplorationConfig(enabled=True)
        sampled: list[tuple[list[str], str, float]] = []

        def fake_engine(moves, time_sec, weights, nodes=None, temperature=0.0, engine=None):
            seq = ["e2", "e8", "e3", "e7", "e4", "e6"]
            return seq[len(moves)] if len(moves) < len(seq) else None

        def fake_sampler(moves, best, _weights, temperature, **_kwargs):
            sampled.append((list(moves), best, temperature))
            if len(moves) == 4:
                return "d4h", True, False
            return best, False, False

        with patch("self_play_overnight.engine_move_budget", side_effect=fake_engine), patch(
            "self_play_overnight.choose_temperature_move", side_effect=fake_sampler
        ):
            r = play_one_game(
                "g_temp_alt",
                0.01,
                None,
                None,
                mixed=False,
                current_is_p0=True,
                opening_exploration=cfg,
                prefix_index=idx,
                rng=random.Random(5),
            )

        assert sampled
        assert r["moves"][4] == "d4h"
        assert r["explored_moves"] > 0
    finally:
        idx.close()
        shutil.rmtree(tmp, ignore_errors=True)


def test_oracle_temperature_fallback_samples_without_root_moves(tmp_path: Path):
    with patch(
        "oracle_game_factory.worker.legal_moves",
        return_value=["best", "close", "bad"],
    ), patch(
        "oracle_game_factory.worker.child_eval_scores",
        return_value={"best": 100, "close": 98, "bad": -300},
    ):
        seen = {
            choose_eval_move_by_temperature(
                tmp_path / "engine",
                ["e2", "e8", "e3", "e7"],
                "best",
                None,
                1.0,
                top_n=2,
                rng=random.Random(seed),
            )[0]
            for seed in range(30)
        }

    assert seen <= {"best", "close"}
    assert "close" in seen
