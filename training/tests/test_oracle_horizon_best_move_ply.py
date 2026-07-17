from __future__ import annotations

from training.oracle_horizon.horizon_rows import (
    build_horizon_row,
    initial_target_move,
    ply_move,
)


def test_each_horizon_row_keeps_the_move_for_its_ply():
    oracle_index = 3
    trajectory_moves = {3: "entry", 2: "ancestor-1", 1: "ancestor-2", 0: "ancestor-3"}
    rows = []

    for index in range(oracle_index, -1, -1):
        move = ply_move(
            index=index,
            oracle_index=oracle_index,
            oracle_move=trajectory_moves[oracle_index],
            deploy_move=trajectory_moves[index],
            proven_move=None if index == oracle_index else trajectory_moves[index],
            exact=index == oracle_index,
        )
        rows.append(build_horizon_row(
            position_id=f"g:{index}",
            packed_state_hex=f"{index:02x}",
            game_id="g",
            lineage_id="l",
            index=index,
            oracle_index=oracle_index,
            band=0,
            label_class="EXACT_ORACLE",
            primary=True,
            oracle_wdl="W",
            oracle_proven=True,
            backed_proven=index != oracle_index,
            deploy={"score": 100, "selected_move": trajectory_moves[index]},
            best_move=move,
            needs_learning_value=False,
            needs_learning_reasons=[],
            ladder=[],
            weights_sha256="w",
            engine_sha256="e",
        ))

    assert [row["best_move"] for row in rows] == [
        trajectory_moves[index] for index in range(oracle_index, -1, -1)
    ]
    assert [row["selected_move"] for row in rows] == [row["best_move"] for row in rows]


def test_horizon_row_moves_are_legal_for_their_ply():
    oracle_index = 2
    legal_moves = {
        2: ["entry", "other-entry"],
        1: ["ancestor-1", "other-1"],
        0: ["ancestor-2", "other-2"],
    }
    rows = []
    for index in range(oracle_index, -1, -1):
        move = ply_move(
            index=index,
            oracle_index=oracle_index,
            oracle_move="entry",
            deploy_move=legal_moves[index][0],
            proven_move=None if index == oracle_index else legal_moves[index][0],
            exact=index == oracle_index,
        )
        rows.append({"ply": index, "best_move": move})

    assert all(row["best_move"] in legal_moves[row["ply"]] for row in rows)


def test_ancestor_has_no_entry_move_before_ladder_proof():
    assert initial_target_move(index=3, oracle_index=3, oracle_move="entry") == "entry"
    assert initial_target_move(index=2, oracle_index=3, oracle_move="entry") is None
