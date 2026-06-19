from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))

from position_store_lib import (  # noqa: E402
    add_label,
    BinaryShardWriter,
    audit_database,
    connect_db,
    db_summary,
    import_alpha_selfplay_text,
    import_binary_shard,
    import_games_file,
    import_path,
    import_search_pressure_jsonl,
    insert_game,
)
from position_store_compact import (  # noqa: E402
    LABEL_TYPE_TEACHER_VALUE,
    export_training_binary,
    rebuild_compact_db,
    score_semantics_report,
    storage_audit,
)
from position_store_state import (  # noqa: E402
    PAWN_EAST,
    PAWN_NORTH,
    PAWN_NORTHEAST,
    PAWN_NORTHWEST,
    PAWN_SOUTH,
    PAWN_SOUTHEAST,
    PAWN_SOUTHWEST,
    PAWN_WEST,
    VALID_MOVE_CODES,
    PositionState,
    apply_move,
    cell_to_notation,
    decode_move,
    encode_move,
    moves_from_u8_blob,
    moves_to_u8_blob,
    notation_to_wall_slot,
    replay_game,
    wall_code_to_notation,
    wall_notation_to_code,
)


def make_jump_state() -> PositionState:
    return PositionState(player0_cell=31, player1_cell=40, side_to_move=0)


def make_diag_state() -> PositionState:
    state = PositionState(player0_cell=31, player1_cell=40, side_to_move=0)
    blocked = apply_move(state, "d5h")
    return PositionState(
        player0_cell=blocked.player0_cell,
        player1_cell=blocked.player1_cell,
        player0_walls=blocked.player0_walls,
        player1_walls=blocked.player1_walls,
        horizontal_walls=blocked.horizontal_walls,
        vertical_walls=blocked.vertical_walls,
        side_to_move=0,
    )


def test_wall_move_codes_roundtrip_all_128() -> None:
    seen = set()
    for code in range(128):
        notation = wall_code_to_notation(code)
        roundtrip = wall_notation_to_code(notation)
        assert roundtrip == code
        assert notation not in seen
        seen.add(notation)
    assert len(seen) == 128


def test_pawn_move_codes_roundtrip_start_and_jump_cases() -> None:
    start = PositionState.initial()
    assert encode_move(start, "e2") == PAWN_NORTH
    assert decode_move(start, PAWN_NORTH) == "e2"

    west_state = replay_game(["e2", "e8", "d2"])[-1]
    assert decode_move(west_state, PAWN_SOUTH) == "e7"

    jump_state = make_jump_state()
    assert encode_move(jump_state, "e6") == PAWN_NORTH
    assert decode_move(jump_state, PAWN_NORTH) == "e6"

    diag_state = make_diag_state()
    diag_moves = {encode_move(diag_state, move): move for move in ("d5", "f5")}
    assert diag_moves[PAWN_NORTHWEST] == "d5"
    assert diag_moves[PAWN_NORTHEAST] == "f5"
    assert decode_move(diag_state, PAWN_NORTHWEST) == "d5"
    assert decode_move(diag_state, PAWN_NORTHEAST) == "f5"


def test_move_blob_roundtrip_game() -> None:
    moves = ["e2", "e8", "e3", "e7", "d3h", "e6", "d3"]
    blob = moves_to_u8_blob(moves)
    assert len(blob) == len(moves)
    assert moves_from_u8_blob(blob) == moves


def test_position_pack_unpack_and_hash_stability() -> None:
    state = replay_game(["e2", "e8", "d3h", "d7v"])[-1]
    packed = state.packed_state()
    unpacked = PositionState.unpack_state(packed)
    assert unpacked == state
    assert unpacked.canonical_hash() == state.canonical_hash()
    assert unpacked.fast_hash() == state.fast_hash()


def test_state_cycle_is_possible() -> None:
    states = replay_game(["e2", "e8", "e1", "e9"])
    assert states[0] == states[-1]


def test_reserved_move_codes_fail_closed() -> None:
    state = PositionState.initial()
    for code in (136, 200, 255):
        assert code not in VALID_MOVE_CODES
        try:
            decode_move(state, code)
        except ValueError:
            pass
        else:
            raise AssertionError(f"reserved code {code} unexpectedly decoded")


def test_side_to_move_and_wall_counts_affect_canonical_state() -> None:
    base = PositionState.initial()
    flipped = PositionState(
        player0_cell=base.player0_cell,
        player1_cell=base.player1_cell,
        player0_walls=base.player0_walls,
        player1_walls=base.player1_walls,
        horizontal_walls=base.horizontal_walls,
        vertical_walls=base.vertical_walls,
        side_to_move=1,
    )
    fewer_walls = PositionState(
        player0_cell=base.player0_cell,
        player1_cell=base.player1_cell,
        player0_walls=9,
        player1_walls=base.player1_walls,
        horizontal_walls=base.horizontal_walls,
        vertical_walls=base.vertical_walls,
        side_to_move=base.side_to_move,
    )
    assert base.packed_state() != flipped.packed_state()
    assert base.packed_state() != fewer_walls.packed_state()
    assert base.canonical_hash() != flipped.canonical_hash()
    assert base.canonical_hash() != fewer_walls.canonical_hash()


def test_duplicate_game_frequency_preserved(tmp_path: Path) -> None:
    db = tmp_path / "graph.db"
    conn = connect_db(db)
    moves = ["e2", "e8", "e3", "e7", "e4", "e6", "d3h", "d6h", "e5"]
    insert_game(conn, moves, 1, source="test-a", source_cohort="test-a")
    insert_game(conn, moves, 1, source="test-b", source_cohort="test-b")
    conn.commit()
    summary = db_summary(db)
    assert summary["games"] == 2
    root_total = conn.execute(
        "SELECT total_visits FROM positions ORDER BY position_id LIMIT 1"
    ).fetchone()[0]
    assert root_total == 2
    edge_visits = conn.execute(
        "SELECT visit_count FROM edges ORDER BY parent_position_id, move_code_u8 LIMIT 1"
    ).fetchone()[0]
    assert edge_visits == 2
    conn.close()


def test_transposition_merges_canonical_position(tmp_path: Path) -> None:
    db = tmp_path / "graph.db"
    conn = connect_db(db)
    g1 = ["e2", "e8", "a1h", "c1h"]
    g2 = ["a1h", "c1h", "e2", "e8"]
    end1 = replay_game(g1)[-1]
    end2 = replay_game(g2)[-1]
    assert end1 == end2
    insert_game(conn, g1, 1, source="g1", source_cohort="g1")
    insert_game(conn, g2, -1, source="g2", source_cohort="g2")
    conn.commit()
    count = conn.execute(
        "SELECT COUNT(*) FROM positions WHERE canonical_hash=? AND packed_state=?",
        (end1.canonical_hash(), end1.packed_state()),
    ).fetchone()[0]
    assert count == 1
    conn.close()


def test_alpha_selfplay_isolated_position_import(tmp_path: Path) -> None:
    db = tmp_path / "graph.db"
    conn = connect_db(db)
    sample = {
        "state": {
            "player0Cell": 4,
            "player1Cell": 76,
            "player0Walls": 10,
            "player1Walls": 10,
            "horizontalWalls": 0,
            "verticalWalls": 0,
            "currentPlayer": 0,
        },
        "policyActions": [13, 82, 145],
        "policyValues": [0.9, 0.05, 0.05],
        "outcome": 1,
        "rootValue": 0.42,
    }
    stats = import_alpha_selfplay_text(conn, json.dumps(sample), source_label="friend")
    conn.commit()
    assert stats.accepted_count == 1
    assert conn.execute("SELECT COUNT(*) FROM positions").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM labels").fetchone()[0] == 1
    conn.close()


def test_search_pressure_import(tmp_path: Path) -> None:
    db = tmp_path / "graph.db"
    conn = connect_db(db)
    path = tmp_path / "search_pressure.jsonl"
    moves = ["e2", "e8", "e3", "e7", "e4"]
    from move_codec import pack_moves  # noqa: E402

    row = {
        "schema": "leaf-search-pressure-v1",
        "moves_bin": __import__("base64").b64encode(pack_moves(moves)).decode("ascii"),
        "ply": 5,
        "src": "pool-test",
        "teacher": "titanium-native",
        "search_pressure": 0.7,
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    stats = import_search_pressure_jsonl(conn, path, dry_run=False)
    conn.commit()
    assert stats.accepted_count == 1
    assert conn.execute("SELECT COUNT(*) FROM labels WHERE label_type='search_pressure'").fetchone()[0] == 1
    conn.close()


def test_games_text_import(tmp_path: Path) -> None:
    db = tmp_path / "graph.db"
    conn = connect_db(db)
    path = tmp_path / "sample.games"
    path.write_text("GAME e2 e8 e3 e7 e4 e6 d3h d6h e5\nRESULT W\n", encoding="utf-8")
    stats = import_games_file(conn, path, dry_run=False)
    conn.commit()
    assert stats.accepted_count == 1
    assert conn.execute("SELECT COUNT(*) FROM games").fetchone()[0] == 1
    conn.close()


def test_binary_shard_roundtrip_and_idempotent_import(tmp_path: Path) -> None:
    db = tmp_path / "graph.db"
    connect_db(db).close()
    shard_dir = tmp_path / "shards"
    writer = BinaryShardWriter(
        shard_dir,
        engine_hash="engine",
        trunk_hash="trunk",
        search_config_hash="cfg",
        worker_id="w1",
        random_seed_range="1-2",
    )
    writer.add_game(["e2", "e8", "e3", "e7", "e4", "e6", "d3h", "d6h", "e5"], result=1)
    shard = writer.write_ready("batch-1")
    stats = import_binary_shard(db, shard, dry_run=False)
    assert stats.accepted_count == 1
    imported = shard.with_suffix(".imported")
    assert imported.exists()


def test_dry_run_import_leaves_database_empty(tmp_path: Path) -> None:
    db = tmp_path / "graph.db"
    connect_db(db).close()
    path = tmp_path / "sample.games"
    path.write_text("GAME e2 e8 e3 e7 e4 e6 d3h d6h e5\nRESULT W\n", encoding="utf-8")
    stats = import_path(db, path, dry_run=True, report_dir=tmp_path)
    assert stats.accepted_count == 1
    conn = connect_db(db)
    assert conn.execute("SELECT COUNT(*) FROM games").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM positions").fetchone()[0] == 0
    conn.close()


def test_audit_passes_for_basic_db(tmp_path: Path) -> None:
    db = tmp_path / "graph.db"
    conn = connect_db(db)
    insert_game(conn, ["e2", "e8", "e3", "e7", "e4", "e6", "d3h", "d6h", "e5"], 1, source="test", source_cohort="test")
    conn.commit()
    conn.close()
    report = audit_database(db)
    assert report["sqlite_integrity_check"] == "ok"
    assert report["issues"] == []


def test_compact_rebuild_and_binary_export(tmp_path: Path) -> None:
    src_db = tmp_path / "graph_v1.db"
    conn = connect_db(src_db)
    insert_game(conn, ["e2", "e8", "e3", "e7", "e4"], 1, source="test", source_cohort="test")
    pos_id = conn.execute("SELECT position_id FROM positions ORDER BY position_id DESC LIMIT 1").fetchone()[0]
    add_label(
        conn,
        pos_id,
        label_type="teacher_value",
        source="friend_selfplay",
        value=0.5,
        payload={"policy_move_codes_u8": [128, 0], "policy_values": [0.9, 0.1], "root_value": 0.5},
    )
    add_label(
        conn,
        pos_id,
        label_type="search_pressure",
        source="titanium-native",
        value=-0.25,
        payload={"search_pressure": -0.25, "src": "unit"},
    )
    conn.commit()
    conn.close()

    dst_db = tmp_path / "graph_v2.db"
    sidecars = tmp_path / "sidecars"
    result = rebuild_compact_db(src_db, dst_db, sidecar_dir=sidecars)
    assert result["migration"]["canonical_labels"] >= 2
    assert sidecars.exists()
    assert any(sidecars.iterdir())

    audit = storage_audit(dst_db, vacuum_measure=False)
    assert audit["schema_kind"] == "compact-v2"
    assert "canonical_labels" in audit["tables"]
    assert "payload_refs" in audit["tables"]

    out = tmp_path / "train.bin"
    export = export_training_binary(dst_db, out_path=out, label_type_code_filter=LABEL_TYPE_TEACHER_VALUE)
    assert export["rows"] >= 1
    assert out.exists()
    assert out.stat().st_size > 0


def test_score_semantics_report_is_stable() -> None:
    report = score_semantics_report()
    assert report["eval_unit_name"] == "centitempo"
    assert report["eval_units_per_tempo"] == 100
    assert report["true_mate_score_band"]["base"] == 100_000
