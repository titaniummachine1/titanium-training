from __future__ import annotations

from opening_prefix_index import (
    OpeningPrefixIndex,
    canonical_move_prefix,
    mirror_move_alg,
    prefix_hash,
    update_metrics,
    OpeningMetricsSnapshot,
)


def test_mirror_symmetry_canonicalizes_equivalent_openings():
    a = ["e2", "e8", "d2", "f8"]
    b = ["e2", "e8", "f2", "d8"]
    assert canonical_move_prefix(a) == canonical_move_prefix(b)
    assert prefix_hash(a) == prefix_hash(b)


def test_mirror_move_alg():
    assert mirror_move_alg("a3h") == "i3h"
    assert mirror_move_alg("e2") == "e2"


def test_prefix_index_register_and_lookup(tmp_path):
    db = tmp_path / "prefix.db"
    idx = OpeningPrefixIndex(db)
    try:
        moves = ["e2", "e8", "e3", "e7"]
        assert not idx.is_known(moves)
        novel = idx.register_game(moves, 1, source="test", max_ply=16)
        assert len(novel) == 4
        assert idx.is_known(moves)
        assert idx.occurrence_count(moves) == 1
        idx.register_game(moves, -1, source="test2", max_ply=16)
        assert idx.occurrence_count(moves) == 2
        rec = idx.lookup(moves)
        assert rec is not None
        assert rec.p0_wins == 1
        assert rec.p1_wins == 1
    finally:
        idx.close()


def test_metrics_snapshot():
    snap = OpeningMetricsSnapshot()
    update_metrics(
        snap,
        {
            "explored_moves": 2,
            "novel_prefix": True,
            "novel_exit_ply": 7,
            "mixed": False,
            "current_won": True,
        },
    )
    d = snap.to_dict()
    assert d["games_total"] == 1
    assert d["pct_novel_prefix"] == 100.0
    assert d["median_novel_exit_ply"] == 7
