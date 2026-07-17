from db_import import GAMES_SCHEMA, LABELS_SCHEMA, open_db, write_batch


def _fake_eval_prefixes(prefixes, chunk_size, workers):
    records = {}
    for prefix in prefixes:
        ply = 0 if not prefix else len(prefix.split())
        records[prefix] = {
            "turn": ply % 2,
            "ply": ply,
            "moves": prefix.split() if prefix else [],
        }
    return records


def test_write_batch_skips_duplicate_full_game_lines(tmp_path, monkeypatch):
    monkeypatch.setattr("db_import.eval_prefixes_parallel", _fake_eval_prefixes)
    games_db = open_db(tmp_path / "games.db", GAMES_SCHEMA)
    labels_db = open_db(tmp_path / "labels.db", LABELS_SCHEMA)

    moves = ["e2", "e8", "e3", "e7"]
    first = ("game-a", moves, 1, None, "test")
    duplicate = ("game-b", list(moves), -1, None, "test")

    assert write_batch(games_db, labels_db, [first, duplicate], 128, 1) == (1, 4, 4)
    assert games_db.execute("SELECT COUNT(*) FROM games").fetchone()[0] == 1
    assert games_db.execute("SELECT COUNT(*) FROM game_line_hashes").fetchone()[0] == 1
    assert labels_db.execute("SELECT SUM(n_samples) FROM labels").fetchone()[0] == 4

    assert write_batch(games_db, labels_db, [duplicate], 128, 1) == (0, 0, 0)
    assert games_db.execute("SELECT COUNT(*) FROM games").fetchone()[0] == 1
    assert labels_db.execute("SELECT SUM(n_samples) FROM labels").fetchone()[0] == 4

    games_db.close()
    labels_db.close()
