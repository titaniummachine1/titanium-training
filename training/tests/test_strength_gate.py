from strength_gate import DEFAULT_OPENINGS, candidate_score, read_openings, score_summary


def test_default_gate_suite_has_one_distinct_opening_per_100_game_pair():
    assert len(DEFAULT_OPENINGS) >= 100
    assert len(DEFAULT_OPENINGS) == len(set(DEFAULT_OPENINGS))
    assert {len(opening) for opening in DEFAULT_OPENINGS} == {14}


def test_candidate_score_accounts_for_colour_and_draws():
    assert candidate_score(1, True) == 1.0
    assert candidate_score(-1, True) == 0.0
    assert candidate_score(-1, False) == 1.0
    assert candidate_score(0, False) == 0.5


def test_score_summary_handles_empty_and_balanced_samples():
    assert score_summary([])["games"] == 0
    summary = score_summary([1.0, 0.0, 0.5, 0.5])
    assert summary["games"] == 4
    assert summary["score"] == 2.0
    assert summary["rate"] == 0.5
    assert summary["elo"] == 0.0


def test_read_openings_accepts_plain_lines_and_book_summaries(tmp_path):
    path = tmp_path / "openings.txt"
    path.write_text("# comment\ne2 e8 e3 e7\nply 2 line=e2 e8\n", encoding="utf-8")
    assert read_openings(path) == (("e2", "e8", "e3", "e7"), ("e2", "e8"))
