from __future__ import annotations

from diversity.claustrophobia_rows import ClaustrophobiaDatasetKind, ClaustrophobiaDerivedRow
from diversity.book_candidates import (
    CandidateStatus,
    admit_to_live_book_allowed,
    candidate_row,
    is_clean_v1_excluded,
    validate_candidate_row,
)
from external_sources.claustrophobia.build_book_candidates import _fixture_legal, verify_row, build


def _row(**changes):
    row = candidate_row(
        prefix_moves=["a2h", "e8"],
        proposed_move="e2",
        opening_id="mine-open-test",
        claustro_checkpoint_sha256="c" * 64,
        repository_commit="r" * 40,
        titanium_weights_sha256="t" * 64,
        **changes,
    )
    return row


def test_clean_v1_exclusion_checks_id_and_moves():
    assert is_clean_v1_excluded(opening_id="open-0001")
    assert is_clean_v1_excluded(moves=["e2", "e8"])
    assert not is_clean_v1_excluded(opening_id="mine-open-0001", moves=["a2h", "e8"])


def test_live_import_is_always_blocked():
    assert admit_to_live_book_allowed(_row()) is False
    assert admit_to_live_book_allowed(_row(status=CandidateStatus.LIVE_BOOK_MERGE_PENDING.value)) is False


def test_stable_legal_candidate_stays_out_of_live_book():
    row = _row(
        status=CandidateStatus.BOOK_CANDIDATE_VERIFIED.value,
        titanium_best="e2",
        budgets={"seconds": [1.0, 4.0]},
        stability={"stable": True, "best_moves": ["e2", "e2"]},
    )
    assert validate_candidate_row(row)["valid"]
    assert row["training_eligible"] is False
    assert row["live_book_eligible"] is False
    assert not admit_to_live_book_allowed(row)


def test_unstable_status_is_valid_but_not_eligible():
    row = _row(
        status=CandidateStatus.REJECTED_UNSTABLE.value,
        budgets={"seconds": [1.0, 4.0]},
        stability={"stable": False, "best_moves": ["e2", "d2"]},
    )
    assert validate_candidate_row(row)["valid"]


def test_missing_provenance_is_invalid():
    row = _row()
    del row["provenance"]["titanium_weights_sha256"]
    result = validate_candidate_row(row)
    assert result["status"] == "INVALID"
    assert "titanium_weights_sha256" in result["missing"]


def test_offline_fixture_rejects_malformed_prefix_and_move():
    assert not _fixture_legal(["not-a-move"], "e2")
    assert not _fixture_legal(["e2"], "not-a-move")
    assert _fixture_legal([], "e2")


class _StableSession:
    def sync(self, moves):
        return True

    def go_detailed(self, seconds):
        return {"bestmove": "e2", "info": {"rootMoves": [{"move": "e2"}]}}


def test_verify_only_marks_stable_titanium_legal_row_verified():
    row = _row()
    verified = verify_row(row, _StableSession())
    assert verified["status"] == CandidateStatus.BOOK_CANDIDATE_VERIFIED.value
    assert verified["legality"]["titanium"] is True


def test_placeholder_provenance_is_invalid():
    row = _row()
    row["provenance"]["titanium_weights_sha256"] = "not-supplied"
    result = validate_candidate_row(row)
    assert result["status"] == "INVALID"


def test_repeated_move_not_rejected_by_syntax_fixture():
    assert _fixture_legal(["e2", "e8", "e2"], "e2")


def test_frozen_rows_cannot_be_book_eligible():
    try:
        ClaustrophobiaDerivedRow(
            dataset_kind=ClaustrophobiaDatasetKind.FROZEN_EVALUATION_GAMES.value,
            claustrophobia_release_tag="v1", claustrophobia_checkpoint_sha256="c",
            repository_commit="r", source_game_id="g", opening_seed_id="o",
            claustrophobia_chosen_move="e2", titanium_move="e2", final_game_outcome="",
            relabeling_status="", evaluation_eligible=True, training_eligible=False,
            book_eligible=True,
        )
    except ValueError as exc:
        assert "book-eligible" in str(exc)
    else:
        raise AssertionError("frozen evaluation row accepted as book eligible")


def test_builder_summary_live_admission_is_false():
    from pathlib import Path
    from types import SimpleNamespace

    args = SimpleNamespace(
        openings=Path("training/external_sources/claustrophobia/mining_openings/mining_openings_pilot_v1.json"),
        results=Path("training/external_sources/claustrophobia/no-such-results.jsonl"),
        claustro_checkpoint=None, weights=None, titanium_bin=None, verify=False, max=0,
    )
    _, summary = build(args)
    assert summary["admit_to_live_book_allowed"] is False
    assert type(summary["admit_to_live_book_allowed"]) is bool
