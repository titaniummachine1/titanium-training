import pytest

from oracle_horizon.build_continuation_manifest import split_rows, audit
from label_perspective import stm_to_target_prob


def row(i, klass="EXACT_ORACLE"):
    return {
        "packed_state_hex": f"{i:04x}", "band": i % 3, "label_class": klass,
        "book_move_used": False, "evaluation_only": False,
        "weights_sha256": "869ad228cfea8bb8964d98d05d6cf5e67a21b27661a36259a3976f60d486be56",
        "oracle_wdl": "W",
    }


def test_holdout_split_is_deterministic_and_disjoint():
    rows = [row(i) for i in range(60)]
    h1, t1 = split_rows(rows)
    h2, t2 = split_rows(rows)
    assert [r["packed_state_hex"] for r in h1] == [r["packed_state_hex"] for r in h2]
    assert {r["packed_state_hex"] for r in h1}.isdisjoint(r["packed_state_hex"] for r in t1)
    assert len(h1) == 9 and len(t1) == 51


def test_search_only_rejected():
    assert audit([row(1, "SEARCH_ONLY")])["status"] == "FAIL"


def test_wdl_conversion():
    assert stm_to_target_prob(1.0) == 1.0
    assert stm_to_target_prob(0.0) == 0.5
    assert stm_to_target_prob(-1.0) == 0.0


def test_oracle_fraction_guard():
    from streaming_db_loader import DbTrainingIterableDataset
    with pytest.raises(ValueError, match="oracle fraction"):
        DbTrainingIterableDataset("missing.db", ["oracle:a"] * 11 + ["json:x"] * 89,
                                  oracle_jsonl="missing.jsonl", oracle_ids=["oracle:a"] * 11)
