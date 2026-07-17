from streaming_epoch_validation import paired_promotion_evidence


def test_paired_promotion_evidence_rejects_coin_flip_results():
    # Ten colour-swapped pairs split exactly evenly.  A raw 50% game score
    # must never be mistaken for evidence that a new net improved.
    evidence = paired_promotion_evidence([1.0] * 10 + [0.0] * 10)

    assert evidence["pair_score"] == 0.5
    assert evidence["decisive_pairs"] == 20
    assert evidence["sign_test_p_value"] > evidence["sign_test_alpha"]
    assert evidence["passed"] is False


def test_paired_promotion_evidence_accepts_a_clear_pairwise_win():
    evidence = paired_promotion_evidence([1.0] * 20)

    assert evidence["pair_score"] == 1.0
    assert evidence["decisive_pairs"] == 20
    assert evidence["sign_test_p_value"] < 0.05
    assert evidence["passed"] is True


def test_paired_promotion_evidence_requires_decisive_pairs():
    # Even a perfect score from too few pairs has insufficient evidence.
    evidence = paired_promotion_evidence([1.0] * 19)

    assert evidence["sign_test_p_value"] < 0.05
    assert evidence["passed"] is False
