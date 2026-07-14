"""Tests for DIVERSITY_SPEC_V1 collapse certificate."""
from __future__ import annotations

from diversity_spec import (
    MIN_N_EFF_2,
    MIN_N_EFF_4,
    collapse_certificate_from_prefixes,
    effective_support,
)


def test_effective_support_uniform():
    from collections import Counter

    counts = Counter({"a": 10, "b": 10, "c": 10, "d": 10})
    assert effective_support(counts) == 4.0


def test_certificate_blocks_single_trunk():
    prefixes = [("e2", "e8", "e3", "e7")] * 100
    cert = collapse_certificate_from_prefixes(prefixes)
    assert not cert.passed
    assert cert.max_two_ply_mass == 1.0
    assert cert.n_eff_2 == 1.0
    assert cert.n_eff_2 < MIN_N_EFF_2
    assert "N_eff(2)" in (cert.block_reason or "")


def test_certificate_two_ply_ceiling_from_standard_start():
    """Standard Quoridor has at most 9 central two-ply keys — below N_eff(2)=16."""
    pairs = [(w, b) for w in ("d2", "e2", "f2") for b in ("d8", "e8", "f8")]
    prefixes = [(w, b, "e3", "e7") for w, b in pairs for _ in range(8)]
    cert = collapse_certificate_from_prefixes(prefixes)
    assert cert.n_eff_2 <= 9.01
    assert not cert.passed
    assert "N_eff(2)" in (cert.block_reason or "")


def test_certificate_passes_rich_four_ply_panel():
    thirds = ("e3", "d2", "f2", "c2", "g2", "e4", "d3", "f3")
    fourths = ("e7", "d7", "f7", "c7", "g7", "e6", "d6", "f6")
    prefixes = [("e2", "e8", t, f) for t in thirds for f in fourths]
    cert = collapse_certificate_from_prefixes(prefixes)
    assert cert.n_eff_4 >= MIN_N_EFF_4
    assert cert.max_two_ply_mass == 1.0  # single two-ply trunk; four-ply diversity is separate
    assert not cert.passed  # N_eff(2) and two-ply mass floors still fail until seeded opens
