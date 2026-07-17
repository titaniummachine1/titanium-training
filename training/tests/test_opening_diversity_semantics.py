"""Fixture-only tests for opening diversity / seed / prefix / dedup / label-cache semantics."""
from __future__ import annotations

import sys
from pathlib import Path

_TRAINING = Path(__file__).resolve().parents[1]
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

from diversity.augmentation_policy import PREFERRED_AUGMENTATION
from diversity.canonical import CanonicalStateRow, deduplicate_finalized_rows
from diversity.claustrophobia_rows import (
    ClaustrophobiaDatasetKind,
    ClaustrophobiaDerivedRow,
    enforce_pilot_caps,
)
from diversity.duplicate_layers import FinalCorpusDedup, GenerationDedup, LabelDedup
from diversity.label_cache_compat import LabelCacheLookup, LabelCompatKey, lookup_label_cache
from diversity.prefix_metrics import (
    PREFIX_METRIC_VERSION_V2,
    fixture_prefix_context,
    prefix2_key,
    prefix2_key_v2,
    prefix4_key_v2,
    prefix_key_from_state_transitions,
    standard_start_state,
)
from diversity.seed_bank_schema import (
    ALLOWED_ORIGIN_CATEGORIES,
    EvaluationLeakageStatus,
    LegalityValidationStatus,
    SeedActiveState,
    SeedRecord,
    validate_seed_for_selection,
)
from diversity.seed_selection import SeedSelectionConfig, SeedUsageTracker, select_seeds_for_batch


def _state(*, pawns: str, stm: int = 0, stocks: str = "10,10", hw: str = "", vw: str = "") -> CanonicalStateRow:
    return CanonicalStateRow(
        pawn_positions=pawns,
        horizontal_walls=hw,
        vertical_walls=vw,
        wall_stocks=stocks,
        side_to_move=stm,
    )


def _fixture_seed(
    seed_id: str,
    *,
    family: str = "fam-a",
    origin: str = "synthetic_fixture",
    phase: str = "opening",
    tension: str = "low",
    stm: int = 0,
    pawns: str = "e2,e8",
    leakage: str = "clear",
    active: str = "active",
    legal: str = "valid",
) -> SeedRecord:
    st = _state(pawns=pawns, stm=stm)
    return SeedRecord(
        seed_id=seed_id,
        seed_family_id=family,
        game_state={"pawn_positions": pawns, "side_to_move": stm, "wall_stocks": "10,10"},
        reflection_canonical_state_key=st.canonical_key(),
        side_to_move=stm,
        pawn_locations=pawns,
        placed_horizontal_walls="",
        placed_vertical_walls="",
        wall_stocks="10,10",
        origin_source=origin,
        origin_game_id=f"game-{seed_id}",
        origin_lineage_id=f"lin-{family}",
        generation_method="fixture",
        source_engine_opponent_ids=("fixture",),
        phase=phase,
        tension_class=tension,
        engine_semantic_hash="sem-fixture",
        canonical_state_version="canonical-state-v1",
        prefix_metric_version=PREFIX_METRIC_VERSION_V2,
        evaluation_leakage_status=leakage,
        legality_validation_status=legal,
        creation_timestamp="2026-07-16T00:00:00Z",
        active_retired_state=active,
    )


def test_unknown_origin_rejected():
    try:
        _fixture_seed("x", origin="not_a_real_origin")
        assert False, "expected ValueError"
    except ValueError as e:
        assert "unknown origin" in str(e)


def test_allowed_origins_cover_required_set():
    required = {
        "standard_start",
        "champion_generated_centroid",
        "historical_champion_centroid",
        "Claustrophobia_disagreement",
        "Claustrophobia_loss_root",
        "paired_fork",
        "solver_seam",
        "exact_anchor",
        "synthetic_fixture",
    }
    assert required <= ALLOWED_ORIGIN_CATEGORIES


def test_deterministic_seed_selection():
    bank = [
        _fixture_seed(f"s{i}", family=f"f{i%3}", pawns=f"e{2+(i%3)},e8", stm=i % 2)
        for i in range(12)
    ]
    cfg = SeedSelectionConfig(planner_seed=7, cooldown_batches=0, max_per_family_per_batch=3)
    a, _ = select_seeds_for_batch(bank, batch_size=5, cfg=cfg, tracker=SeedUsageTracker())
    b, _ = select_seeds_for_batch(bank, batch_size=5, cfg=cfg, tracker=SeedUsageTracker())
    assert [s.seed_id for s in a] == [s.seed_id for s in b]


def test_no_duplicate_seed_canonical_in_one_batch():
    # Two seed_ids, identical board -> same canonical key
    bank = [
        _fixture_seed("a", family="f1", pawns="e2,e8"),
        _fixture_seed("b", family="f2", pawns="e2,e8"),
        _fixture_seed("c", family="f3", pawns="d2,e8"),
    ]
    selected, decisions = select_seeds_for_batch(
        bank, batch_size=5, cfg=SeedSelectionConfig(cooldown_batches=0), tracker=SeedUsageTracker()
    )
    keys = [s.reflection_canonical_state_key for s in selected]
    assert len(keys) == len(set(keys))
    assert any(d.reason == "reject:duplicate_canonical_in_batch" for d in decisions)


def test_seed_cooldown():
    bank = [_fixture_seed("only", pawns="e2,e8")]
    tracker = SeedUsageTracker()
    cfg = SeedSelectionConfig(cooldown_batches=2, planner_seed=1)
    s1, _ = select_seeds_for_batch(bank, batch_size=1, cfg=cfg, tracker=tracker)
    assert len(s1) == 1
    s2, d2 = select_seeds_for_batch(bank, batch_size=1, cfg=cfg, tracker=tracker)
    assert s2 == []
    assert any(d.reason == "reject:cooldown" for d in d2)


def test_family_caps():
    bank = [_fixture_seed(f"s{i}", family="same", pawns=f"e{2+i},e8") for i in range(5)]
    selected, decisions = select_seeds_for_batch(
        bank,
        batch_size=5,
        cfg=SeedSelectionConfig(max_per_family_per_batch=2, cooldown_batches=0),
        tracker=SeedUsageTracker(),
    )
    assert len(selected) == 2
    assert sum(1 for d in decisions if d.reason == "reject:family_cap") >= 1


def test_phase_tension_and_stm_balancing_priority():
    bank = [
        _fixture_seed("common", family="f0", phase="opening", tension="low", stm=0, pawns="e2,e8"),
        _fixture_seed("rare", family="f1", phase="endgame", tension="high", stm=1, pawns="d2,e8"),
    ]
    deficit = {("endgame", "high"): 10.0}
    stm_def = {1: 10.0}
    selected, _ = select_seeds_for_batch(
        bank,
        batch_size=1,
        cfg=SeedSelectionConfig(cooldown_batches=0, planner_seed=0),
        tracker=SeedUsageTracker(),
        phase_tension_deficit=deficit,
        stm_deficit=stm_def,
    )
    assert selected[0].seed_id == "rare"


def test_evaluation_leakage_and_retired_rejected():
    leak = _fixture_seed("leak", leakage=EvaluationLeakageStatus.EVAL_ONLY.value)
    retired = _fixture_seed("ret", active=SeedActiveState.RETIRED.value, pawns="d2,e8")
    assert "evaluation_leakage" in validate_seed_for_selection(leak)
    assert "retired" in validate_seed_for_selection(retired)
    selected, _ = select_seeds_for_batch(
        [leak, retired],
        batch_size=2,
        cfg=SeedSelectionConfig(cooldown_batches=0),
        tracker=SeedUsageTracker(),
    )
    assert selected == []


def test_reflection_canonical_seed_identity_and_wall_stocks_stm():
    a = _state(pawns="e2,e8", stm=0, stocks="10,10")
    b = _state(pawns="e2,e8", stm=1, stocks="10,10")
    c = _state(pawns="e2,e8", stm=0, stocks="9,10")
    assert a.canonical_key() != b.canonical_key()
    assert a.canonical_key() != c.canonical_key()


def test_different_seeds_identical_move_strings_different_prefix_when_start_differs():
    start_a = standard_start_state()
    start_b = _state(pawns="d2,e8", stm=0)  # different seed start
    moves = ("e3", "e7")
    ctx_a = fixture_prefix_context(root_seed_id="seed-a", start_state=start_a)
    ctx_b = fixture_prefix_context(root_seed_id="seed-b", start_state=start_b)
    ka = prefix2_key(ctx_a, moves)
    kb = prefix2_key(ctx_b, moves)
    assert ka is not None and kb is not None
    assert ka != kb


def test_prefix2_prefix4_v2_state_transition_keys():
    start = standard_start_state()
    p1 = _state(pawns="e3,e8", stm=1)
    p2 = _state(pawns="e3,e7", stm=0)
    p3 = _state(pawns="e4,e7", stm=1)
    p4 = _state(pawns="e4,e6", stm=0)
    k2 = prefix2_key_v2(start, p1, p2)
    k4 = prefix4_key_v2(start, p1, p2, p3, p4)
    assert k2 and k4 and k2 != k4
    # Missing transition => INVALID
    assert prefix_key_from_state_transitions(
        prefix_metric_version=PREFIX_METRIC_VERSION_V2,
        start_state=start,
        states_after_plies=(),
    ) is None


def test_compatible_label_reuse_and_semantic_change():
    base = LabelCompatKey(
        canonical_state_key="abc",
        engine_semantic_hash="eng1",
        search_configuration_hash="search1",
        evaluation_semantics_version="eval1",
        score_band_version="band1",
        oracle_semantics_version="oracle1",
        move_encoding_version="move1",
        label_configuration_hash="label1",
        exact_label_kind="search",
        side_to_move=0,
    )
    assert lookup_label_cache(base, base) == LabelCacheLookup.HIT_COMPATIBLE
    assert lookup_label_cache(base, None) == LabelCacheLookup.MISS_NEW_STATE
    changed_search = LabelCompatKey(**{**base.__dict__, "search_configuration_hash": "search2"})
    assert lookup_label_cache(base, changed_search) == LabelCacheLookup.MISS_SEARCH_CONFIG_CHANGED
    changed_oracle = LabelCompatKey(**{**base.__dict__, "oracle_semantics_version": "oracle2"})
    assert lookup_label_cache(base, changed_oracle) == LabelCacheLookup.MISS_ORACLE_CHANGED
    changed_eng = LabelCompatKey(**{**base.__dict__, "engine_semantic_hash": "eng2"})
    assert lookup_label_cache(base, changed_eng) == LabelCacheLookup.MISS_SEMANTICS_CHANGED


def test_final_corpus_zero_duplicates_and_generation_layers():
    rows = [
        _state(pawns="e2,e8"),
        _state(pawns="e2,e8"),
        _state(pawns="d2,e8"),
    ]
    unique, dupes = deduplicate_finalized_rows(rows)
    assert dupes == 1 and len(unique) == 2
    final = FinalCorpusDedup()
    assert final.accept(unique[0].canonical_key())
    assert not final.accept(unique[0].canonical_key())
    final.assert_zero_duplicates([r.canonical_key() for r in unique])

    gen = GenerationDedup()
    assert gen.observe("k1")
    assert not gen.observe("k1")
    gen.allow_controlled_revisit = True
    assert gen.observe("k1", force_revisit=True)

    lab = LabelDedup()
    assert not lab.already_labeled_compatible("fp1")
    lab.remember("fp1")
    assert lab.already_labeled_compatible("fp1")


def test_augmentation_policy_is_dynamic_only():
    assert PREFERRED_AUGMENTATION.dynamic_during_batching is True
    assert PREFERRED_AUGMENTATION.materialize_reflected_rows is False


def test_eval_game_excluded_from_training_claustrophobia_row():
    try:
        ClaustrophobiaDerivedRow(
            dataset_kind=ClaustrophobiaDatasetKind.FROZEN_EVALUATION_GAMES.value,
            claustrophobia_release_tag="v1.0.0",
            claustrophobia_checkpoint_sha256="abc",
            repository_commit="deadbeef",
            source_game_id="g1",
            opening_seed_id="s1",
            claustrophobia_chosen_move="e2",
            titanium_move="e2",
            final_game_outcome="0",
            relabeling_status="none",
            evaluation_eligible=True,
            training_eligible=True,
        )
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_claustrophobia_source_game_and_lineage_caps():
    assert enforce_pilot_caps(
        total_pilot_rows=100,
        claustrophobia_rows=4,
        rows_for_source_game=33,
        rows_for_fork_lineage=10,
    ) == ["source_game_cap_exceeded"]
    assert enforce_pilot_caps(
        total_pilot_rows=100,
        claustrophobia_rows=4,
        rows_for_source_game=10,
        rows_for_fork_lineage=129,
    ) == ["fork_lineage_cap_exceeded"]
    assert (
        enforce_pilot_caps(
            total_pilot_rows=100,
            claustrophobia_rows=5,
            rows_for_source_game=32,
            rows_for_fork_lineage=128,
        )
        == []
    )
    assert "pilot_cap_fraction_exceeded" in enforce_pilot_caps(
        total_pilot_rows=100,
        claustrophobia_rows=6,
        rows_for_source_game=1,
        rows_for_fork_lineage=1,
    )
