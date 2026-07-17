from __future__ import annotations
import json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from diversity.eval_denylist import is_evaluation_leakage
from diversity.claustrophobia_rows import enforce_pilot_caps
from external_sources.claustrophobia.relabel_pilot_roots import score, stable_search_pair
from external_sources.claustrophobia.run_mining_pilot import merge_pilot_row
from external_sources.claustrophobia.audit_pilot import validate_pilot_row

def test_clean_openings_excluded():
    p=ROOT/"external_sources/claustrophobia/mining_openings/mining_openings_pilot_v1.json"
    d=json.loads(p.read_text())
    assert d["clean_v1_excluded"]
    assert all(x["opening_id"].startswith("mine-open-") for x in d["openings"])
    assert min(x["plies"] for x in d["openings"]) >= 2
    assert max(x["plies"] for x in d["openings"]) >= 6
    assert "clean_v1_exclusion_evidence" in d
    assert all(tuple(x["moves"]) not in {(),("e2","e8"),("e2","e8","e3","e7")} for x in d["openings"])

def test_clean_lineage_leaks():
    leaked,asset=is_evaluation_leakage(lineage_id="clean_v1")
    assert leaked and asset=="claustrophobia_clean_v1"

def test_caps():
    assert "source_game_cap_exceeded" in enforce_pilot_caps(total_pilot_rows=100,claustrophobia_rows=1,rows_for_source_game=33,rows_for_fork_lineage=1)
    assert "fork_lineage_cap_exceeded" in enforce_pilot_caps(total_pilot_rows=100,claustrophobia_rows=1,rows_for_source_game=1,rows_for_fork_lineage=129)

def test_stability_rule_accepts_only_same_move_within_margin():
    assert stable_search_pair("e4", "e4", 100, 150)
    assert not stable_search_pair("e4", "e4", 100, 151)
    assert not stable_search_pair("e4", "d4", 100, 100)
    assert stable_search_pair("e4", "e4", None, 100)

def test_score_prefers_root_score_then_root_moves():
    assert score({"rootScore": 17, "rootMoves": [{"score": 99}]}) == 17.0
    assert score({"rootMoves": [{"score": {"cp": 23}}]}) == 23.0

def test_same_move_without_scores_is_stable():
    assert stable_search_pair("e4", "e4", None, None)

def test_different_moves_are_unstable_without_scores():
    assert not stable_search_pair("e4", "d4", None, None)

def test_merge_pilot_row_preserves_play_result_and_metadata_wins():
    row = merge_pilot_row(
        {"moves": ["e4"], "actions": [{"move": "e4"}], "winner_side": "titanium"},
        {"source_game_id": "g", "winner_side": "metadata"},
    )
    assert row["moves"] == ["e4"]
    assert row["winner_side"] == "metadata"
    assert row["actions"] == [{"move": "e4"}]

def test_missing_provenance_is_invalid():
    result=validate_pilot_row({"source_game_id":"g"})
    assert result["status"]=="INVALID"
    assert result["valid"] is False
    assert "source_kind" in result["missing"]

def test_frozen_eval_never_training_eligible():
    p=ROOT/"external_sources/claustrophobia/eval_games/clean_v1/EVAL_DENYLIST_KEYS.json"
    d=json.loads(p.read_text())
    assert d["fail_closed"] and d["training_eligible"] is False
