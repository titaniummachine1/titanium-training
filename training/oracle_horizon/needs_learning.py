"""Compare shallow search with oracle references to find teachable positions."""
from __future__ import annotations

import math
from typing import Any


def needs_learning(shallow_result: dict, oracle_ref: dict) -> tuple[bool, list[str]]:
    """Return deterministic reasons; absent evidence is not silently a pass."""
    reasons: list[str] = []
    shallow, oracle = shallow_result or {}, oracle_ref or {}
    swdl = shallow.get("wdl", shallow.get("shallow_wdl"))
    owdl = oracle.get("wdl", oracle.get("oracle_wdl_stm", oracle.get("oracle_wdl")))
    if swdl is not None and owdl is not None and swdl != owdl:
        reasons.append("wrong_wdl_sign")
    if shallow.get("forced_wdl") is False and oracle.get("forced_wdl") is True:
        reasons.append("missed_forced_w_l")
    smove = shallow.get("best_move", shallow.get("move", shallow.get("shallow_best_move")))
    omove = oracle.get("best_move", oracle.get("oracle_best_move"))
    if smove is not None and omove is not None and smove != omove:
        reasons.append("wrong_move")
    if oracle.get("only_defense") and smove != omove:
        reasons.append("missed_only_defense")
    ratio = shallow.get("nodes_ratio", shallow.get("searched_nodes_ratio"))
    if ratio is not None and float(ratio) >= float(oracle.get("nodes_ratio_threshold", 16.0)):
        reasons.append("nodes_ratio")
    value = shallow.get("eval", shallow.get("score"))
    decisive = oracle.get("decisive", owdl in {"W", "L", 1, -1})
    if decisive and value is not None and math.isfinite(float(value)) and abs(float(value)) <= float(
        oracle.get("near_zero_threshold", 25.0)
    ):
        reasons.append("near_zero_eval_on_decisive")
    if shallow.get("move_flip") or shallow.get("best_move_flipped"):
        reasons.append("move_flip")
    if shallow.get("missed_forced_win") or shallow.get("missed_forced_loss"):
        reasons.append("missed_forced_w_l")
    return bool(reasons), list(dict.fromkeys(reasons))
