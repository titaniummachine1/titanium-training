"""Short, fixed-width labels for pool progress dock + pairing payloads."""

from __future__ import annotations

from tools.maintenance.manifest import ANCHOR_ENGINE, CURRENT_ENGINE, engine_display_short


def pairing_display_label(pairing) -> str:
    """Compact slot label (fits progress dock, no truncation garbage)."""
    tc_a = pairing.tc_a or "5s"
    if pairing.kind == "remote" and pairing.engine_b == "ka":
        if pairing.tc_b == "adaptive":
            visits = getattr(pairing, "opponent_visits", None)
            return f"v15@5s vs Ka@{visits or '?'}v"
        ka = "Ka-imm" if pairing.tc_b == "intuition" else f"Ka-{pairing.tc_b}"
        return f"v15@5s vs {ka}"
    if pairing.kind == "remote" and pairing.engine_b == "zero":
        visits = getattr(pairing, "opponent_visits", None)
        return f"v15@5s vs zero@{visits or '?'}v"
    if pairing.engine_a == pairing.engine_b == CURRENT_ENGINE:
        return f"v15 self@{tc_a}"
    if pairing.engine_b == ANCHOR_ENGINE and tc_a == "10s":
        return f"v15@10s vs ti-pure@10s"
    if pairing.engine_b == "ace-v13":
        return f"v15@5s vs JS-v13"
    tc_b = pairing.tc_b or tc_a
    b = engine_display_short(pairing.engine_b, tc_b)
    a = engine_display_short(pairing.engine_a, tc_a)
    return f"{a} vs {b}"
