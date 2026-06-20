"""Pool startup housekeeping — prune stale manifest rows, pack DB moves.

Graceful: migrate in place, log what was removed, never delete game rows.
Loud fail: reserved for engine/parity/schema (see supervise.py).
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Remote presets no longer in the pool (Ka intuition is active again).
DEPRECATED_REMOTE_TC = frozenset({"immediate"})


def _deprecated_matchup_entry(m: dict) -> bool:
    ea = m.get("a_engine", "")
    eb = m.get("b_engine", "")
    tc_b = (m.get("tc_b") or "5s").strip()
    if eb in ("ishtar",) or ea in ("ishtar", "titanium", "titanium-cert"):
        return True
    if eb in ("ka", "ishtar") and tc_b in DEPRECATED_REMOTE_TC:
        return True
    # Ace-only games — not in current pool
    if {ea, eb} <= {"ace-v13", "ace-v13-ti-pure"}:
        return True
    return False


def run_pool_housekeeping(*, reset_pool_counter: bool = True) -> list[str]:
    """Prune manifest matchups outside current pool; backfill compact moves_bin."""
    from tools.datagen.datagen import DB_PATH, backfill_moves_bin, compact_moves_text, count_pool_games
    from tools.maintenance.manifest import load_manifest, matchup_key, save_manifest
    from tools.operations.swiss_tournament import all_pairings

    allowed = {p.manifest_key() for p in all_pairings()}
    manifest = load_manifest()
    msgs: list[str] = []

    removed: list[str] = []
    for key in list(manifest.get("matchups", {})):
        m = manifest["matchups"][key]
        canon = matchup_key(
            m.get("a_engine", ""),
            m.get("b_engine", ""),
            m.get("tc_a", "5s"),
            m.get("tc_b", "5s"),
        )
        if canon not in allowed or _deprecated_matchup_entry(m):
            del manifest["matchups"][key]
            removed.append(key)

    if removed:
        msgs.append(f"pruned {len(removed)} stale manifest matchup(s)")
        if reset_pool_counter:
            t = manifest.setdefault("tournament", {})
            t["mode"] = "random-pool"
            t["games"] = 0
        save_manifest(manifest)

    try:
        packed = backfill_moves_bin(DB_PATH)
        if packed:
            msgs.append(f"packed moves_bin for {packed} row(s)")
        compacted = compact_moves_text(DB_PATH)
        if compacted:
            msgs.append(f"cleared redundant moves TEXT for {compacted} row(s)")
    except Exception as e:
        msgs.append(f"moves compaction error: {e}")

    pool_n = count_pool_games(DB_PATH)
    total = manifest.get("matchups", {})
    _ = total  # manifest saved above if needed
    msgs.append(f"DB: {pool_n} pool-tagged games (current schema)")
    return msgs


if __name__ == "__main__":
    for line in run_pool_housekeeping(reset_pool_counter=False):
        print(line)
