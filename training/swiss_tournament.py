"""Random tournament pairing for overnight Elo estimation.

Each batch runs PARALLEL_MATCHUPS different matchups concurrently (1 game each).
Pairings are chosen at random, weighted toward underplayed matchups so every
engine eventually meets every other eligible opponent.

Remote opponents unavailable (e.g. Ishtar down) are skipped.
Global ladder propagates from anchor ace-v13-ti-pure@5s = 1400.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

import json
from pathlib import Path

from manifest import (
    ANCHOR_ENTITY,
    ANCHOR_RATING,
    entity_label,
    load_manifest,
    matchup_key,
)

ROOT = Path(__file__).resolve().parent.parent
REMOTE_TIMING = ROOT / "training" / "data" / "remote_timing.json"

PARALLEL_MATCHUPS = 4
GAMES_PER_MATCHUP = 1

Kind = Literal["local", "remote"]


@dataclass(frozen=True)
class Pairing:
    kind: Kind
    engine_a: str
    engine_b: str
    tc_a: str
    tc_b: str
    label: str
    priority: int = 0
    target_games: int = 0
    our_time: float = 5.0
    ponder_time: float | None = None

    def manifest_key(self) -> str:
        return matchup_key(self.engine_a, self.engine_b, self.tc_a, self.tc_b)

    def games_file_name(self, batch_id: str = "") -> str:
        safe = self.label.replace("/", "-").replace(" ", "_")
        if batch_id:
            return f"{safe}-{batch_id}.games"
        return f"{safe}.games"

    def source_tag(self, batch_id: str = "") -> str:
        base = f"random-{self.label}"
        return f"{base}-{batch_id}" if batch_id else base


def remote_availability() -> dict[str, bool]:
    """From last wiring probe; missing file → assume available."""
    try:
        doc = json.loads(REMOTE_TIMING.read_text(encoding="utf-8"))
        avail = doc.get("availability") or {}
        return {k: bool(v) for k, v in avail.items()}
    except (OSError, json.JSONDecodeError):
        return {}


def all_pairings() -> list[Pairing]:
    """Every eligible engine-vs-engine matchup (local @5s + v15 vs remote presets)."""
    p: list[Pairing] = []

    def local(a: str, b: str) -> None:
        p.append(Pairing(
            kind="local", engine_a=a, engine_b=b,
            tc_a="5s", tc_b="5s",
            label=f"{a}-vs-{b}-5s",
        ))

    local("titanium-v15", "ace-v13-ti-pure")
    # NOTE: bare "ace-v13" removed — legacy engine (no Titanium movegen), uninformative
    # matchup that wastes CPU slots. History kept in manifest; just no longer scheduled.
    # NOTE: bare "titanium" also excluded — legacy MCTS (GameSearchSession), loses 0-98.

    def remote(opp: str, opp_time: str) -> None:
        p.append(Pairing(
            kind="remote",
            engine_a="titanium-v15",
            engine_b=opp,
            tc_a="5s",
            tc_b=opp_time,
            label=f"v15-vs-{opp}-{opp_time}",
            our_time=0, ponder_time=0,
        ))

    remote("ka", "intuition")
    remote("ka", "short")
    remote("ka", "medium")
    remote("ka", "long")

    remote("ishtar", "intuition")
    remote("ishtar", "short")
    remote("ishtar", "medium")
    remote("ishtar", "long")

    return p


def eligible_pairings(manifest: dict | None = None) -> list[Pairing]:
    """Available pairings (skips unavailable remotes)."""
    avail = remote_availability()
    out: list[Pairing] = []
    for pairing in all_pairings():
        if pairing.kind == "remote" and avail.get(pairing.engine_b) is False:
            continue
        out.append(pairing)
    return out or all_pairings()


def _games_played(manifest: dict, pairing: Pairing) -> int:
    m = manifest.get("matchups", {}).get(pairing.manifest_key(), {})
    return m.get("games_played", m.get("a_wins", 0) + m.get("b_wins", 0))


MAX_REMOTE_PARALLEL = 4  # Allow concurrent remote games against Ka/Ishtar (connections are isolated)


def pick_one_pairing(manifest: dict | None = None, *, allow_remote: bool = True) -> Pairing | None:
    """Pick one random matchup; underplayed pairs are more likely."""
    manifest = manifest or load_manifest()
    pool = eligible_pairings(manifest)
    if not allow_remote:
        pool = [p for p in pool if p.kind != "remote"]
    if not pool:
        return None
    weights = [1.0 / (_games_played(manifest, p) + 1) for p in pool]
    return random.choices(pool, weights=weights, k=1)[0]


def pairing_game_entry(pairing: Pairing, game_id: str, tournament_dir: Path) -> dict:
    """JSON payload for one game worker (local or remote)."""
    tournament_dir.mkdir(parents=True, exist_ok=True)
    return {
        "kind": pairing.kind,
        "label": pairing.label,
        "engine_a": pairing.engine_a,
        "engine_b": pairing.engine_b,
        "tc_a": pairing.tc_a,
        "tc_b": pairing.tc_b,
        "games_file": str(tournament_dir / pairing.games_file_name(game_id)),
        "source_tag": pairing.source_tag(game_id),
        "game_id": game_id,
    }


def pick_random_batch(manifest: dict | None = None, n: int = PARALLEL_MATCHUPS) -> list[Pairing]:
    """Pick n distinct matchups at random; underplayed pairs are more likely."""
    manifest = manifest or load_manifest()
    pool = eligible_pairings(manifest)
    if not pool:
        raise RuntimeError("no pairings available")
    n = min(n, len(pool))
    remaining = list(pool)
    weights = [1.0 / (_games_played(manifest, p) + 1) for p in remaining]
    chosen: list[Pairing] = []
    seen: set[str] = set()
    remotes_used = 0
    while len(chosen) < n and remaining:
        candidates: list[int] = []
        candidate_weights: list[float] = []
        for i, p in enumerate(remaining):
            key = p.manifest_key()
            if key in seen:
                continue
            if p.kind == "remote" and remotes_used >= MAX_REMOTE_PARALLEL:
                continue
            candidates.append(i)
            candidate_weights.append(weights[i])
        if not candidates:
            break
        pick = random.choices(range(len(candidates)), weights=candidate_weights, k=1)[0]
        idx = candidates[pick]
        pairing = remaining.pop(idx)
        weights.pop(idx)
        key = pairing.manifest_key()
        assert key not in seen, f"duplicate pairing in batch: {key}"
        seen.add(key)
        if pairing.kind == "remote":
            remotes_used += 1
        chosen.append(pairing)
    return chosen


def _tournament_state(manifest: dict) -> dict:
    return manifest.setdefault("tournament", {})


def record_batch(manifest: dict, pairings: list[Pairing], elapsed: float) -> dict:
    """Update tournament metadata after a parallel batch."""
    t = _tournament_state(manifest)
    t["mode"] = "random"
    t["batch"] = int(t.get("batch", 0)) + 1
    t["last_batch"] = [p.label for p in pairings]
    t["last_elapsed_min"] = round(elapsed / 60, 1)
    t["parallel"] = len(pairings)
    return manifest


def _entity_rating(manifest: dict, engine: str, tc: str) -> float | None:
    ent = entity_label(engine, tc)
    gr = manifest.get("global_ratings", {})
    if ent in gr:
        return float(gr[ent]["rating"])
    base = engine.split("@")[0]
    for k, v in gr.items():
        if k.split("@")[0] == base:
            return float(v["rating"])
    if ent == ANCHOR_ENTITY:
        return ANCHOR_RATING
    return None


def list_pairings(manifest: dict | None = None) -> list[tuple[Pairing, int, str]]:
    manifest = manifest or load_manifest()
    rows: list[tuple[Pairing, int, str]] = []
    for pairing in eligible_pairings(manifest):
        rows.append((pairing, _games_played(manifest, pairing), ""))
    rows.sort(key=lambda x: x[1])
    return rows
