"""Random tournament pairing for overnight Elo estimation.

Each batch runs up to POOL_SLOTS_MAX matchups concurrently (1 game each).

Pool split (7 slots — reserved, deterministic):
  4× Ka           — intuition + short + medium + long (train: search labels)
  1× ti-pure@10s  — anchor adversary @ 10s think (train)
  1× v15 self@10s — self-play @ 10s (train)
  1× frozen@5s    — A/B yardstick, no train

JS-v13 bench stays in manifest for manual runs; not auto-picked in 7-slot layout.

Global ladder anchor: ace-v13-ti-pure@5s = 1200.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Literal

import json
from pathlib import Path

from tools.maintenance.manifest import (
    ANCHOR_ENTITY,
    ANCHOR_RATING,
    CURRENT_ENGINE,
    FROZEN_ENGINE,
    entity_label,
    load_manifest,
    matchup_key,
)
from tools.operations.pool_labels import pairing_display_label
from tools.operations.opponent_curriculum import preferred_adaptive_opponent

ROOT = Path(__file__).resolve().parent.parent
REMOTE_TIMING = ROOT / "training" / "data" / "remote_timing.json"

POOL_SLOTS_MAX = 8
POOL_SLOTS_WITH_TRAIN = 7  # 8th CPU budget: eval-batch + micro-train (see datagen._eval_batch_lock)
PARALLEL_MATCHUPS = POOL_SLOTS_MAX  # legacy alias

GAMES_PER_MATCHUP = 1


def pool_slots(*, train: bool = True, override: int | None = None) -> int:
    """Game slots for the continuous pool (max 8 on ProgressBoard).

    Default: 7 with background NNUE (reserve one thread), 8 with --no-train.
    Override via --parallel or POOL_SLOTS env.
    """
    if override is not None and override > 0:
        return min(override, POOL_SLOTS_MAX)
    env = os.environ.get("POOL_SLOTS")
    if env:
        return max(1, min(int(env), POOL_SLOTS_MAX))
    return POOL_SLOTS_WITH_TRAIN if train else POOL_SLOTS_MAX


def max_remote_parallel() -> int:
    """Ka remote games in flight — matches active pool slot count."""
    return pool_slots(train=True)

Kind = Literal["local", "remote"]


@dataclass(frozen=True)
class Pairing:
    kind: Kind
    engine_a: str
    engine_b: str
    tc_a: str
    tc_b: str
    label: str
    trainable: bool = False
    priority: int = 0
    target_games: int = 0
    our_time: float = 5.0
    ponder_time: float | None = None
    opponent_profile: str | None = None
    opponent_visits: int | None = None

    def manifest_key(self) -> str:
        return matchup_key(self.engine_a, self.engine_b, self.tc_a, self.tc_b)

    def games_file_name(self, batch_id: str = "") -> str:
        safe = self.label.replace("/", "-").replace(" ", "_")
        if batch_id:
            return f"{safe}-{batch_id}.games"
        return f"{safe}.games"

    def source_tag(self, _game_id: str = "") -> str:
        return f"pool-{self.label}"


def pairing_game_entry(pairing: Pairing, game_id: str, _tournament_dir: Path) -> dict:
    """JSON payload for one game worker (local or remote). Games go to SQLite only."""
    entry = {
        "kind": pairing.kind,
        "label": pairing.label,
        "display_label": pairing_display_label(pairing),
        "engine_a": pairing.engine_a,
        "engine_b": pairing.engine_b,
        "tc_a": pairing.tc_a,
        "tc_b": pairing.tc_b,
        "source_tag": pairing.source_tag(),
        "game_id": game_id,
        "release_remote": True,
    }
    if pairing.opponent_profile:
        entry["opponent_profile"] = pairing.opponent_profile
    if pairing.opponent_visits is not None:
        entry["opponent_visits"] = pairing.opponent_visits
    return entry


def remote_availability() -> dict[str, bool]:
    """From last wiring probe; missing file → assume available."""
    try:
        doc = json.loads(REMOTE_TIMING.read_text(encoding="utf-8"))
        avail = doc.get("availability") or {}
        return {k: bool(v) for k, v in avail.items()}
    except (OSError, json.JSONDecodeError):
        return {}


def all_pairings() -> list[Pairing]:
    """v15 vs Ka + v15 vs ti-pure (train); v15 vs JS ace-v13 (bench only)."""
    p: list[Pairing] = []

    # JS v13 — Elo bench only (not anchor)
    p.append(Pairing(
        kind="local",
        engine_a=CURRENT_ENGINE,
        engine_b="ace-v13",
        tc_a="5s",
        tc_b="5s",
        label=f"{CURRENT_ENGINE}-vs-ace-v13-5s",
        trainable=False,
    ))

    # Rust ti-pure @ 5s — anchor @ 1200 + deploy gate (manual / legacy pool)
    p.append(Pairing(
        kind="local",
        engine_a=CURRENT_ENGINE,
        engine_b="ace-v13-ti-pure",
        tc_a="5s",
        tc_b="5s",
        label=f"{CURRENT_ENGINE}-vs-ace-v13-ti-pure-5s",
        trainable=True,
    ))

    # ti-pure @ 10s — reserved train slot (search labels, longer think)
    p.append(Pairing(
        kind="local",
        engine_a=CURRENT_ENGINE,
        engine_b="ace-v13-ti-pure",
        tc_a="10s",
        tc_b="10s",
        label=f"{CURRENT_ENGINE}-vs-ace-v13-ti-pure-10s",
        trainable=True,
    ))

    # v15 self-play @ 10s — reserved train slot
    p.append(Pairing(
        kind="local",
        engine_a=CURRENT_ENGINE,
        engine_b=CURRENT_ENGINE,
        tc_a="10s",
        tc_b="10s",
        label=f"{CURRENT_ENGINE}-vs-{CURRENT_ENGINE}-10s",
        trainable=True,
    ))

    # Training vs frozen v13 HalfPW @ 5s — A/B yardstick
    p.append(Pairing(
        kind="local",
        engine_a=CURRENT_ENGINE,
        engine_b=FROZEN_ENGINE,
        tc_a="5s",
        tc_b="5s",
        label=f"{CURRENT_ENGINE}-vs-{FROZEN_ENGINE}-5s",
        trainable=False,
    ))

    # Same search, 10s think — longer-horizon net comparison
    p.append(Pairing(
        kind="local",
        engine_a=CURRENT_ENGINE,
        engine_b=FROZEN_ENGINE,
        tc_a="10s",
        tc_b="10s",
        label=f"{CURRENT_ENGINE}-vs-{FROZEN_ENGINE}-10s",
        trainable=False,
    ))

    def train_adaptive(opp: str) -> None:
        p.append(Pairing(
            kind="remote",
            engine_a=CURRENT_ENGINE,
            engine_b=opp,
            tc_a="5s",
            tc_b="adaptive",
            label=f"v15-vs-{opp}-adaptive",
            trainable=True,
            our_time=0,
            ponder_time=0,
            opponent_profile="adaptive",
        ))

    train_adaptive("ka")
    train_adaptive("zero")

    return p


def eligible_pairings(manifest: dict | None = None) -> list[Pairing]:
    """Available pairings (skips unavailable remotes)."""
    avail = remote_availability()
    local_only = os.environ.get("POOL_LOCAL_ONLY") == "1"
    out: list[Pairing] = []
    for pairing in all_pairings():
        if local_only and pairing.kind == "remote":
            continue
        if pairing.kind == "remote" and avail.get(pairing.engine_b) is False:
            continue
        out.append(pairing)
    return out or all_pairings()


def _games_played(manifest: dict, pairing: Pairing) -> int:
    m = manifest.get("matchups", {}).get(pairing.manifest_key(), {})
    return m.get("games_played", m.get("a_wins", 0) + m.get("b_wins", 0))


ANCHOR_BENCH_RATE = 0.08  # legacy (JS no longer random — 1 slot reserved)
FROZEN_BENCH_RATE = float(os.environ.get("FROZEN_BENCH_RATE", "0.28"))  # legacy alias
KA_TIME_CONTROLS = ("adaptive",)
MAX_KA_PER_TC = int(os.environ.get("MAX_KA_PER_TC", "1"))
OUR_TIME_CONTROLS = ("5s", "10s")
RESERVED_ADAPTIVE_SLOTS = 1
RESERVED_TI_PURE_10S = 1
RESERVED_SELF_10S = 1
RESERVED_FROZEN_5S = 1


def pairing_slot_tag(pairing: Pairing) -> str:
    """Coordinator slot bucket for reserved pool layout."""
    if pairing.kind == "remote" and pairing.engine_b == "ka":
        return f"ka:{pairing.tc_b}"
    if pairing.kind == "remote" and pairing.engine_b == "zero":
        return "zero:adaptive"
    if pairing.engine_b == "ace-v13":
        return "js"
    if pairing.engine_b == FROZEN_ENGINE:
        return f"frozen:{pairing.tc_a}"
    if pairing.engine_b == "ace-v13-ti-pure" and pairing.tc_a == "10s":
        return "ti-pure:10s"
    if pairing.engine_a == pairing.engine_b == CURRENT_ENGINE:
        return f"self:{pairing.tc_a}"
    return "other"


def is_trainable_source_tag(tag: str) -> bool:
    """True if a completed DB game can train from its final winner."""
    _ = tag
    return True


def pick_one_pairing(
    manifest: dict | None = None,
    *,
    slot_counts: dict | None = None,
    n_pool_slots: int | None = None,
    allow_remote: bool = True,  # legacy — ignored when slot_counts set
    allow_ka_long: bool = True,  # legacy
    ka_tc_free: dict[str, bool] | None = None,  # legacy
) -> Pairing | None:
    """Adaptive Ka + zero-ink, ti-pure@10s, self@10s, and frozen@5s."""
    manifest = manifest or load_manifest()
    pool = eligible_pairings(manifest)

    ka_inf = (slot_counts or {}).get("ka") or {tc: 0 for tc in KA_TIME_CONTROLS}
    frozen_inf = (slot_counts or {}).get("frozen") or {tc: 0 for tc in OUR_TIME_CONTROLS}
    ti_pure_inf = int((slot_counts or {}).get("ti_pure_10s", 0))
    self_inf = int((slot_counts or {}).get("self_10s", 0))
    zero_inf = int((slot_counts or {}).get("zero", 0))

    def pick_ka(tc: str) -> Pairing | None:
        for p in pool:
            if p.kind == "remote" and p.engine_b == "ka" and p.tc_b == tc:
                return p
        return None

    def pick_zero() -> Pairing | None:
        for p in pool:
            if p.kind == "remote" and p.engine_b == "zero":
                return p
        return None

    def pick_frozen_5s() -> Pairing | None:
        for p in pool:
            if p.engine_b == FROZEN_ENGINE and p.tc_a == "5s":
                return p
        return None

    def pick_ti_pure_10s() -> Pairing | None:
        for p in pool:
            if p.engine_b == "ace-v13-ti-pure" and p.tc_a == "10s":
                return p
        return None

    def pick_self_10s() -> Pairing | None:
        for p in pool:
            if p.engine_a == p.engine_b == CURRENT_ENGINE and p.tc_a == "10s":
                return p
        return None

    if allow_remote:
        adaptive_in_flight = zero_inf + sum(ka_inf.values())
        if adaptive_in_flight < RESERVED_ADAPTIVE_SLOTS:
            preferred = preferred_adaptive_opponent()
            p = pick_zero() if preferred == "zero" else pick_ka("adaptive")
            if p:
                return p

    if ti_pure_inf < RESERVED_TI_PURE_10S:
        p = pick_ti_pure_10s()
        if p:
            return p

    if self_inf < RESERVED_SELF_10S:
        p = pick_self_10s()
        if p:
            return p

    if frozen_inf.get("5s", 0) < RESERVED_FROZEN_5S:
        p = pick_frozen_5s()
        if p:
            return p

    return None


def pick_random_batch(manifest: dict | None = None, n: int | None = None) -> list[Pairing]:
    n = n if n is not None else max_remote_parallel()
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
            if p.kind == "remote" and remotes_used >= max_remote_parallel():
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
        role = "train" if pairing.trainable else "bench"
        rows.append((pairing, _games_played(manifest, pairing), role))
    rows.sort(key=lambda x: (x[2] != "train", x[1]))
    return rows
