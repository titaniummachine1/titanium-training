"""Training-data manifest — Elo tracker + global rating ladder.

Per matchup: only a_wins / b_wins (+ optional time-control labels).
Elo diff recomputed from W/L ratio. Global ladder propagates diffs from
anchor ace-v13-ti-pure@5s = 1200 (Rust ti-pure; JS ace-v13 is bench-only).
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "training" / "data"
MANIFEST_PATH = DATA / "manifest.json"
STATUS_PATH = DATA / "STATUS.txt"
LOCK_PATH = DATA / "manifest.lock"

CURRENT_ENGINE = "titanium-v15"
FROZEN_ENGINE = "titanium-v15-frozen"
V15_SHORT = "v15"
V15_FROZEN_SHORT = "v15-frozen"
OUR_TRACKED_TCS = ("5s", "10s")
# Ladder anchor — Rust ace-v13-ti-pure @ 1200 (same binary as train/deploy gate).
ANCHOR_ENGINE = "ace-v13-ti-pure"
BASELINE_ENGINE = ANCHOR_ENGINE
# Bare "titanium" = legacy GameSearchSession (MCTS), NOT v15 or ace-v13 — exclude from ladder.
DEPRECATED_LADDER_ENGINES = frozenset({"titanium", "titanium-cert", "titanium-plain"})
ANCHOR_ENTITY = f"{ANCHOR_ENGINE}@5s"
REMOTE_ENGINES = frozenset({"ka", "ishtar", "zero"})
# Site UI labels for Ka/Ishtar time presets (strength fixed at Alpha on wire).
REMOTE_TIME_LABELS = {
    "intuition": "Immediate",
    "immediate": "Immediate",
    "short": "Short",
    "medium": "Medium",
    "long": "Long",
    "adaptive": "Adaptive",
}
# ti-pure @ 5s pinned as ladder reference.
ANCHOR_RATING = 1200.0
MIN_GAMES_GLOBAL = 2  # include Ka/remote on ladder after a few games
MIN_GAMES_LADDER_STABLE = 4  # shown as note in STATUS when below this

PATHS = {
    "training_db": str(DATA / "all_games.db"),
    "benchmark_log": str(DATA / "benchmarks_log.jsonl"),
    "strength_tracker_games": str(DATA / "v15_vs_ti_pure.games"),
    "self_match_games": str(DATA / "self_match_games.games"),
    "benchmark_games_dir": str(DATA / "benchmarks"),
    "tournament_games_dir": str(DATA / "tournament"),
}

_LEGACY_STRENGTH_GAMES = DATA / "v14_vs_ti_pure.games"
_LEGACY_MANIFEST_KEY = "v14_vs_ti_pure"


def entity_label(engine: str, tc: str | None) -> str:
    tc = (tc or "5s").strip()
    return f"{engine}@{tc}"


def is_deprecated_engine(engine: str) -> bool:
    """Legacy session flags that are not ace-v13 family or titanium-v15."""
    return engine.split("@", 1)[0] in DEPRECATED_LADDER_ENGINES


def is_deprecated_entity(ent: str) -> bool:
    return is_deprecated_engine(ent.split("@", 1)[0])


def engine_display_short(engine: str, tc: str | None = None) -> str:
    """Compact engine@tc label for pool dock + matchups."""
    tc = (tc or "5s").strip()
    if engine == CURRENT_ENGINE:
        return f"{V15_SHORT}@{tc}"
    if engine == FROZEN_ENGINE:
        return f"{V15_FROZEN_SHORT}@{tc}"
    if engine == "ace-v13":
        return f"JS-v13@{tc}"
    if engine == ANCHOR_ENGINE:
        return f"ti-pure@{tc}"
    if engine in REMOTE_ENGINES:
        ui = REMOTE_TIME_LABELS.get(tc, tc)
        name = {"ka": "Ka", "ishtar": "Ishtar", "zero": "zero"}[engine]
        return f"{name}-{ui}"
    return f"{engine}@{tc}"


def display_entity(ent: str) -> str:
    """Scoreboard-friendly label (remote presets show UI time name)."""
    if "@" not in ent:
        return ent
    base, tc = ent.split("@", 1)
    if base in REMOTE_ENGINES:
        ui = REMOTE_TIME_LABELS.get(tc, tc)
        name = {"ka": "Ka", "ishtar": "Ishtar", "zero": "zero"}[base]
        return f"{name}@{tc} ({ui})"
    if base == ANCHOR_ENGINE:
        return f"ti-pure@{tc} (Rust ref)"
    if base == "ace-v13":
        return f"JS-v13@{tc} (bench)"
    if base == FROZEN_ENGINE:
        return f"{V15_FROZEN_SHORT}@{tc} (HalfPW frozen)"
    if base == CURRENT_ENGINE:
        return f"{V15_SHORT}@{tc} (HalfPW live)"
    return ent


def _short_engine(engine: str, tc: str | None = None) -> str:
    return engine_display_short(engine, tc)


def display_matchup_label(m: dict) -> str:
    tc_a = m.get("tc_a", "5s")
    tc_b = m.get("tc_b", "5s")
    return f"{_short_engine(m['a_engine'], tc_a)} vs {_short_engine(m['b_engine'], tc_b)}"


def matchup_key(
    engine_a: str,
    engine_b: str,
    tc_a: str | None = None,
    tc_b: str | None = None,
) -> str:
    return f"{engine_a}|{engine_b}|{tc_a or '5s'}|{tc_b or '5s'}"


def _legacy_matchup_key(engine_a: str, engine_b: str) -> str:
    return f"{engine_a}|{engine_b}"


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    if path.suffix == ".db":
        import sqlite3
        try:
            conn = sqlite3.connect(str(path))
            n = conn.execute("SELECT COUNT(*) FROM games").fetchone()[0]
            conn.close()
            return n
        except Exception:
            return 0
    with open(path, encoding="utf-8", errors="replace") as f:
        return sum(1 for _ in f)


def count_games_in_file(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("RESULT ") and line.split()[1] in ("W", "B"):
                n += 1
    return n


def elo_a_vs_b(a_wins: int, b_wins: int) -> float | None:
    n = a_wins + b_wins
    if n == 0:
        return None
    # Pseudocount avoids None at 0% / 100% (needed for Ka@medium/long on global ladder).
    p = (a_wins + 0.5) / (n + 1)
    if p <= 0 or p >= 1:
        return None
    import math
    return round(-400 * math.log10((1 - p) / p), 1)


def _migrate_legacy_strength(manifest: dict) -> dict:
    if "strength_tracker" in manifest:
        return manifest
    legacy = manifest.pop(_LEGACY_MANIFEST_KEY, None)
    if legacy:
        manifest["strength_tracker"] = legacy
    return manifest


def _canonical_tc(engine: str, tc: str | None) -> str:
    """Drop legacy fair-* labels — v15 is always tracked as @5s on the ladder."""
    tc = (tc or "5s").strip()
    if tc.startswith("fair-"):
        return "5s"
    return tc


def _normalize_matchups(manifest: dict) -> dict:
    """Merge legacy 2-part keys; fold fair-* tc_a into 5s."""
    raw = manifest.get("matchups", {})
    out: dict = {}

    def merge_into(nk: str, entry: dict) -> None:
        if nk in out:
            out[nk]["a_wins"] = out[nk].get("a_wins", 0) + entry.get("a_wins", 0)
            out[nk]["b_wins"] = out[nk].get("b_wins", 0) + entry.get("b_wins", 0)
            out[nk]["games_played"] = out[nk]["a_wins"] + out[nk]["b_wins"]
            out[nk]["elo_a_vs_b"] = elo_a_vs_b(out[nk]["a_wins"], out[nk]["b_wins"])
            for k in ("games_file", "last_source", "games_in_file"):
                if entry.get(k):
                    out[nk][k] = entry[k]
        else:
            out[nk] = entry

    for key, entry in raw.items():
        entry = dict(entry)
        parts = key.split("|")
        if len(parts) == 2:
            nk = matchup_key(parts[0], parts[1])
            entry.setdefault("tc_a", "5s")
            entry.setdefault("tc_b", "5s")
            entry["a_engine"] = parts[0]
            entry["b_engine"] = parts[1]
            merge_into(nk, entry)
        elif len(parts) >= 4:
            ea, eb = parts[0], parts[1]
            tc_a = _canonical_tc(ea, entry.get("tc_a", parts[2]))
            tc_b = _canonical_tc(eb, entry.get("tc_b", parts[3]))
            entry["tc_a"] = tc_a
            entry["tc_b"] = tc_b
            entry["a_engine"] = ea
            entry["b_engine"] = eb
            nk = matchup_key(ea, eb, tc_a, tc_b)
            merge_into(nk, entry)
        else:
            out[key] = entry
    for entry in out.values():
        aw, bw = entry.get("a_wins", 0), entry.get("b_wins", 0)
        entry["games_played"] = aw + bw
        entry["elo_a_vs_b"] = elo_a_vs_b(aw, bw)
    manifest["matchups"] = out
    return manifest


def aggregate_entity_wl(matchups: dict) -> dict[str, dict]:
    stats: dict[str, dict] = defaultdict(lambda: {"wins": 0, "losses": 0})
    for m in matchups.values():
        tc_a = m.get("tc_a", "5s")
        tc_b = m.get("tc_b", "5s")
        ea = entity_label(m["a_engine"], tc_a)
        eb = entity_label(m["b_engine"], tc_b)
        aw, bw = m.get("a_wins", 0), m.get("b_wins", 0)
        stats[ea]["wins"] += aw
        stats[ea]["losses"] += bw
        stats[eb]["wins"] += bw
        stats[eb]["losses"] += aw
        stats[ea]["games"] = stats[ea]["wins"] + stats[ea]["losses"]
        stats[eb]["games"] = stats[eb]["wins"] + stats[eb]["losses"]
    return dict(stats)


def compute_global_ratings(matchups: dict) -> dict[str, dict]:
    """Naive fixed-point propagation from anchor through measured matchup edges."""
    edges: list[tuple[str, str, float]] = []
    entities: set[str] = {ANCHOR_ENTITY}

    for m in matchups.values():
        n = m.get("games_played", m.get("a_wins", 0) + m.get("b_wins", 0))
        if n < MIN_GAMES_GLOBAL:
            continue
        diff = m.get("elo_a_vs_b")
        if diff is None:
            continue
        ea = entity_label(m["a_engine"], m.get("tc_a", "5s"))
        eb = entity_label(m["b_engine"], m.get("tc_b", "5s"))
        if is_deprecated_entity(ea) or is_deprecated_entity(eb):
            continue
        entities.add(ea)
        entities.add(eb)
        edges.append((ea, eb, float(diff)))

    ratings: dict[str, float] = {ANCHOR_ENTITY: ANCHOR_RATING}
    for _ in range(40):
        accum: dict[str, list[float]] = defaultdict(list)
        for ea, eb, diff in edges:
            if eb in ratings:
                accum[ea].append(ratings[eb] + diff)
            if ea in ratings:
                accum[eb].append(ratings[ea] - diff)
        changed = False
        for ent, vals in accum.items():
            if ent == ANCHOR_ENTITY or not vals:
                continue
            new_r = sum(vals) / len(vals)
            if ent not in ratings or abs(ratings.get(ent, 0) - new_r) > 0.05:
                changed = True
            ratings[ent] = new_r
        if not changed:
            break

    # Same engine @ different time controls = same strength (remotes only; v15 tracks per tc).
    bases: dict[str, list[float]] = defaultdict(list)
    for ent, r in ratings.items():
        base = ent.split("@", 1)[0]
        if base in REMOTE_ENGINES or base in (CURRENT_ENGINE, FROZEN_ENGINE):
            continue
        bases[base].append(r)
    base_rating = {b: sum(v) / len(v) for b, v in bases.items() if v}
    for ent in entities:
        if ent in ratings:
            continue
        base = ent.split("@", 1)[0]
        if base in REMOTE_ENGINES or base in (CURRENT_ENGINE, FROZEN_ENGINE):
            continue
        if base in base_rating:
            ratings[ent] = base_rating[base]

    # Fill remaining nodes one edge away from any rated entity.
    for _ in range(10):
        for ea, eb, diff in edges:
            if eb in ratings and ea not in ratings:
                ratings[ea] = ratings[eb] + diff
            elif ea in ratings and eb not in ratings:
                ratings[eb] = ratings[ea] - diff

    wl = aggregate_entity_wl(matchups)
    out: dict[str, dict] = {}
    for ent in entities:
        if ent not in ratings or is_deprecated_entity(ent):
            continue
        s = wl.get(ent, {})
        out[ent] = {
            "rating": round(ratings[ent]),
            "wins": s.get("wins", 0),
            "losses": s.get("losses", 0),
            "games": s.get("games", 0),
            "anchor": ent == ANCHOR_ENTITY,
            "provisional": s.get("games", 0) < MIN_GAMES_LADDER_STABLE,
        }
    return out


@contextmanager
def manifest_lock(timeout_sec: float = 30.0):
    """Exclusive lock for manifest read-modify-write (parallel game updates)."""
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        try:
            fd = os.open(str(LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            time.sleep(0.05)
    else:
        raise RuntimeError("manifest lock timeout")
    try:
        yield
    finally:
        LOCK_PATH.unlink(missing_ok=True)


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    else:
        manifest = {"paths": PATHS, "sources": {}, "matchups": {}, "global_ratings": {}}
    manifest["paths"] = PATHS
    manifest["current_engine"] = CURRENT_ENGINE
    manifest["baseline_engine"] = ANCHOR_ENGINE
    manifest["anchor_entity"] = ANCHOR_ENTITY
    manifest["anchor_rating"] = ANCHOR_RATING
    manifest = _migrate_legacy_strength(manifest)
    manifest = _normalize_matchups(manifest)
    manifest["global_ratings"] = compute_global_ratings(manifest.get("matchups", {}))
    return manifest


def _write_manifest(manifest: dict) -> None:
    manifest["current_engine"] = CURRENT_ENGINE
    manifest["baseline_engine"] = ANCHOR_ENGINE
    manifest["anchor_entity"] = ANCHOR_ENTITY
    manifest["anchor_rating"] = ANCHOR_RATING
    manifest["global_ratings"] = compute_global_ratings(manifest.get("matchups", {}))
    DATA.mkdir(parents=True, exist_ok=True)
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    manifest["paths"] = PATHS
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    _write_status_txt(manifest)


def save_manifest(manifest: dict) -> None:
    with manifest_lock():
        _write_manifest(manifest)


def lookup_prior_wins(
    engine_a: str,
    engine_b: str,
    tc_a: str | None = None,
    tc_b: str | None = None,
) -> tuple[int, int]:
    manifest = load_manifest()
    matchups = manifest.get("matchups", {})
    for key in (matchup_key(engine_a, engine_b, tc_a, tc_b), _legacy_matchup_key(engine_a, engine_b)):
        m = matchups.get(key)
        if m:
            return m.get("a_wins", 0), m.get("b_wins", 0)
    return 0, 0


def update_matchup(
    engine_a: str,
    engine_b: str,
    a_wins: int,
    b_wins: int,
    tc_a: str | None = None,
    tc_b: str | None = None,
    games_file: str | Path | None = None,
    source: str | None = None,
) -> dict:
    with manifest_lock():
        manifest = load_manifest()
        tc_a = tc_a or "5s"
        tc_b = tc_b or "5s"
        key = matchup_key(engine_a, engine_b, tc_a, tc_b)
        n = a_wins + b_wins
        entry = {
            "a_engine": engine_a,
            "b_engine": engine_b,
            "tc_a": tc_a,
            "tc_b": tc_b,
            "a_wins": a_wins,
            "b_wins": b_wins,
            "games_played": n,
            "elo_a_vs_b": elo_a_vs_b(a_wins, b_wins),
        }
        if games_file:
            entry["games_file"] = str(games_file)
            entry["games_in_file"] = count_games_in_file(Path(games_file))
        if source:
            entry["last_source"] = source
        manifest.setdefault("matchups", {})[key] = entry
        legacy = _legacy_matchup_key(engine_a, engine_b)
        if legacy in manifest["matchups"] and legacy != key:
            del manifest["matchups"][legacy]
        if engine_a == CURRENT_ENGINE and engine_b == BASELINE_ENGINE and tc_a == "5s" and tc_b == "5s":
            manifest["strength_tracker"] = {**entry, "elo_vs_baseline": entry.get("elo_a_vs_b")}
        _write_manifest(manifest)
        return entry


def update_strength_tracker(a_wins: int, b_wins: int, batch: int | None = None, **_kw) -> None:
    update_matchup(CURRENT_ENGINE, BASELINE_ENGINE, a_wins, b_wins, "5s", "5s")
    if batch is not None:
        manifest = load_manifest()
        k = matchup_key(CURRENT_ENGINE, BASELINE_ENGINE, "5s", "5s")
        manifest["matchups"][k]["batches_completed"] = batch
        save_manifest(manifest)


update_v15_vs_ti_pure = update_strength_tracker


def update_source(name: str, games_file: str | Path, **extra) -> None:
    games_file = Path(games_file)
    manifest = load_manifest()
    manifest.setdefault("sources", {})[name] = {
        "games_file": str(games_file),
        "games": count_games_in_file(games_file),
        "bytes": games_file.stat().st_size if games_file.exists() else 0,
        **extra,
    }
    save_manifest(manifest)


def _board_width() -> int:
    try:
        import shutil
        return max(72, min(120, shutil.get_terminal_size(fallback=(100, 40)).columns))
    except Exception:
        return 100


def _format_tc(tc_a: str, tc_b: str) -> str:
    if tc_a == tc_b:
        return tc_a
    return f"A:{tc_a} B:{tc_b}"


def _nnue_status_line() -> str:
    try:
        from titanium_training.training.plateau_probe import nnue_status_compact
        return nnue_status_compact()
    except Exception:
        pass
    try:
        g = json.loads((DATA / "nnue_guard_state.json").read_text(encoding="utf-8"))
        return f"NNUE  trained {g.get('games_trained', '?')}  deploys {g.get('deploy_runs', 0)}"
    except Exception:
        return "NNUE  (status unavailable)"


def format_scoreboard_compact(manifest: dict) -> str:
    """Short dashboard for live pool terminal — one opponent table + our v15."""
    matchups = manifest.get("matchups", {})
    global_ratings = compute_global_ratings(matchups)
    t = manifest.get("tournament", {})
    try:
        from tools.datagen.datagen import count_pool_games, max_game_id
        pool_db = count_pool_games(Path(PATHS["training_db"]))
        total_db = max_game_id(Path(PATHS["training_db"]))
    except Exception:
        pool_db = _count_lines(Path(PATHS["training_db"]))
        total_db = pool_db
    pool_run = int(t.get("games", 0)) if t.get("mode") == "random-pool" else 0
    slots = t.get("parallel", 7)

    cur5 = entity_label(CURRENT_ENGINE, "5s")
    cur10 = entity_label(CURRENT_ENGINE, "10s")
    gr5 = global_ratings.get(cur5, {})
    gr10 = global_ratings.get(cur10, {})
    v15_elo = int(gr5.get("rating", 0)) if gr5 else 0
    v15_d = v15_elo - int(ANCHOR_RATING) if gr5 else 0
    v15_sign = "+" if v15_d >= 0 else ""

    def opp_label(m: dict) -> tuple[str, str]:
        eb = m.get("b_engine", "")
        tc_a, tc_b = m.get("tc_a", "5s"), m.get("tc_b", "5s")
        if eb == "ka":
            label = "Ka-imm" if tc_b == "intuition" else f"Ka-{tc_b}"
            ent = f"ka@{tc_b}"
        elif eb == "zero":
            label = f"zero-{tc_b}"
            ent = f"zero@{tc_b}"
        elif eb == "ace-v13":
            label = f"JS-v13@{tc_a}"
            ent = entity_label("ace-v13", tc_a)
        elif eb == ANCHOR_ENGINE:
            label = f"ti-pure@{tc_a}"
            ent = entity_label(ANCHOR_ENGINE, tc_a)
        elif eb == FROZEN_ENGINE:
            label = f"v15-frozen@{tc_a}"
            ent = entity_label(FROZEN_ENGINE, tc_a)
        else:
            label = f"{eb}@{tc_b}"
            ent = entity_label(eb, tc_b)
        return label, ent

    def vs_us_note(m: dict, aw: int, bw: int, n: int, diff: float | None) -> str:
        eb = m.get("b_engine", "")
        if eb == ANCHOR_ENGINE:
            return "anchor"
        if eb == FROZEN_ENGINE:
            return "past-self"
        if eb == "ace-v13":
            return "js orig"
        if diff is not None and diff < -50:
            return "losing"
        if aw == 0 and bw > 0 and n >= 3:
            return "losing"
        if aw > bw and n >= 2:
            return "winning"
        return "even"

    rows: list[tuple[float, str]] = []
    for m in matchups.values():
        if m.get("a_engine") != CURRENT_ENGINE:
            continue
        if m.get("b_engine") == CURRENT_ENGINE:
            continue
        n = m.get("games_played", m.get("a_wins", 0) + m.get("b_wins", 0))
        if n == 0:
            continue
        label, ent = opp_label(m)
        aw, bw = m.get("a_wins", 0), m.get("b_wins", 0)
        diff = m.get("elo_a_vs_b")
        diff_s = f"{diff:+.0f}" if diff is not None else "  ?"
        gr = global_ratings.get(ent, {})
        elo = float(gr.get("rating", 0))
        note = vs_us_note(m, aw, bw, n, diff)
        line = (
            f"| {label:<16} | {int(elo):>4} | {aw:>2}-{bw:<2} | {n:>3} | {diff_s:>6} | {note:<8} |"
        )
        rows.append((elo, line))
    rows.sort(key=lambda x: -x[0])

    w = 72
    lines = [
        "+" + "-" * w + "+",
        f"| QUORIDOR POOL  run={pool_run}  pool={pool_db}  total={total_db}  slots={slots}"
        f"  v15@5s={v15_elo} ({v15_sign}{v15_d})".ljust(w - 1) + "|",
        "+" + "-" * w + "+",
        "| opponent         | Elo  | W-L |  n  |  H2H  | vs us    |",
        "|------------------+------+-----+-----+-------+----------|",
    ]
    for i, (_, line) in enumerate(rows[:11], 1):
        lines.append(line)
    if len(rows) > 11:
        lines.append(f"| ... +{len(rows) - 11} more opponents".ljust(w - 1) + "|")
    lines.append("+" + "-" * w + "+")
    if gr10.get("games", 0):
        r10 = gr10.get("rating", 0)
        lines.append(
            f"| v15@10s {r10} Elo  {gr10.get('wins',0)}-{gr10.get('losses',0)}"
            f" ({gr10.get('games',0)}g)".ljust(w - 1) + "|"
        )
    lines.append("+" + "-" * w + "+")
    try:
        from titanium_training.training.learning_metrics import format_scoreboard_train_block
        lines.extend(format_scoreboard_train_block())
    except Exception:
        nnue = _nnue_status_line().strip()
        if len(nnue) > w - 4:
            nnue = nnue[: w - 7] + "..."
        lines.append("| " + nnue.ljust(w - 4) + " |")
    lines.append("+" + "-" * w + "+")
    return "\n".join(lines)


def format_scoreboard(manifest: dict) -> str:
    """Terminal-friendly ladder + matchup W/L (same data as STATUS.txt core)."""
    matchups = manifest.get("matchups", {})
    global_ratings = compute_global_ratings(matchups)
    t = manifest.get("tournament", {})
    db_path = Path(PATHS["training_db"])
    db_records = _count_lines(db_path)
    try:
        from tools.datagen.datagen import count_pool_games

        pool_db = count_pool_games(db_path)
    except Exception:
        pool_db = db_records
    legacy_db = max(0, db_records - pool_db)

    bw = _board_width()
    lines = [
        "",
        "=" * bw,
        f" SCOREBOARD   anchor {display_entity(ANCHOR_ENTITY)} = {int(ANCHOR_RATING)} Elo"
        f"   |   {pool_db} pool games in DB"
        + (f"  (+{legacy_db} older rows kept, excluded from ladder)" if legacy_db else ""),
        "=" * bw,
    ]
    if t:
        mode = t.get("mode", "random")
        if mode == "random":
            batch = t.get("batch", 0)
            last = t.get("last_batch") or []
            par = t.get("parallel", 4)
            lines.append(f" Random batch #{batch}  |  {par} parallel matchups, 1 game each")
            if last:
                lines.append(f" Last batch: {', '.join(last)}")
        elif mode == "random-pool":
            pool_run = int(t.get("games", 0))
            par = t.get("parallel", 4)
            lines.append(
                f" Continuous pool  |  {par} slots  |  {pool_run} games this pool run"
                f"  |  {pool_db} pool-tagged in DB"
            )
        elif mode == "round_robin":
            cycle = t.get("cycle", 1)
            idx = t.get("cycle_index", 0)
            total = t.get("cycle_total") or "?"
            slot = (idx % int(total)) + 1 if total != "?" and int(total) else "?"
            lines.append(
                f" Round-robin cycle {cycle}  |  next slot {slot}/{total}"
            )
            lines.append(f" Last pairing: {t.get('last_pairing', '?')}")
            lines.append(f" Next pairing: {t.get('next_pairing', '?')}")
        else:
            lines.append(
                f" Swiss round {t.get('round', '?')}  |  last: {t.get('last_pairing', '?')} ({t.get('last_kind', '?')})"
            )
            lines.append(f" Last pairing: {t.get('last_pairing', '?')}")
        lines.append("-" * bw)

    if global_ratings:
        lines.append(" GLOBAL LADDER")
        ranked = sorted(
            ((k, v) for k, v in global_ratings.items() if not is_deprecated_entity(k)),
            key=lambda x: -x[1]["rating"],
        )
        for i, (ent, info) in enumerate(ranked, 1):
            w, l, g = info.get("wins", 0), info.get("losses", 0), info.get("games", 0)
            tag = " [anchor]" if info.get("anchor") else ""
            if info.get("provisional"):
                tag += " [prov]"
            wr = f"{100 * w / g:.0f}%" if g else "?"
            lines.append(
                f"  #{i:<2} {display_entity(ent):<34} {info['rating']:>4} Elo   {w}-{l}  ({g}g, {wr}){tag}"
            )
        wl = aggregate_entity_wl(matchups)
        cur5 = entity_label(CURRENT_ENGINE, "5s")
        if cur5 in global_ratings:
            delta = global_ratings[cur5]["rating"] - int(ANCHOR_RATING)
            sign = "+" if delta >= 0 else ""
            lines.append(f"  >> {display_entity(cur5)} = {global_ratings[cur5]['rating']} ({sign}{delta} vs anchor)")
        cur10 = entity_label(CURRENT_ENGINE, "10s")
        s10 = wl.get(cur10, {"games": 0})
        if cur10 in global_ratings:
            lines.append(f"  >> {display_entity(cur10)} = {global_ratings[cur10]['rating']} Elo")
        elif s10.get("games", 0) == 0:
            lines.append(f"  >> {display_entity(cur10)} - awaiting pool games @10s")
        shown = {ent for ent, _ in ranked}
        pending = [
            (ent, wl[ent])
            for ent in wl
            if ent not in shown and wl[ent].get("games", 0) > 0 and not is_deprecated_entity(ent)
        ]
        for ent, s in sorted(pending, key=lambda x: -x[1].get("games", 0)):
            w, l, g = s.get("wins", 0), s.get("losses", 0), s.get("games", 0)
            lines.append(
                f"  .. {display_entity(ent):<34}   -- Elo   {w}-{l}  ({g}g) [pending graph]"
            )
        lines.append("-" * bw)
        lines.append(" V15 LIVE vs FROZEN (by think time)")
        for tc in OUR_TRACKED_TCS:
            for engine in (CURRENT_ENGINE, FROZEN_ENGINE):
                ent = entity_label(engine, tc)
                s = wl.get(ent, {"wins": 0, "losses": 0, "games": 0})
                w, l, g = s.get("wins", 0), s.get("losses", 0), s.get("games", 0)
                gr = global_ratings.get(ent, {})
                if g >= MIN_GAMES_GLOBAL and gr.get("rating") is not None:
                    elo_s = f"{gr['rating']:>4} Elo"
                elif g > 0:
                    elo_s = "  -- Elo"
                else:
                    elo_s = "  -- Elo"
                slot = " [pool]" if g == 0 else ""
                lines.append(
                    f"  {display_entity(ent):<34} {elo_s}   {w}-{l}  ({g}g){slot}"
                )
        lines.append("-" * bw)

    if matchups:
        lines.append(" MATCHUPS (A wins - B wins)")
        rows = []
        for key in sorted(matchups.keys()):
            m = matchups[key]
            aw, bw = m.get("a_wins", 0), m.get("b_wins", 0)
            n = m.get("games_played", aw + bw)
            if n == 0:
                continue
            if is_deprecated_engine(m.get("a_engine", "")) or is_deprecated_engine(m.get("b_engine", "")):
                continue
            elo = m.get("elo_a_vs_b")
            elo_s = f"{elo:+.0f}" if elo is not None else "?"
            se = ((aw / n * (1 - aw / n)) / n) ** 0.5 * 196 if n else 0
            tc = _format_tc(m.get("tc_a", "5s"), m.get("tc_b", "5s"))
            label = display_matchup_label(m)
            rows.append((label, tc, aw, bw, n, elo_s, se))
        for label, tc, aw, bw, n, elo_s, se in rows:
            tc_col = f"({tc})" if tc != "5s" else ""
            lines.append(
                f"  {label:<36} {tc_col:<8}  {aw:>3}-{bw:<3}  {n:>4}g  ~{elo_s} diff (+/-{se:.0f}%)"
            )
    else:
        lines.append(" (no matchups yet)")

    lines.append("=" * bw)
    lines.append("")
    return "\n".join(lines)


def _write_status_txt(manifest: dict) -> None:
    db_records = _count_lines(Path(PATHS["training_db"]))
    matchups = manifest.get("matchups", {})
    global_ratings = manifest.get("global_ratings", {})

    lines = [
        "=== Quoridor training data ===",
        f"Updated: {manifest.get('updated_at', '?')}",
        f"Anchor: {ANCHOR_ENTITY} = {int(ANCHOR_RATING)} Elo",
        "",
        "KEY FILES (training/data/):",
        f"  Training DB:     all_games.db  ({db_records} games)",
        f"  Elo manifest:    manifest.json",
        f"  This summary:    STATUS.txt",
        "",
    ]

    t = manifest.get("tournament", {})
    if t:
        lines += [
            "SWISS TOURNAMENT (overnight):",
            f"  Round {t.get('round', '?')}  last: {t.get('last_pairing', '?')} ({t.get('last_kind', '?')})",
            f"  Next: {t.get('next_pairing', '?')}",
            "",
        ]

    if global_ratings:
        lines.append(
            f"GLOBAL RATING LADDER (anchor {int(ANCHOR_RATING)} = {BASELINE_ENGINE}; "
            "direct H2H is more precise per pairing):"
        )
        ranked = sorted(
            ((k, v) for k, v in global_ratings.items() if not is_deprecated_entity(k)),
            key=lambda x: -x[1]["rating"],
        )
        for i, (ent, info) in enumerate(ranked, 1):
            w, l, g = info.get("wins", 0), info.get("losses", 0), info.get("games", 0)
            anchor = "  [anchor]" if info.get("anchor") else ""
            if info.get("provisional"):
                anchor += "  [prov]"
            wr = f"{100 * w / g:.0f}%" if g else "?"
            lines.append(
                f"  #{i:<2} {ent:<32} {info['rating']:>4} Elo  {w}-{l} ({g}g, {wr} win){anchor}"
            )
        if ranked:
            top = ranked[0][0]
            cur = entity_label(CURRENT_ENGINE, "5s")
            if cur in global_ratings:
                lines.append(
                    f"  >> {cur} = {global_ratings[cur]['rating']} Elo "
                    f"(+{global_ratings[cur]['rating'] - int(ANCHOR_RATING)} vs anchor)"
                )
        lines.append("")

    if matchups:
        lines.append("MATCHUP DETAILS (direct W/L → Elo diff):")
        for key in sorted(matchups.keys()):
            m = matchups[key]
            aw, bw = m.get("a_wins", 0), m.get("b_wins", 0)
            n = m.get("games_played", aw + bw)
            elo = m.get("elo_a_vs_b")
            elo_s = f"{elo:+.0f}" if elo is not None else "?"
            se = ((aw / n * (1 - aw / n)) / n) ** 0.5 * 196 if n else 0
            tc = _format_tc(m.get("tc_a", "5s"), m.get("tc_b", "5s"))
            src = m.get("last_source", "")
            src_s = f" [{src}]" if src else ""
            lines.append(
                f"  {m['a_engine']} vs {m['b_engine']} ({tc}): "
                f"{aw}-{bw} / {n}g  ~{elo_s} diff (±{se:.0f}%){src_s}"
            )
        lines.append("")

    sources = manifest.get("sources", {})
    if sources:
        lines.append("GAME SOURCES:")
        for name, info in sorted(sources.items()):
            lines.append(f"  {name}: {info.get('games', 0)} games -> {info.get('games_file', '?')}")
        lines.append("")

    STATUS_PATH.write_text("\n".join(lines), encoding="utf-8")


def _cli_update_matchup(args) -> None:
    update_matchup(
        args.engine_a, args.engine_b, args.a_wins, args.b_wins,
        tc_a=args.tc_a, tc_b=args.tc_b,
        games_file=args.games_file, source=args.source,
    )
    m = load_manifest()["matchups"][matchup_key(args.engine_a, args.engine_b, args.tc_a, args.tc_b)]
    elo = m.get("elo_a_vs_b")
    elo_s = f"{elo:+.0f}" if elo is not None else "?"
    gr = load_manifest().get("global_ratings", {})
    ea = entity_label(args.engine_a, args.tc_a or "5s")
    eb = entity_label(args.engine_b, args.tc_b or "5s")
    ga = gr.get(ea, {}).get("rating", "?")
    gb = gr.get(eb, {}).get("rating", "?")
    print(
        f"{args.engine_a} {args.a_wins}-{args.b_wins} {args.engine_b} ({args.tc_a}|{args.tc_b}) "
        f"/ {m['games_played']}g  diff ~{elo_s}  ladder {ea}={ga} {eb}={gb}"
    )


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--update-matchup", action="store_true")
    ap.add_argument("--lookup-prior", action="store_true")
    ap.add_argument("--engine-a", required=True)
    ap.add_argument("--engine-b", required=True)
    ap.add_argument("--a-wins", type=int, default=None)
    ap.add_argument("--b-wins", type=int, default=None)
    ap.add_argument("--tc-a", default="5s")
    ap.add_argument("--tc-b", default="5s")
    ap.add_argument("--games-file", default=None)
    ap.add_argument("--source", default=None)
    args = ap.parse_args()
    if args.lookup_prior:
        aw, bw = lookup_prior_wins(args.engine_a, args.engine_b, args.tc_a, args.tc_b)
        print(json.dumps({"a_wins": aw, "b_wins": bw}))
    elif args.update_matchup:
        if args.a_wins is None or args.b_wins is None:
            ap.error("--a-wins and --b-wins required with --update-matchup")
        _cli_update_matchup(args)
