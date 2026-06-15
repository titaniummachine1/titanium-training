"""Training-data manifest — single place to see what games exist and match progress.

Updates training/data/manifest.json (machine-readable) and training/data/STATUS.txt
(human-readable one-glance summary).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "training" / "data"
MANIFEST_PATH = DATA / "manifest.json"
STATUS_PATH = DATA / "STATUS.txt"

# Current production engine vs the JS-v13 baseline used for Elo measurement.
CURRENT_ENGINE = "titanium-v15"
BASELINE_ENGINE = "ace-v13-ti-pure"

# Canonical paths — referenced everywhere so you always know where things live.
PATHS = {
    "training_db": str(DATA / "all_games.db"),
    "benchmark_log": str(DATA / "benchmarks_log.jsonl"),
    "strength_tracker_games": str(DATA / "v15_vs_ti_pure.games"),
    "self_match_games": str(DATA / "self_match_games.games"),
    "benchmark_games_dir": str(DATA / "benchmarks"),
}

# Legacy path from before v15 rename — read for migration only.
_LEGACY_STRENGTH_GAMES = DATA / "v14_vs_ti_pure.games"
_LEGACY_MANIFEST_KEY = "v14_vs_ti_pure"


def _count_lines(path: Path) -> int:
    """Count records; handles SQLite .db and plain text files."""
    if not path.exists():
        return 0
    if path.suffix == ".db":
        import sqlite3
        try:
            conn = sqlite3.connect(str(path))
            n = conn.execute("SELECT COUNT(*) FROM records").fetchone()[0]
            conn.close()
            return n
        except Exception:
            return 0
    with open(path, encoding="utf-8", errors="replace") as f:
        return sum(1 for _ in f)


def count_games_in_file(path: Path) -> int:
    """Count complete GAME/RESULT pairs in a .games dump file."""
    if not path.exists():
        return 0
    n = 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("RESULT ") and line.split()[1] in ("W", "B"):
                n += 1
    return n


def _migrate_legacy_strength(manifest: dict) -> dict:
    """Carry forward totals from pre-v15 manifest keys / game files."""
    key = "strength_tracker"
    if key in manifest:
        return manifest
    legacy = manifest.pop(_LEGACY_MANIFEST_KEY, None)
    if legacy:
        manifest[key] = legacy
        return manifest
    # No manifest entry yet — seed from legacy games file if it exists.
    games_file = Path(PATHS["strength_tracker_games"])
    legacy_file = _LEGACY_STRENGTH_GAMES
    if not games_file.exists() and legacy_file.exists():
        manifest[key] = {
            "a_engine": CURRENT_ENGINE,
            "b_engine": BASELINE_ENGINE,
            "games_file": str(legacy_file),
            "games_in_file": count_games_in_file(legacy_file),
            "note": "legacy v14_vs_ti_pure.games — rename or copy to v15_vs_ti_pure.games",
        }
    return manifest


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    else:
        manifest = {"paths": PATHS, "sources": {}, "strength_tracker": {}}
    manifest["paths"] = PATHS
    manifest["current_engine"] = CURRENT_ENGINE
    manifest["baseline_engine"] = BASELINE_ENGINE
    return _migrate_legacy_strength(manifest)


def save_manifest(manifest: dict) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    manifest["paths"] = PATHS
    manifest["current_engine"] = CURRENT_ENGINE
    manifest["baseline_engine"] = BASELINE_ENGINE
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    _write_status_txt(manifest)


def update_source(name: str, games_file: str | Path, **extra) -> None:
    """Record/update one games source (benchmark name, self-match, etc.)."""
    games_file = Path(games_file)
    manifest = load_manifest()
    manifest.setdefault("sources", {})[name] = {
        "games_file": str(games_file),
        "games": count_games_in_file(games_file),
        "bytes": games_file.stat().st_size if games_file.exists() else 0,
        **extra,
    }
    save_manifest(manifest)


def update_strength_tracker(
    a_wins: int,
    b_wins: int,
    draws: int,
    batch: int | None = None,
    elo: float | None = None,
) -> None:
    """Update running totals for Titanium v15 vs ace-v13-ti-pure baseline."""
    manifest = load_manifest()
    games_file = Path(PATHS["strength_tracker_games"])
    entry = manifest.setdefault("strength_tracker", {})
    entry.update({
        "a_engine": CURRENT_ENGINE,
        "b_engine": BASELINE_ENGINE,
        "a_wins": a_wins,
        "b_wins": b_wins,
        "draws": draws,
        "games_played": a_wins + b_wins + draws,
        "games_file": str(games_file),
        "games_in_file": count_games_in_file(games_file),
    })
    if batch is not None:
        entry["batches_completed"] = batch
    if elo is not None:
        entry["elo_vs_baseline"] = round(elo, 1)
    n = entry["games_played"] or 1
    score = a_wins + 0.5 * draws
    p = score / n
    if 0 < p < 1:
        import math
        entry["elo_vs_baseline"] = round(-400 * math.log10((1 - p) / p), 1)
    save_manifest(manifest)


# Backward-compat alias for run_infinite_benchmark.py imports.
update_v15_vs_ti_pure = update_strength_tracker


def _write_status_txt(manifest: dict) -> None:
    db = Path(PATHS["training_db"])
    db_records = _count_lines(db)
    lines = [
        "=== Quoridor training data ===",
        f"Updated: {manifest.get('updated_at', '?')}",
        f"Current engine: {CURRENT_ENGINE}  |  Baseline: {BASELINE_ENGINE} (JS v13 + O1 movegen)",
        "",
        "KEY FILES (all under training/data/):",
        f"  Training DB (NNUE records):  all_games.db  ({db_records} records, SQLite)",
        f"  Strength tracker games:      v15_vs_ti_pure.games",
        f"  Default self-match games:    self_match_games.games",
        f"  Per-benchmark raw games:     benchmarks/*.games",
        f"  Benchmark results log:       benchmarks_log.jsonl",
        f"  This summary:                STATUS.txt",
        f"  Full manifest:               manifest.json",
        "",
    ]

    v = manifest.get("strength_tracker", {})
    if v:
        n = v.get("games_played", 0)
        aw, bw, d = v.get("a_wins", 0), v.get("b_wins", 0), v.get("draws", 0)
        elo = v.get("elo_vs_baseline")
        elo_s = f"{elo:+.0f}" if elo is not None else "?"
        a_eng = v.get("a_engine", CURRENT_ENGINE)
        b_eng = v.get("b_engine", BASELINE_ENGINE)
        lines += [
            f"STRENGTH TRACKER ({a_eng} vs {b_eng}):",
            f"  Score: {a_eng} {aw} - {bw} {b_eng}  ({d} draws)  /  {n} games",
            f"  Elo vs baseline: {elo_s}",
            f"  Raw games: {v.get('games_in_file', 0)} in v15_vs_ti_pure.games",
            "",
        ]

    sources = manifest.get("sources", {})
    if sources:
        lines.append("OTHER GAME SOURCES:")
        for name, info in sorted(sources.items()):
            lines.append(
                f"  {name}: {info.get('games', 0)} games  ->  {info.get('games_file', '?')}"
            )
        lines.append("")

    STATUS_PATH.write_text("\n".join(lines), encoding="utf-8")
