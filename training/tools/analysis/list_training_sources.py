"""List all training data sources with counts for user filtering decisions."""
from __future__ import annotations

import sqlite3
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
TRAINING = ROOT / "training"
GAMES_DB = TRAINING / "data" / "canonical" / "games.db"
TEACHER = TRAINING / "data" / "teacher_dataset_good"


def main() -> None:
    print("=" * 60)
    print("GAMES.DB — raw game records")
    print("=" * 60)
    if GAMES_DB.is_file():
        con = sqlite3.connect(GAMES_DB)
        for src, cnt in con.execute(
            "SELECT source, COUNT(*) FROM games GROUP BY source ORDER BY COUNT(*) DESC"
        ):
            print(f"  {src}: {cnt:,} games")
            samples = [
                r[0]
                for r in con.execute(
                    "SELECT game_id FROM games WHERE source=? LIMIT 2", (src,)
                )
            ]
            print(f"    sample ids: {samples}")
        con.close()
    else:
        print("  (missing)")

    print()
    print("=" * 60)
    print("LABELS.DB — outcome labels by source")
    print("=" * 60)
    labels_db = TRAINING / "data" / "canonical" / "labels.db"
    if labels_db.is_file():
        con = sqlite3.connect(labels_db)
        for src, cnt, avg in con.execute(
            "SELECT source, COUNT(*), AVG(value_stm) FROM labels GROUP BY source ORDER BY COUNT(*) DESC"
        ):
            print(f"  {src}: {cnt:,} labels  avg_value={avg:+.4f}")
        con.close()

    print()
    print("=" * 60)
    print("TEACHER_DATASET_GOOD — source_cohort (training positions)")
    print("=" * 60)
    labels_path = TEACHER / "labels" / "part-00000.parquet"
    if labels_path.is_file():
        tbl = pq.read_table(labels_path, columns=["source_cohort"])
        cohorts = Counter(str(x or "(empty)") for x in tbl.column("source_cohort").to_pylist())
        for k, v in sorted(cohorts.items(), key=lambda x: -x[1]):
            print(f"  {k}: {v:,} label rows")

        obs_path = TEACHER / "observations" / "part-00000.parquet"
        if obs_path.is_file():
            obs = pq.read_table(
                obs_path, columns=["source_cohort", "observation_count"]
            )
            oc: Counter[str] = Counter()
            for sc, cnt in zip(
                obs.column("source_cohort").to_pylist(),
                obs.column("observation_count").to_pylist(),
            ):
                oc[str(sc or "(empty)")] += int(cnt)
            print()
            print("  observations (weighted):")
            for k, v in sorted(oc.items(), key=lambda x: -x[1]):
                print(f"    {k}: {v:,}")

    print()
    print("=" * 60)
    print("SOURCE CLASSIFICATION (for filtering)")
    print("=" * 60)
    our_engine = [
        ("oracle_selfplay", "Oracle cloud — Titanium engine self-play"),
        ("overnight_selfplay", "Local overnight — Titanium vs Titanium"),
        ("overnight_mixed", "Local overnight — mixed Titanium tiers"),
        ("selfplay_train", "Local self-play for training loop"),
        ("selfplay_verify", "Local self-play for verification"),
    ]
    external = [
        ("friend_selfplay", "External friend corpus (KaAiData iter 1-20)"),
        ("zeroink", "Zero-ink positions (no engine games, synthetic?)"),
        ("v7_jsonl", "Legacy JSONL corpus import"),
        ("oracle-gen-*", "Oracle game factory remote gen (check quality)"),
    ]
    print("\n  OUR ENGINE (likely trash if model is random):")
    for name, desc in our_engine:
        print(f"    - {name}: {desc}")
    print("\n  POTENTIALLY EXTERNAL / HIGHER QUALITY:")
    for name, desc in external:
        print(f"    - {name}: {desc}")


if __name__ == "__main__":
    main()
