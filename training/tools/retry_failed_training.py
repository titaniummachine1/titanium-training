#!/usr/bin/env python3
"""Re-arm the games trigger and pending claim after train_failed.

Use when a trainer crash consumed the trigger window without producing an epoch.
Does not change label/weight formulas — only coordinator bookkeeping.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_TRAINING = Path(__file__).resolve().parents[1]
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

from db_import import LABELS_DB_PATH
from position_usage_db import open_labels_db, pending_new_eligible, release_pending_claim
from streaming_checkpoint_chain import restore_candidate_from_last_accepted
from training_coordinator import (
    GAMES_TRIGGER_THRESHOLD,
    STATE_FILE,
    cleanup_incomplete_training_artifacts,
    games_db_max_rowid,
    read_json,
    utc_now,
    write_json,
)


def main() -> int:
    from prep_guard import guard_real_work

    guard_real_work("optimizer_training", detail="retry_failed_training.py")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--claimed-count",
        type=int,
        default=0,
        help="Pending positions to restore (default: last failed claim count)",
    )
    ap.add_argument("--no-release-claim", action="store_true")
    ap.add_argument("--no-restore-weights", action="store_true")
    ap.add_argument("--no-cleanup", action="store_true")
    args = ap.parse_args()

    state = read_json(STATE_FILE)
    last = state.get("last_training_result") or {}
    claim = last.get("claim") or state.get("last_claim") or {}
    claimed = int(args.claimed_count or claim.get("claimed_count") or 0)

    games_now = games_db_max_rowid()
    state["last_train_games_rowid"] = max(0, games_now - GAMES_TRIGGER_THRESHOLD)
    state["games_since_last_train"] = games_now - int(state["last_train_games_rowid"])
    state.pop("train_failed_retry_after", None)
    state["state"] = "IDLE"
    state["updated_at"] = utc_now()

    if not args.no_restore_weights:
        restored = restore_candidate_from_last_accepted()
        print(f"restored parent weights: {restored}")

    if not args.no_cleanup:
        cycle = (last.get("pre_train_snapshot") or {}).get("cycle")
        removed = cleanup_incomplete_training_artifacts(
            cycle_num=int(cycle) if cycle is not None else None,
        )
        for path in removed:
            print(f"removed incomplete artifact: {path}")

    if not args.no_release_claim and claimed > 0:
        con = open_labels_db(LABELS_DB_PATH)
        try:
            pending = release_pending_claim(con, claimed)
            print(f"released pending claim={claimed} (pending_new_eligible now {pending})")
        finally:
            con.close()
    else:
        con = open_labels_db(LABELS_DB_PATH)
        try:
            print(f"pending_new_eligible={pending_new_eligible(con)} (claim not released)")
        finally:
            con.close()

    write_json(STATE_FILE, state)
    print(
        json.dumps(
            {
                "games_now": games_now,
                "last_train_games_rowid": state["last_train_games_rowid"],
                "games_since_last_train": state["games_since_last_train"],
                "games_trigger_threshold": GAMES_TRIGGER_THRESHOLD,
                "ready_for_immediate_retry": state["games_since_last_train"]
                >= GAMES_TRIGGER_THRESHOLD,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
