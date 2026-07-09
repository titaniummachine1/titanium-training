from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))

from position_usage_db import claim_all_pending, pending_new_eligible, release_pending_claim
import training_coordinator
from training_coordinator import should_use_full_active_epoch, training_cycle_consumed


def test_release_pending_claim_restores_counter(tmp_path: Path) -> None:
    db = tmp_path / "labels.db"
    con = sqlite3.connect(db)
    con.executescript(
        """
        CREATE TABLE training_trigger_state (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            pending_new_eligible INTEGER NOT NULL DEFAULT 0,
            claimed_total INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL
        );
        INSERT INTO training_trigger_state(id, pending_new_eligible, claimed_total, updated_at)
        VALUES (1, 5000, 0, 't');
        """
    )
    claim = claim_all_pending(con)
    assert claim["claimed_count"] == 5000
    assert pending_new_eligible(con) == 0
    pending = release_pending_claim(con, claim["claimed_count"])
    assert pending == 5000
    con.close()


def test_training_cycle_consumed_only_on_finished_epochs() -> None:
    assert training_cycle_consumed({"returncode": 0, "decision": "accepted"})
    assert training_cycle_consumed({"returncode": 0, "decision": "quarantined"})
    assert not training_cycle_consumed({"returncode": 1, "decision": "train_failed"})
    assert not training_cycle_consumed({"returncode": 0, "decision": "lock_busy"})
    assert not training_cycle_consumed({"returncode": 0, "decision": "no_export"})


def test_repair_mode_does_not_force_full_active_epoch(monkeypatch) -> None:
    monkeypatch.setattr(
        training_coordinator,
        "load_chain",
        lambda: {"epochs": [{"epoch": 0}, {"epoch": 1}, {"epoch": 2}]},
    )
    monkeypatch.setenv("STREAM_REPAIR_MODE", "1")
    monkeypatch.delenv("STREAM_FULL_ACTIVE_EPOCH", raising=False)
    assert not should_use_full_active_epoch({})

    monkeypatch.setenv("STREAM_FULL_ACTIVE_EPOCH", "1")
    assert should_use_full_active_epoch({})
