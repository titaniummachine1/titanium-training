"""Prep guards on coordinator, pool, and streaming loader entry points."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

_TRAINING = Path(__file__).resolve().parents[1]
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

import continuous_pool
import training_coordinator
from continuous_pool import ContinuousPool, PoolConfig, PoolState
from streaming_db_loader import LabelsRepository, iter_db_training_batches, sample_epoch_keys


@pytest.fixture
def prep_only(monkeypatch):
    monkeypatch.setenv("TRAINING_PREP_ONLY", "1")


def _pool_config(tmp_path: Path) -> PoolConfig:
    cur = tmp_path / "cur.bin"
    prev = tmp_path / "prev.bin"
    cur.write_bytes(b"x" * 64)
    prev.write_bytes(b"y" * 64)
    return PoolConfig(
        threads=1,
        time_sec=0.1,
        nodes=1,
        current=cur,
        previous=prev,
        batch_games=1,
        train_after_new_positions=0,
        use_position_trigger=False,
        initial_epoch=False,
        force_epoch=False,
    )


def test_training_coordinator_cli_refuses(prep_only, tmp_path):
    proc = subprocess.run(
        [
            sys.executable,
            str(_TRAINING / "training_coordinator.py"),
            "--poll-sec",
            "1",
            "--epoch-size",
            "100",
        ],
        cwd=str(_TRAINING.parent),
        capture_output=True,
        text=True,
        env={**os.environ, "TRAINING_PREP_ONLY": "1", "PYTHONPATH": str(_TRAINING)},
    )
    assert proc.returncode == 2
    assert "REFUSED" in proc.stderr


def test_continuous_pool_cli_refuses(prep_only):
    proc = subprocess.run(
        [sys.executable, str(_TRAINING / "continuous_pool.py"), "--threads", "1", "--time", "0.1"],
        cwd=str(_TRAINING.parent),
        capture_output=True,
        text=True,
        env={**os.environ, "TRAINING_PREP_ONLY": "1", "PYTHONPATH": str(_TRAINING)},
    )
    assert proc.returncode == 2
    assert "REFUSED" in proc.stderr


def test_coordinator_loop_refuses_before_lock(prep_only, monkeypatch):
    acquire = mock.Mock(return_value=True)
    monkeypatch.setattr(training_coordinator, "acquire_lock", acquire)
    with pytest.raises(SystemExit) as exc:
        training_coordinator.coordinator_loop(
            poll_sec=0.01, epoch_size=100, batch=32, featurize_chunk=64
        )
    assert exc.value.code == 2
    acquire.assert_not_called()


def test_run_training_cycle_refuses_before_subprocess(prep_only, monkeypatch):
    spawn = mock.Mock()
    monkeypatch.setattr(training_coordinator.subprocess, "run", spawn)
    with pytest.raises(SystemExit) as exc:
        training_coordinator.run_training_cycle(
            epoch_size=100, batch=32, featurize_chunk=64
        )
    assert exc.value.code == 2
    spawn.assert_not_called()


def test_continuous_pool_run_refuses_before_workers(prep_only, monkeypatch, tmp_path):
    monkeypatch.setattr(continuous_pool, "BEST", tmp_path / "best.bin")
    (tmp_path / "best.bin").write_bytes(b"z" * 64)
    thread_start = mock.Mock()
    monkeypatch.setattr("threading.Thread.start", thread_start)
    pool = ContinuousPool(_pool_config(tmp_path))
    with pytest.raises(SystemExit) as exc:
        pool.run()
    assert exc.value.code == 2
    thread_start.assert_not_called()


def test_sample_epoch_keys_refuses_before_queue_mutation(prep_only):
    con = mock.Mock()
    with pytest.raises(SystemExit) as exc:
        sample_epoch_keys(con, epoch_size=10)
    assert exc.value.code == 2
    con.execute.assert_not_called()


def test_labels_repository_refuses_before_sqlite_open(prep_only, monkeypatch, tmp_path):
    open_db = mock.Mock()
    monkeypatch.setattr("streaming_db_loader.open_labels_db", open_db)
    with pytest.raises(SystemExit) as exc:
        LabelsRepository(tmp_path / "labels.db")
    assert exc.value.code == 2
    open_db.assert_not_called()


def test_iter_db_training_batches_refuses_before_chunk_work(prep_only):
    repo = mock.Mock()
    with pytest.raises(SystemExit) as exc:
        list(iter_db_training_batches(repo, ["json:a"]))
    assert exc.value.code == 2
    repo.load_labeled_positions.assert_not_called()


def test_prep_off_preserves_coordinator_loop_entry(prep_only, monkeypatch):
    monkeypatch.setenv("TRAINING_PREP_ONLY", "0")
    monkeypatch.setattr(training_coordinator, "acquire_lock", lambda: False)
    rc = training_coordinator.coordinator_loop(
        poll_sec=0.01, epoch_size=100, batch=32, featurize_chunk=64
    )
    assert rc == 0
