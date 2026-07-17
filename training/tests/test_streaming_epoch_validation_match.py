from pathlib import Path

import engine_session
import self_play_overnight
from streaming_epoch_validation import _match_candidate_vs_parent


class _FakeSession:
    created: list[Path] = []
    sync_ok = True

    def __init__(self, _engine: str, weights: Path | None, threads: int = 1):
        assert threads == 1
        assert weights is not None
        self.created.append(Path(weights))

    def sync(self, _moves: list[str]) -> bool:
        return self.sync_ok

    def alive(self) -> bool:
        return True

    def go(self, _time_sec: float) -> str | None:
        return "e2"

    def close(self) -> None:
        pass


def _weights(tmp_path):
    candidate = tmp_path / "candidate.bin"
    baseline = tmp_path / "baseline.bin"
    candidate.write_bytes(b"candidate")
    baseline.write_bytes(b"baseline")
    return candidate, baseline


def test_validation_match_uses_colour_swapped_pairs_and_counts_completion(monkeypatch, tmp_path):
    candidate, baseline = _weights(tmp_path)
    _FakeSession.created = []
    _FakeSession.sync_ok = True
    monkeypatch.setattr(engine_session, "EngineSession", _FakeSession)

    result = _match_candidate_vs_parent(
        candidate_bin=candidate,
        parent_bin=baseline,
        games=2,
        max_ply=16,
        concurrency=1,
    )

    assert result["games"] == 2
    assert result["aborted_games"] == 0
    assert result["paired_openings"] is True
    assert _FakeSession.created.count(candidate) == 2
    assert _FakeSession.created.count(baseline) == 2


def test_validation_match_fails_closed_when_a_session_cannot_sync(monkeypatch, tmp_path):
    candidate, baseline = _weights(tmp_path)
    _FakeSession.created = []
    _FakeSession.sync_ok = False
    monkeypatch.setattr(engine_session, "EngineSession", _FakeSession)

    result = _match_candidate_vs_parent(
        candidate_bin=candidate,
        parent_bin=baseline,
        games=2,
        max_ply=16,
        concurrency=1,
    )

    assert result["games"] == 0
    assert result["aborted_games"] == 2
    assert result["aborted_reasons"] == {"session_sync_failed": 2}


def test_validation_match_rejects_a_ply_limit_that_never_searches(monkeypatch, tmp_path):
    candidate, baseline = _weights(tmp_path)
    monkeypatch.setattr(engine_session, "EngineSession", _FakeSession)

    result = _match_candidate_vs_parent(
        candidate_bin=candidate,
        parent_bin=baseline,
        games=2,
        max_ply=14,
        concurrency=1,
    )

    assert result["games"] == 0
    assert result["aborted_reasons"] == {"max_ply_not_beyond_opening": 2}


def test_validation_match_stops_after_a_terminal_move(monkeypatch, tmp_path):
    candidate, baseline = _weights(tmp_path)
    _FakeSession.sync_ok = True
    go_calls = []

    def go_after_recording(self, _time_sec: float) -> str:
        go_calls.append(self)
        return "e2"

    monkeypatch.setattr(engine_session, "EngineSession", _FakeSession)
    monkeypatch.setattr(_FakeSession, "go", go_after_recording)
    # Winning immediately after the first searched move must complete each
    # colour-swapped game; the evaluator must not keep searching to max_ply.
    monkeypatch.setattr(self_play_overnight, "check_winner", lambda _moves: 0)

    result = _match_candidate_vs_parent(
        candidate_bin=candidate,
        parent_bin=baseline,
        games=2,
        max_ply=20,
        concurrency=1,
    )

    assert result["games"] == 2
    assert result["wins"] == 1
    assert result["losses"] == 1
    assert result["draws"] == 0
    assert len(go_calls) == 2
