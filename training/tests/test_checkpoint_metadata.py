from __future__ import annotations

import struct
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))

from titanium_training.training.trainer import HalfPW, read_net_h
from titanium_training.validation.checkpoint_metadata import (
    ARCHITECTURE_INFERENCE_ERROR,
    CheckpointArchitectureError,
    architecture_bin_for_checkpoint,
    infer_hidden_size,
)
from titanium_training.validation.export_parity import verify_export_parity
import streaming_epoch_validation


def test_infer_hidden_size_from_accepted_h96() -> None:
    weights = ROOT / "training" / "runs" / "v16" / "accepted" / "epoch_0037.bin"
    h = infer_hidden_size(weights_bin=weights)
    assert h == 96


def test_architecture_bin_loads_h96_candidate_as_h96() -> None:
    weights = ROOT / "training" / "runs" / "quarantine" / "cycle_0038_validation_blocked" / "cycle_0038_candidate.bin"
    ckpt = ROOT / "training" / "runs" / "quarantine" / "cycle_0038_validation_blocked" / "ckpt_epoch0001.pt"
    if not weights.is_file() or not ckpt.is_file():
        pytest.skip("preserved h96 candidate missing")
    arch = architecture_bin_for_checkpoint(candidate_bin=weights, checkpoint=ckpt)
    assert arch == weights
    assert HalfPW(arch).h == 96


def test_architecture_bin_loads_h48_candidate_as_h48() -> None:
    weights = ROOT / "training" / "runs" / "v16" / "accepted" / "epoch_0036.bin"
    ckpt = ROOT / "training" / "runs" / "v16" / "accepted" / "epoch_0036.pt"
    if not weights.is_file() or not ckpt.is_file():
        pytest.skip("h48 accepted checkpoint missing")
    arch = architecture_bin_for_checkpoint(candidate_bin=weights, checkpoint=ckpt)
    assert arch == weights
    assert infer_hidden_size(weights_bin=arch, checkpoint=ckpt) == 48
    assert HalfPW(arch).h == 48


def test_missing_architecture_rejects_without_default() -> None:
    with pytest.raises(CheckpointArchitectureError, match=ARCHITECTURE_INFERENCE_ERROR):
        infer_hidden_size()


def test_ambiguous_architecture_rejects_without_parent_fallback() -> None:
    h48 = ROOT / "training" / "runs" / "v16" / "accepted" / "epoch_0036.bin"
    h96_ckpt = ROOT / "training" / "runs" / "quarantine" / "cycle_0038_validation_blocked" / "ckpt_epoch0001.pt"
    h96_parent = ROOT / "training" / "runs" / "v16" / "accepted" / "epoch_0037.bin"
    if not h48.is_file() or not h96_ckpt.is_file() or not h96_parent.is_file():
        pytest.skip("architecture mismatch fixtures missing")
    with pytest.raises(CheckpointArchitectureError, match=ARCHITECTURE_INFERENCE_ERROR):
        architecture_bin_for_checkpoint(
            candidate_bin=h48,
            checkpoint=h96_ckpt,
            parent_bin=h48,
        )
    assert architecture_bin_for_checkpoint(
        candidate_bin=h48,
        checkpoint=h96_ckpt,
        parent_bin=h96_parent,
    ) == h96_parent


def test_export_parity_uses_candidate_architecture(tmp_path: Path) -> None:
    src = ROOT / "training" / "runs" / "v16" / "accepted" / "epoch_0037.bin"
    ckpt = ROOT / "training" / "runs" / "quarantine" / "cycle_0038_validation_blocked" / "ckpt_epoch0001.pt"
    if not ckpt.is_file():
        pytest.skip("preserved checkpoint missing")
    export = tmp_path / "candidate.bin"
    parent = tmp_path / "parent.bin"
    export.write_bytes(src.read_bytes())
    parent.write_bytes(src.read_bytes())
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = HalfPW(src)
    model.load_state_dict(payload["model"])
    model.save_weights(export)
    result = verify_export_parity(ckpt, export)
    assert result.checkpoint_python_vs_export_python


def test_streaming_validation_passes_candidate_paths_to_helpers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    candidate = tmp_path / "candidate_h96.bin"
    checkpoint = tmp_path / "candidate_h96.pt"
    parent = tmp_path / "parent_h96.bin"
    candidate.write_bytes((ROOT / "training" / "runs" / "v16" / "accepted" / "epoch_0037.bin").read_bytes())
    parent.write_bytes(candidate.read_bytes())
    checkpoint.write_bytes(
        (ROOT / "training" / "runs" / "quarantine" / "cycle_0038_validation_blocked" / "ckpt_epoch0001.pt").read_bytes()
    )
    calls: dict[str, tuple[Path, ...]] = {}

    class DummyParity:
        passed = True
        max_parity_error = 0

    def fake_verify_export_parity(ckpt: Path, export_path: Path):
        calls["parity"] = (ckpt, export_path)
        return DummyParity()

    def fake_opening_sanity(weights_path: Path):
        calls["opening"] = (weights_path,)
        return ("e2", "e8", "e3", "e7")

    def fake_match_candidate_vs_parent(*, candidate_bin: Path, parent_bin: Path, games: int, **_kwargs):
        calls["match"] = (candidate_bin, parent_bin)
        return {"games": games, "wins": games, "draws": 0, "losses": 0, "score": 1.0}

    monkeypatch.setattr(streaming_epoch_validation, "verify_export_parity", fake_verify_export_parity)
    monkeypatch.setattr(streaming_epoch_validation, "assert_opening_sanity", fake_opening_sanity)
    monkeypatch.setattr(streaming_epoch_validation, "_match_candidate_vs_parent", fake_match_candidate_vs_parent)
    monkeypatch.setattr(streaming_epoch_validation, "_search_bench", lambda _weights: {"skipped": True})
    monkeypatch.setattr(streaming_epoch_validation, "previous_accepted", lambda: None)
    monkeypatch.setenv("TRAINING_PREP_ONLY", "0")

    report = streaming_epoch_validation.run_epoch_validation(
        checkpoint=checkpoint,
        candidate_bin=candidate,
        previous_bin=parent,
        short_games=2,
    )

    assert report["passed"]
    assert calls["parity"] == (checkpoint, candidate)
    assert calls["opening"] == (candidate,)
    assert calls["match"] == (candidate, parent)
