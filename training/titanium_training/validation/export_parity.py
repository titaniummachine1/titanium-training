"""Parity checks for exported HalfPW value nets (Python round-trip + engine)."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from titanium_training.models.halfpw import Net, forward
from titanium_training.paths import ENGINE_BIN, REPO_ROOT
from titanium_training.training.trainer import HalfPW
from titanium_training.validation.checkpoint_metadata import (
    CheckpointArchitectureError,
    architecture_bin_for_checkpoint,
    infer_hidden_size,
)

# Fixed positions for exported-net parity (mid-game, diverse geometry).
PARITY_POSITIONS = [
    [],
    ["e2", "e8", "e3", "e7", "d3h", "f5v"],
    ["e2", "e8", "e3", "e7", "e4", "e6", "a3h", "d4v"],
    ["e2", "e8", "d2", "f8", "c4h", "g5h"],
    ["e2", "e8", "e3", "e7", "d3h", "f5v", "c2h"],
    ["e2", "e8", "e3", "e7", "e4", "e6", "e5", "d6", "f4h"],
]
MAX_ALLOWED_DIFF_CP = 1


@dataclass
class ExportParityResult:
    checkpoint_python_vs_export_python: bool
    export_python_vs_engine: bool
    max_parity_error: int
    details: list[str]

    @property
    def passed(self) -> bool:
        return self.checkpoint_python_vs_export_python and self.export_python_vs_engine


def _engine_eval(moves: list[str], *, weights_path: Path) -> dict[str, Any]:
    env = os.environ.copy()
    env["TITANIUM_NET_WEIGHTS_PATH"] = str(weights_path.resolve())
    out = subprocess.run(
        [str(ENGINE_BIN), "eval", *moves, "--json"],
        capture_output=True,
        text=True,
        check=True,
        cwd=str(REPO_ROOT),
        env=env,
    )
    return json.loads(out.stdout.strip())


def verify_export_parity(
    checkpoint: Path,
    export_path: Path,
    *,
    positions: list[list[str]] | None = None,
) -> ExportParityResult:
    positions = positions or PARITY_POSITIONS
    details: list[str] = []
    max_err = 0

    payload = torch.load(checkpoint, weights_only=False, map_location="cpu")
    if not isinstance(payload, dict) or "model" not in payload:
        raise CheckpointArchitectureError(f"checkpoint missing model state: {checkpoint}")
    state_dict = payload["model"]
    arch_bin = architecture_bin_for_checkpoint(
        candidate_bin=export_path,
        checkpoint=checkpoint,
    )
    ckpt_h = infer_hidden_size(weights_bin=arch_bin, state_dict=state_dict)
    ckpt_model = HalfPW(arch_bin)
    ckpt_model.load_state_dict(state_dict)
    ckpt_model.eval()

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        ckpt_bin = Path(tmp.name)
    try:
        ckpt_model.save_weights(ckpt_bin)
        ckpt_net = Net.load(ckpt_bin)
    finally:
        ckpt_bin.unlink(missing_ok=True)

    export_net = Net.load(export_path)

    py_export_ok = True
    eng_ok = True

    for moves in positions:
        rec = _engine_eval(moves, weights_path=arch_bin)
        ckpt_cp = int(forward(ckpt_net, rec))
        export_cp = int(forward(export_net, rec))
        err_ckpt = abs(ckpt_cp - export_cp)
        max_err = max(max_err, err_ckpt)
        if err_ckpt > MAX_ALLOWED_DIFF_CP:
            py_export_ok = False
            details.append(f"ckpt vs export py DIFF {moves}: {ckpt_cp} vs {export_cp}")

        eng_rec = _engine_eval(moves, weights_path=export_path)
        eng_cp = int(eng_rec["eval"])
        err_eng = abs(export_cp - eng_cp)
        max_err = max(max_err, err_eng)
        if err_eng > MAX_ALLOWED_DIFF_CP:
            eng_ok = False
            details.append(f"export py vs engine DIFF {moves}: {export_cp} vs {eng_cp}")

    return ExportParityResult(
        checkpoint_python_vs_export_python=py_export_ok,
        export_python_vs_engine=eng_ok,
        max_parity_error=max_err,
        details=details,
    )
