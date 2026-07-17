from __future__ import annotations

import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKER = HERE / "ka_nn_batch_worker.mjs"


def _evaluate(backend: str) -> dict:
    request = {
        "id": "parity",
        "verify_replay_legality": True,
        "positions": [
            {"id": "start", "moves": []},
            {"id": "black", "moves": ["e2"]},
            {"id": "opening", "moves": ["e2", "e8", "e3", "e7"]},
            {"id": "wall", "moves": ["a1h"]},
        ],
    }
    proc = subprocess.run(
        [
            "node",
            str(WORKER),
            "--backend",
            backend,
            "--model-batch",
            "16",
            "--batch-max",
            "16",
        ],
        input=json.dumps(request) + "\n",
        text=True,
        encoding="utf-8",
        capture_output=True,
        timeout=60,
        check=True,
    )
    return json.loads(proc.stdout.strip())


def test_native_cpu_matches_wasm() -> None:
    native = _evaluate("cpu")
    wasm = _evaluate("wasm")
    assert native["ok"] and wasm["ok"]
    assert not native["rejected"] and not wasm["rejected"]
    for actual, expected in zip(native["rows"], wasm["rows"], strict=True):
        assert actual["id"] == expected["id"]
        assert actual["side_to_move"] == expected["side_to_move"]
        assert abs(actual["value_black"] - expected["value_black"]) < 2e-6
        assert abs(actual["value_stm"] - expected["value_stm"]) < 2e-6
        assert actual["policy"]["best_move"] == expected["policy"]["best_move"]
        assert abs(actual["policy"]["confidence"] - expected["policy"]["confidence"]) < 2e-6
