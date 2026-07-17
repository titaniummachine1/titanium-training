"""Tri-path HalfPW parity: trainer vs halfpw vs Rust engine.

Intermediate tensors must match between Python paths within 1e-5.
Final centipawn score must match Rust within 1 cp.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

TRAINING_ROOT = Path(__file__).resolve().parents[1]
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from titanium_training.models.eval_forward import record_to_trainer_batch
from titanium_training.models.halfpw import Net, forward_trace
from titanium_training.paths import ENGINE_BIN, REPO_ROOT, WEIGHTS_BIN
from titanium_training.training.trainer import HalfPW

POSITIONS = [
    ["e2", "e8", "e3", "e7", "d3h", "f5v"],
    ["e2", "e8", "e3", "e7", "e4", "e6", "a3h", "d4v"],
    ["e2", "e8", "d2", "f8", "c4h", "g5h"],
    ["e2", "e8", "e3", "e7", "d3h", "f5v", "c2h"],
    ["e2", "e8", "e3", "e7", "e4", "e6", "c6h", "f3v", "b5h"],
    ["e2", "e8", "e3", "e7", "e4", "e6", "e5", "d6", "f4h"],
]

# Float32 trainer vs f64 reference: scalar/neural tails accumulate ws/w2 rounding.
MAX_INTERMEDIATE_ABS = 1e-5
MAX_SCALAR_OUT_ABS = 2e-5
MAX_NEURAL_OUT_ABS = 3e-5
MAX_FINAL_CP = 1


def _engine_dump(moves: list[str], *, parity_trace: bool = False) -> dict:
    cmd = [str(ENGINE_BIN), "eval", *moves]
    if parity_trace:
        cmd.append("--parity-trace")
    else:
        cmd.append("--json")
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    return json.loads(out)


def _max_abs(a: float, b: float) -> float:
    return abs(float(a) - float(b))


def _compare_scalar_inputs(hp: dict, tr: dict) -> float:
    worst = 0.0
    for key in ("d_me", "d_opp", "w_me", "w_opp", "pd", "wd", "width_opp"):
        worst = max(worst, _max_abs(hp.scalar_inputs[key], tr.scalar_inputs[key]))
    return worst


def _compare_vectors(a: list[float], b: list[float]) -> float:
    assert len(a) == len(b)
    return max(abs(float(x) - float(y)) for x, y in zip(a, b))


def _compare_traces(hp, tr, label: str) -> dict:
    report = {
        "label": label,
        "scalar_inputs": _compare_scalar_inputs(hp, tr),
        "scalar_out": _max_abs(hp.scalar_out, tr.scalar_out),
        "route_out": _max_abs(hp.route_out, tr.route_out),
        "cat_out": _max_abs(hp.cat_out, tr.cat_out),
        "width_contrib": _max_abs(hp.width_contrib, tr.width_contrib),
        "wall_acc": _compare_vectors(hp.wall_acc, tr.wall_acc),
        "hidden_pre": _compare_vectors(hp.hidden_pre, tr.hidden_pre),
        "hidden_clip": _compare_vectors(hp.hidden_clip, tr.hidden_clip),
        "neural_out": _max_abs(hp.neural_out, tr.neural_out),
        "final_cp": abs(int(hp.final_cp) - int(tr.final_cp)),
    }
    return report


def _rust_trace_to_compare(rust: dict):
    """Normalize Rust JSON trace for comparison with EvalTrace."""

    class _T:
        pass

    si = rust["scalar_inputs"]
    t = _T()
    t.scalar_inputs = {k: float(si[k]) for k in si}
    t.scalar_out = float(rust["scalar_out"])
    t.route_out = float(rust["route_out"])
    t.cat_out = float(rust["cat_out"])
    t.width_contrib = float(rust["width_contrib"])
    t.wall_acc = [float(x) for x in rust["wall_acc"]]
    t.hidden_pre = [float(x) for x in rust["hidden_pre"]]
    t.hidden_clip = [float(x) for x in rust["hidden_clip"]]
    t.neural_out = float(rust["neural_out"])
    t.final_cp = int(rust["eval"])
    return t


@pytest.fixture(scope="module")
def halfpw_net() -> Net:
    return Net.load(WEIGHTS_BIN)


@pytest.fixture(scope="module")
def trainer_model() -> HalfPW:
    return HalfPW(WEIGHTS_BIN)


@pytest.mark.parametrize("moves", POSITIONS, ids=lambda m: "-".join(m))
def test_tri_path_parity(moves: list[str], halfpw_net: Net, trainer_model: HalfPW):
    if not ENGINE_BIN.is_file():
        pytest.skip(f"missing engine binary: {ENGINE_BIN}")

    rec = _engine_dump(moves, parity_trace=False)
    rust = _engine_dump(moves, parity_trace=True)

    hp = forward_trace(halfpw_net, rec, normed=False)
    batch = record_to_trainer_batch(rec)
    tr = trainer_model.forward_trace(batch)[0]
    rs = _rust_trace_to_compare(rust)

    py_report = _compare_traces(hp, tr, "halfpw_vs_trainer")
    rs_report = _compare_traces(hp, rs, "halfpw_vs_rust")

    failures = []
    for key, val in py_report.items():
        if key in ("label", "final_cp"):
            continue
        if key == "scalar_out":
            limit = MAX_SCALAR_OUT_ABS
        elif key == "neural_out":
            limit = MAX_NEURAL_OUT_ABS
        else:
            limit = MAX_INTERMEDIATE_ABS
        if val > limit:
            failures.append(f"halfpw_vs_trainer {key} max_abs={val}")
    if py_report["final_cp"] != 0:
        failures.append(f"halfpw_vs_trainer final_cp diff={py_report['final_cp']}")
    if rs_report["final_cp"] > MAX_FINAL_CP:
        failures.append(f"halfpw_vs_rust final_cp diff={rs_report['final_cp']}")

    if failures:
        detail = {
            "moves": moves,
            "halfpw_final_cp": hp.final_cp,
            "trainer_final_cp": tr.final_cp,
            "rust_final_cp": rs.final_cp,
            "halfpw_vs_trainer": py_report,
            "halfpw_vs_rust": rs_report,
            "failures": failures,
        }
        pytest.fail(json.dumps(detail, indent=2))

    assert py_report["final_cp"] == 0
    assert rs_report["final_cp"] <= MAX_FINAL_CP


def test_parity_check_still_passes():
    script = REPO_ROOT / "training" / "titanium_training" / "validation" / "parity_check.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
