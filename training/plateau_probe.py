"""Learning-health probes — eval drift and root-move delta (not just Elo).

Elo can flatline while the net still recalibrates (search absorbs small eval shifts).
These probes track whether the model still changes *what it thinks is good*.

  eval_drift       — mean |eval_new - eval_old| on fixed probe positions
  move_change_rate — fraction of probe roots whose bestmove changed vs last deploy

Trainer probe (after micro-train): halfpw on engine features + checkpoint weights.
Engine probe (after deploy+rebuild): titanium eval + genmove on embedded weights.

State: training/data/nnue_eval_probe.json
Log:   training/data/nnue_train.log (via nnue_log)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "engine" / "target" / "release" / "titanium.exe"
PROBE_PATH = ROOT / "training" / "data" / "nnue_eval_probe.json"
CKPT_WEIGHTS = ROOT / "training" / "checkpoints" / "net_weights_best.bin"

# Mid-game positions — pure net eval path (parity_check set).
PROBE_POSITIONS = [
    ["e2", "e8", "e3", "e7", "d3h", "f5v"],
    ["e2", "e8", "e3", "e7", "e4", "e6", "a3h", "d4v"],
    ["e2", "e8", "d2", "f8", "c4h", "g5h"],
    ["e2", "e8", "e3", "e7", "d3h", "f5v", "c2h"],
    ["e2", "e8", "e3", "e7", "e4", "e6", "c6h", "f3v", "b5h"],
    ["e2", "e8", "e3", "e7", "e4", "e6", "e5", "d6", "f4h"],
]
# Shorter think budget for root-move probes (3 positions).
MOVE_PROBE_POSITIONS = PROBE_POSITIONS[:3]
MOVE_PROBE_TIME_SEC = 0.25

# Safe overnight promote gate (candidate vs deployed, before rebuild ships).
PROMOTE_MIN_DRIFT_CP = float(os.environ.get("NNUE_PROMOTE_MIN_DRIFT_CP", "2"))
PROMOTE_MIN_MOVE_RATE = float(os.environ.get("NNUE_PROMOTE_MIN_MOVE_RATE", "0.05"))
MEANINGFUL_MOVE_RATE = 0.10  # log hint: search materially reordered

DEPLOYED_WEIGHTS = ROOT / "engine" / "src" / "acev13" / "net_weights.bin"

TRAINER_STALE_RUNS = 3
TRAINER_DRIFT_STALE_CP = PROMOTE_MIN_DRIFT_CP


def _load_state() -> dict:
    try:
        return json.loads(PROBE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"trainer_history": [], "deploy_history": [], "last_trainer": None, "last_deploy": None}


def _save_state(state: dict) -> None:
    PROBE_PATH.parent.mkdir(parents=True, exist_ok=True)
    state["trainer_history"] = state.get("trainer_history", [])[-128:]
    state["deploy_history"] = state.get("deploy_history", [])[-64:]
    PROBE_PATH.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def _engine_eval_batch() -> list[int]:
    stdin_text = "\n".join(" ".join(m) for m in PROBE_POSITIONS) + "\n"
    out = subprocess.run(
        [str(BIN), "eval-batch"],
        input=stdin_text.encode("utf-8"),
        capture_output=True,
        check=True,
    )
    evals = []
    for line in out.stdout.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        evals.append(int(json.loads(line)["eval"]))
    return evals


def _engine_root_moves() -> list[str]:
    roots = []
    for moves in MOVE_PROBE_POSITIONS:
        cmd = [
            str(BIN), "genmove", "--engine", "titanium-v15",
            "--time", str(MOVE_PROBE_TIME_SEC), *moves,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        best = "(none)"
        for line in (proc.stdout + proc.stderr).splitlines():
            line = line.strip()
            if line.startswith("bestmove "):
                best = line.split(" ", 1)[1].strip()
                break
        roots.append(best)
    return roots


def _halfpw_evals(weights_path: Path) -> list[int]:
    sys.path.insert(0, str(ROOT / "training"))
    from halfpw import Net, forward

    net = Net.load(weights_path)
    stdin_text = "\n".join(" ".join(m) for m in PROBE_POSITIONS) + "\n"
    out = subprocess.run(
        [str(BIN), "eval-batch"],
        input=stdin_text.encode("utf-8"),
        capture_output=True,
        check=True,
    )
    evals = []
    for line in out.stdout.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        evals.append(forward(net, json.loads(line)))
    return evals


def candidate_vs_deployed_drift() -> float:
    """Mean |eval_candidate - eval_deployed| on probe positions (halfpw, no rebuild)."""
    if not CKPT_WEIGHTS.exists() or not DEPLOYED_WEIGHTS.exists():
        return 0.0
    try:
        deployed = _halfpw_evals(DEPLOYED_WEIGHTS)
        candidate = _halfpw_evals(CKPT_WEIGHTS)
    except Exception:
        return 0.0
    return _drift(candidate, deployed)


def evaluate_promotion_gate(*, force: bool = False) -> tuple[bool, str, bool]:
    """Safe deploy rule: (drift > 2cp OR move > 5%) and caller handles Elo separately.

    Returns (promote, reason, already_staged).
    already_staged=True when borderline move probe copied weights + rebuilt in-process.
    """
    if force:
        return True, "forced promote", False

    drift = candidate_vs_deployed_drift()
    if drift > PROMOTE_MIN_DRIFT_CP:
        return True, f"promote drift {drift:.1f}cp > {PROMOTE_MIN_DRIFT_CP}cp", False

    if not CKPT_WEIGHTS.exists() or not DEPLOYED_WEIGHTS.exists() or not BIN.exists():
        return False, f"hold drift {drift:.1f}cp (missing weights/binary)", False

    state = _load_state()
    prev_roots = (state.get("last_deploy") or {}).get("roots")
    if not prev_roots:
        return True, f"promote first deploy (drift {drift:.1f}cp)", False

    backup = DEPLOYED_WEIGHTS.read_bytes()
    try:
        shutil.copy2(CKPT_WEIGHTS, DEPLOYED_WEIGHTS)
        from nnue_guards import rebuild_titanium_release, net_weights_size_ok, HALFPW_WEIGHT_BYTES

        if not net_weights_size_ok(DEPLOYED_WEIGHTS):
            DEPLOYED_WEIGHTS.write_bytes(backup)
            return False, f"hold bad candidate size != {HALFPW_WEIGHT_BYTES}B", False

        ok, rebuild_msg = rebuild_titanium_release()
        if not ok:
            DEPLOYED_WEIGHTS.write_bytes(backup)
            return False, f"hold rebuild failed: {rebuild_msg}", False

        roots = _engine_root_moves()
        move_rate = _move_change_rate(roots, prev_roots)
        if move_rate > PROMOTE_MIN_MOVE_RATE:
            hint = " (material)" if move_rate >= MEANINGFUL_MOVE_RATE else ""
            return True, f"promote move {move_rate:.0%} > {PROMOTE_MIN_MOVE_RATE:.0%}{hint}", True

        DEPLOYED_WEIGHTS.write_bytes(backup)
        rebuild_titanium_release()
        return False, f"hold drift {drift:.1f}cp move {move_rate:.0%}", False
    except Exception as e:
        if DEPLOYED_WEIGHTS.exists():
            DEPLOYED_WEIGHTS.write_bytes(backup)
        try:
            from nnue_guards import rebuild_titanium_release
            rebuild_titanium_release()
        except Exception:
            pass
        return False, f"hold probe error: {e}", False


def _trainer_eval_batch() -> list[int]:
    weights = CKPT_WEIGHTS if CKPT_WEIGHTS.exists() else DEPLOYED_WEIGHTS
    return _halfpw_evals(weights)


def _drift(a: list[int], b: list[int] | None) -> float:
    if not b or len(a) != len(b):
        return 0.0
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


def _move_change_rate(a: list[str], b: list[str] | None) -> float:
    if not b or len(a) != len(b):
        return 0.0
    return sum(1 for x, y in zip(a, b) if x != y) / len(a)


def _classify_trainer(drift: float, history: list) -> str:
    recent = [h.get("eval_drift_cp", 999.0) for h in history[-TRAINER_STALE_RUNS:]]
    if len(recent) >= TRAINER_STALE_RUNS and all(d < TRAINER_DRIFT_STALE_CP for d in recent):
        return "trainer_plateau (eval drift < {:.0f} cp × {} runs)".format(
            TRAINER_DRIFT_STALE_CP, TRAINER_STALE_RUNS
        )
    if drift >= TRAINER_DRIFT_STALE_CP:
        return "trainer_learning (eval drift {:.1f} cp)".format(drift)
    return "trainer_quiet (drift {:.1f} cp — search may hide Elo)".format(drift)


def record_trainer_probe(*, game_id: int | None = None) -> dict | None:
    """After micro-train: checkpoint weights vs last trainer snapshot."""
    if not BIN.exists():
        return None
    try:
        evals = _trainer_eval_batch()
    except Exception as e:
        from nnue_guards import nnue_log
        nnue_log(f"probe trainer failed: {e}")
        return None

    state = _load_state()
    prev = state.get("last_trainer")
    prev_evals = prev.get("evals") if prev else None
    drift = _drift(evals, prev_evals)

    sample = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "kind": "trainer",
        "game_id": game_id,
        "evals": evals,
        "eval_drift_cp": round(drift, 2),
    }
    state["trainer_history"].append(sample)
    state["last_trainer"] = {"evals": evals, "ts": sample["ts"]}
    _save_state(state)

    label = _classify_trainer(drift, state["trainer_history"])
    from nnue_guards import nnue_log
    nnue_log(f"probe trainer: drift={drift:.1f}cp evals={evals} — {label}")
    return sample


def record_engine_probe(*, deploy_run: int | None = None) -> dict | None:
    """After deploy+rebuild: embedded weights vs last deploy snapshot + root moves."""
    if not BIN.exists():
        return None
    try:
        evals = _engine_eval_batch()
        roots = _engine_root_moves()
    except Exception as e:
        from nnue_guards import nnue_log
        nnue_log(f"probe engine failed: {e}")
        return None

    state = _load_state()
    prev = state.get("last_deploy")
    prev_evals = prev.get("evals") if prev else None
    prev_roots = prev.get("roots") if prev else None
    eval_drift = _drift(evals, prev_evals)
    move_rate = _move_change_rate(roots, prev_roots)

    sample = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "kind": "deploy",
        "deploy_run": deploy_run,
        "evals": evals,
        "roots": roots,
        "eval_drift_cp": round(eval_drift, 2),
        "move_change_rate": round(move_rate, 3),
    }
    state["deploy_history"].append(sample)
    state["last_deploy"] = {"evals": evals, "roots": roots, "ts": sample["ts"]}
    _save_state(state)

    if prev is None:
        status = "deploy_baseline (first probe after deploy)"
    elif eval_drift < TRAINER_DRIFT_STALE_CP and move_rate == 0.0:
        status = "deploy_plateau (eval flat + same roots)"
    elif eval_drift >= TRAINER_DRIFT_STALE_CP or move_rate > 0:
        status = "deploy_shift (eval {:.1f}cp, {:.0%} roots changed)".format(
            eval_drift, move_rate
        )
    else:
        status = "deploy_quiet (small eval recalibration)"

    from nnue_guards import nnue_log
    nnue_log(
        f"probe deploy: drift={eval_drift:.1f}cp move_delta={move_rate:.0%} "
        f"roots={roots} — {status}"
    )
    return sample


def maybe_trainer_probe(game_id: int, *, every: int = 8) -> None:
    """Periodic trainer probe — cheap halfpw, no rebuild required."""
    from nnue_guards import load_guard_state
    runs = load_guard_state().get("games_trained", 0)
    if every <= 0 or runs % every != 0:
        return
    record_trainer_probe(game_id=game_id)


def nnue_status_compact() -> str:
    """One line for compact scoreboard (legacy callers)."""
    try:
        from nnue_learning_metrics import collect_learning_report
        r = collect_learning_report(write_json=False)
        return (
            f" NNUE [{r['phase']}] trained {r['games_trained']}  "
            f"drift {r['last_trainer_drift_cp']:.1f}cp  "
            f"cand {r['candidate_vs_deployed_cp']:.1f}cp  "
            f"deploys {r['deploy_runs']} (+{r['games_since_deploy']})"
        )
    except Exception:
        pass
    from nnue_guards import load_guard_state

    g = load_guard_state()
    trained = g.get("games_trained", 0)
    deploys = g.get("deploy_runs", 0)
    since = g.get("games_since_deploy", 0)
    state = _load_state()
    th = state.get("trainer_history", [])
    drift = th[-1].get("eval_drift_cp", 0.0) if th else 0.0
    recent = [h.get("eval_drift_cp", 0.0) for h in th[-5:]]
    if len(recent) >= 3 and all(d < TRAINER_DRIFT_STALE_CP for d in recent):
        phase = "PLATEAU"
    elif drift >= TRAINER_DRIFT_STALE_CP:
        phase = "learning"
    else:
        phase = "quiet"
    return (
        f" NNUE [{phase}] trained {trained}  drift {drift:.1f}cp  "
        f"deploys {deploys} ({since} since)"
    )


def print_report() -> None:
    state = _load_state()
    print("NNUE learning probe report")
    print(f"  file: {PROBE_PATH}")
    lt = state.get("last_trainer")
    ld = state.get("last_deploy")
    if lt:
        print(f"  last trainer evals: {lt.get('evals')}")
    if ld:
        print(f"  last deploy evals:  {ld.get('evals')}")
        print(f"  last deploy roots:  {ld.get('roots')}")
    th = state.get("trainer_history", [])
    if len(th) >= 2:
        print(f"  recent trainer drift: {[h.get('eval_drift_cp') for h in th[-5:]]}")
    dh = state.get("deploy_history", [])
    if len(dh) >= 2:
        print(f"  recent deploy drift:  {[h.get('eval_drift_cp') for h in dh[-5:]]}")
        print(f"  recent move delta:    {[h.get('move_change_rate') for h in dh[-5:]]}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--report":
        print_report()
    else:
        print("Usage: python training/plateau_probe.py --report")
