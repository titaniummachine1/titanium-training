#!/usr/bin/env python3
"""Gate: packed-row label perspective, tri-path parity, target correlation.

Does not remove TRAINING_PAUSED.json — caller decides after reviewing output.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
TRAINING = REPO / "training"
sys.path.insert(0, str(TRAINING))

from build_feature_cache import FV_LEN, record_to_fv
from db_import import GAMES_DB_PATH, LABELS_DB_PATH, aggregate_outcome_label
from label_perspective import LABEL_PERSPECTIVE_CONVENTION, packed_row_target_prob, value_i16_to_dataset_stm
from streaming_db_loader import LabeledPosition, _featurize_records, features_to_torch_batch
from titanium_training.data.eval_packed import eval_packed_batch_allow_errors
from titanium_training.models.eval_forward import record_to_trainer_batch
from titanium_training.models.halfpw import Net, forward_trace
from titanium_training.paths import ENGINE_BIN, WEIGHTS_BIN
from titanium_training.training.trainer import HalfPW

OUT_PATH = REPO / "training" / "runs" / "packed_row_parity_gate.json"
N_PACKED = 1000
N_OUTCOME_AUDIT = 200
MAX_INTERMEDIATE_ABS = 1e-5
MAX_SCALAR_OUT_ABS = 2e-5
MAX_NEURAL_OUT_ABS = 3e-5
MAX_FINAL_CP = 1


def rust_net_cp(rec: dict) -> int:
    if "net_eval" in rec:
        return int(rec["net_eval"])
    return int(rec["eval"])


def _corr(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    a, b = np.array(xs), np.array(ys)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _max_abs(a: float, b: float) -> float:
    return abs(float(a) - float(b))


def _compare_traces(hp, tr) -> dict:
    worst_scalar = max(
        _max_abs(hp.scalar_inputs[k], tr.scalar_inputs[k])
        for k in ("d_me", "d_opp", "w_me", "w_opp", "pd", "wd", "width_opp")
    )
    return {
        "scalar_inputs": worst_scalar,
        "scalar_out": _max_abs(hp.scalar_out, tr.scalar_out),
        "route_out": _max_abs(hp.route_out, tr.route_out),
        "cat_out": _max_abs(hp.cat_out, tr.cat_out),
        "width_contrib": _max_abs(hp.width_contrib, tr.width_contrib),
        "wall_acc": max(abs(float(x) - float(y)) for x, y in zip(hp.wall_acc, tr.wall_acc)),
        "hidden_pre": max(abs(float(x) - float(y)) for x, y in zip(hp.hidden_pre, tr.hidden_pre)),
        "hidden_clip": max(abs(float(x) - float(y)) for x, y in zip(hp.hidden_clip, tr.hidden_clip)),
        "neural_out": _max_abs(hp.neural_out, tr.neural_out),
        "final_cp": abs(int(hp.final_cp) - int(tr.final_cp)),
    }


def sample_packed(con: sqlite3.Connection, n: int) -> list[tuple[bytes, int, float]]:
    return con.execute(
        """
        SELECT p.packed_state, p.side_to_move,
               AVG(l.value_i16) / 100.0
        FROM teacher_positions p
        JOIN teacher_labels l ON l.position_key = p.position_key
        WHERE l.value_i16 IS NOT NULL
        GROUP BY p.position_key, p.packed_state, p.side_to_move
        ORDER BY RANDOM()
        LIMIT ?
        """,
        (n,),
    ).fetchall()


def audit_outcome_mismatches(con_labels: sqlite3.Connection, games_con: sqlite3.Connection, n: int) -> dict:
    rows = con_labels.execute(
        """
        SELECT p.pos_key, p.side_to_move, l.value_stm, l.source, l.n_samples
        FROM positions p
        JOIN labels l ON l.pos_key = p.pos_key
        WHERE l.source LIKE '%_outcome'
        ORDER BY RANDOM()
        LIMIT ?
        """,
        (n,),
    ).fetchall()
    mismatches = []
    exact = 0
    traced = 0
    for pos_key, stm, value_stm, source, n_samples in rows:
        g = games_con.execute(
            """
            SELECT g.game_id, g.outcome_p0, gm.move_num
            FROM game_moves gm
            JOIN games g ON g.game_id = gm.game_id
            WHERE gm.pos_key = ?
            LIMIT 1
            """,
            (str(pos_key),),
        ).fetchone()
        if not g:
            continue
        traced += 1
        game_id, outcome_p0, move_num = g
        moves = [
            r[0]
            for r in games_con.execute(
                "SELECT move_alg FROM game_moves WHERE game_id=? ORDER BY move_num",
                (game_id,),
            )
        ]
        expected = float(outcome_p0) if int(stm) == 0 else float(-outcome_p0)
        delta = float(value_stm) - expected
        if abs(delta) < 1e-6:
            exact += 1
            continue
        is_terminal = int(move_num) >= len(moves)
        likely = None
        if not is_terminal or int(n_samples) > 1:
            likely = "averaged_outcome_labels_across_duplicate_pos_key"
        mismatches.append(
            {
                "pos_key": str(pos_key),
                "source": source,
                "side_to_move": int(stm),
                "stored_value_stm": float(value_stm),
                "expected_value_stm": expected,
                "outcome_p0": int(outcome_p0),
                "n_samples": int(n_samples),
                "game_id": game_id,
                "move_num": int(move_num),
                "is_terminal_position": is_terminal,
                "delta": delta,
                "likely_cause": likely,
            }
        )
    return {
        "traced": traced,
        "exact": exact,
        "mismatches": mismatches,
        "fraction_exact": round(exact / traced, 4) if traced else None,
        "aggregation_rule": (
            "labels INSERT ON CONFLICT: value_stm = (value_stm*n_samples + new) / (n_samples+1)"
        ),
    }


def main() -> int:
    if not ENGINE_BIN.is_file() or not WEIGHTS_BIN.is_file():
        print("ERROR: engine or weights missing")
        return 1

    labels_con = sqlite3.connect(str(LABELS_DB_PATH))
    games_con = sqlite3.connect(str(GAMES_DB_PATH)) if GAMES_DB_PATH.is_file() else None
    rows = sample_packed(labels_con, N_PACKED)

    model = HalfPW(WEIGHTS_BIN)
    net = Net.load(WEIGHTS_BIN)

    n_ok = 0
    cp_fail = 0
    trace_fail = 0
    engine_turn_flipped = 0
    targets: list[float] = []
    preds: list[float] = []
    engine_net_cps: list[float] = []
    worst_cp = 0
    first_cp_mismatch: dict | None = None
    first_trace_mismatch: dict | None = None

    for packed, dataset_stm, value_dataset in rows:
        recs = eval_packed_batch_allow_errors([(0, bytes(packed))])
        rec = recs[0] if recs else None
        if not rec or not rec.get("ok"):
            continue
        engine_turn = int(rec["turn"])
        if engine_turn != int(dataset_stm):
            engine_turn_flipped += 1
        target = packed_row_target_prob(
            value_dataset_stm=float(value_dataset),
            engine_turn=engine_turn,
            dataset_side_to_move=int(dataset_stm),
        )
        labeled = LabeledPosition(
            position_id="teacher:gate",
            packed_state=bytes(packed),
            value_target=0.0,
            sample_weight=1.0,
            storage_kind="packed",
            dataset_side_to_move=int(dataset_stm),
            value_dataset_stm=float(value_dataset),
        )
        _ids, features, value_targets, *_ = _featurize_records([labeled])
        if not _ids or abs(float(value_targets[0]) - target) > 1e-6:
            continue
        loader_target = float(features[0, 0])
        batch = features_to_torch_batch(features, _ids)
        pred = int(model(batch).item())
        rust_net = rust_net_cp(rec)
        dcp = abs(pred - rust_net)
        worst_cp = max(worst_cp, dcp)
        if dcp > MAX_FINAL_CP:
            cp_fail += 1
            if first_cp_mismatch is None:
                first_cp_mismatch = {
                    "pred": pred,
                    "rust_net_eval": rust_net,
                    "rust_full_eval": int(rec["eval"]),
                    "engine_turn": engine_turn,
                    "dataset_stm": int(dataset_stm),
                }
        hp = forward_trace(net, rec, normed=False)
        tr = model.forward_trace(record_to_trainer_batch(rec))[0]
        cmp_tr = _compare_traces(hp, tr)
        trace_bad = (
            cmp_tr["final_cp"] > MAX_FINAL_CP
            or abs(hp.final_cp - rust_net) > MAX_FINAL_CP
        )
        if abs(loader_target - target) > 1e-5:
            trace_bad = True
        if trace_bad:
            trace_fail += 1
            if first_trace_mismatch is None:
                first_trace_mismatch = {"halfpw_vs_trainer": cmp_tr, "halfpw_vs_rust_net": abs(hp.final_cp - rust_net)}
        targets.append(target)
        preds.append(float(pred))
        engine_net_cps.append(float(rust_net))
        n_ok += 1

    outcome_audit = (
        audit_outcome_mismatches(labels_con, games_con, N_OUTCOME_AUDIT)
        if games_con is not None
        else {"skipped": True}
    )
    labels_con.close()
    if games_con:
        games_con.close()

    corr_target_trainer = _corr(targets, preds)
    corr_target_engine = _corr(targets, engine_net_cps)
    mism = outcome_audit.get("mismatches") or []
    unexplained = [
        m
        for m in mism
        if m.get("likely_cause") is None and abs(m.get("delta", 0)) >= 0.15
    ]
    report = {
        "gate": "packed_row_parity_v2",
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "label_perspective_convention": LABEL_PERSPECTIVE_CONVENTION,
        "n_requested": N_PACKED,
        "n_evaluated": n_ok,
        "engine_turn_flipped_count": engine_turn_flipped,
        "trainer_rust_net_cp_failures": cp_fail,
        "halfpw_trainer_trace_failures": trace_fail,
        "worst_trainer_rust_net_cp_delta": worst_cp,
        "corr_target_engine_net_cp": corr_target_engine,
        "corr_target_deployed_trainer_cp": corr_target_trainer,
        "first_cp_mismatch": first_cp_mismatch,
        "first_trace_mismatch": first_trace_mismatch,
        "outcome_audit": outcome_audit,
        "passed": (
            n_ok >= N_PACKED
            and cp_fail == 0
            and trace_fail == 0
            and corr_target_engine is not None
            and corr_target_engine > 0.15
            and corr_target_trainer is not None
            and corr_target_trainer > 0.15
            and not unexplained
        ),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in report if k not in ("outcome_audit",)}, indent=2))
    if outcome_audit.get("mismatches"):
        print("outcome mismatches:", json.dumps(outcome_audit["mismatches"][:5], indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
