#!/usr/bin/env python3
"""End-to-end label orientation audit — do not retrain until this passes."""
from __future__ import annotations

import json
import math
import os
import random
import sqlite3
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[3]
TRAINING = REPO / "training"
sys.path.insert(0, str(TRAINING))

from build_feature_cache import record_to_fv
from db_import import GAMES_DB_PATH, LABELS_DB_PATH
from extend_teacher_dataset import float_stm_to_value_i16, make_position_key_from_state
from streaming_db_loader import LabelsRepository
from titanium_training.data.teacher_value import teacher_value_target
from titanium_training.models.halfpw import NET_MIRC
from titanium_training.training.trainer import HalfPW
from titanium_training.paths import ENGINE_BIN, REPO_ROOT, WEIGHTS_BIN
from titanium_training.store.state import PositionState

OUT_PATH = REPO / "training" / "runs" / "label_orientation_audit.json"
SCALE = 400.0
N_SAMPLE = 1200
N_EXAMPLES = 20


def cp_to_prob(cp: float) -> float:
    return 1.0 / (1.0 + math.exp(-cp / SCALE))


def corr(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    a, b = np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return None
    return float(np.corrcoef(a, b)[0, 1])


@dataclass
class MappingHypothesis:
    name: str
    description: str

    def target(self, *, value_raw: float, stm: int, value_kind: str) -> float:
        raise NotImplementedError


class StmNormalized(MappingHypothesis):
    def __init__(self) -> None:
        super().__init__(
            name="stm_normalized_neg1_pos1",
            description="value is STM advantage in [-1,+1]; target = (v+1)/2",
        )

    def target(self, *, value_raw: float, stm: int, value_kind: str) -> float:
        v = value_raw / 100.0 if value_kind == "i16" else float(value_raw)
        return (v + 1.0) / 2.0


class StmWinProb(MappingHypothesis):
    def __init__(self) -> None:
        super().__init__(
            name="stm_win_prob_0_1",
            description="value is STM win probability in [0,1]; target = v",
        )

    def target(self, *, value_raw: float, stm: int, value_kind: str) -> float:
        v = value_raw / 100.0 if value_kind == "i16" else float(value_raw)
        return max(0.0, min(1.0, v))


class P0NormalizedToStm(MappingHypothesis):
    def __init__(self) -> None:
        super().__init__(
            name="p0_normalized_to_stm",
            description="value is P0 advantage [-1,+1]; flip sign when stm==1 then (v+1)/2",
        )

    def target(self, *, value_raw: float, stm: int, value_kind: str) -> float:
        v = value_raw / 100.0 if value_kind == "i16" else float(value_raw)
        stm_v = v if stm == 0 else -v
        return (stm_v + 1.0) / 2.0


class P0WinProbToStm(MappingHypothesis):
    def __init__(self) -> None:
        super().__init__(
            name="p0_win_prob_to_stm",
            description="value is P0 win prob [0,1]; STM target = v if stm==0 else 1-v",
        )

    def target(self, *, value_raw: float, stm: int, value_kind: str) -> float:
        v = value_raw / 100.0 if value_kind == "i16" else float(value_raw)
        v = max(0.0, min(1.0, v))
        return v if stm == 0 else (1.0 - v)


class InvertedStmNormalized(MappingHypothesis):
    def __init__(self) -> None:
        super().__init__(
            name="inverted_stm_normalized",
            description="bug hypothesis: target = (1-v)/2 for normalized STM",
        )

    def target(self, *, value_raw: float, stm: int, value_kind: str) -> float:
        v = value_raw / 100.0 if value_kind == "i16" else float(value_raw)
        return (1.0 - v) / 2.0


HYPOTHESES = [
    StmNormalized(),
    StmWinProb(),
    P0NormalizedToStm(),
    P0WinProbToStm(),
    InvertedStmNormalized(),
]


def engine_eval_json(moves: list[str], weights: Path | None = None) -> dict | None:
    env = os.environ.copy()
    if weights:
        env["TITANIUM_NET_WEIGHTS_PATH"] = str(weights.resolve())
    else:
        env.pop("TITANIUM_NET_WEIGHTS_PATH", None)
    try:
        proc = subprocess.run(
            [str(ENGINE_BIN), "eval", *moves, "--json"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            env=env,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return None
    if proc.returncode != 0:
        return None
    try:
        return json.loads(proc.stdout.strip())
    except json.JSONDecodeError:
        return None


def engine_eval_packed(packed: bytes, weights: Path | None = None) -> dict | None:
    from titanium_training.data.eval_packed import eval_packed_batch_allow_errors

    env = os.environ.copy()
    if weights:
        env["TITANIUM_NET_WEIGHTS_PATH"] = str(weights.resolve())
    else:
        env.pop("TITANIUM_NET_WEIGHTS_PATH", None)
    old = os.environ.get("TITANIUM_NET_WEIGHTS_PATH")
    try:
        if weights:
            os.environ["TITANIUM_NET_WEIGHTS_PATH"] = str(weights.resolve())
        elif "TITANIUM_NET_WEIGHTS_PATH" in os.environ:
            del os.environ["TITANIUM_NET_WEIGHTS_PATH"]
        recs = eval_packed_batch_allow_errors([(0, packed)])
    finally:
        if old is not None:
            os.environ["TITANIUM_NET_WEIGHTS_PATH"] = old
        elif "TITANIUM_NET_WEIGHTS_PATH" in os.environ:
            del os.environ["TITANIUM_NET_WEIGHTS_PATH"]
    if not recs or not recs[0].get("ok"):
        return None
    return recs[0]


def model_eval_from_rec(rec: dict, model: HalfPW) -> float:
    import torch
    from titanium_training.training.trainer import QuoridorDataset

    row = dict(rec)
    row.setdefault("outcome", 0.0)
    row.setdefault("legal_wall_count", rec.get("legal_wall_count", 0))
    row.setdefault("legal_path_cross_p0", 0)
    row.setdefault("legal_path_cross_p1", 0)
    for k in ("hw", "vw"):
        if k not in row:
            row[k] = [0.0] * 64
    ds = QuoridorDataset([row])
    batch = ds[0]
    b = {k: (v.unsqueeze(0) if hasattr(v, "unsqueeze") else v) for k, v in batch.items()}
    with torch.no_grad():
        return float(model(b).item())


def sample_teacher_rows(con: sqlite3.Connection, n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows = con.execute(
        """
        SELECT p.position_key, p.packed_state, p.side_to_move, l.value_i16, l.source_cohort
        FROM teacher_positions p
        JOIN teacher_labels l ON l.position_key = p.position_key
        WHERE l.value_i16 IS NOT NULL
        ORDER BY RANDOM()
        LIMIT ?
        """,
        (n,),
    ).fetchall()
    out = []
    for position_key, packed_state, stm, value_i16, cohort in rows:
        out.append(
            {
                "storage": "teacher",
                "position_key_hex": bytes(position_key).hex(),
                "packed_state": bytes(packed_state),
                "side_to_move": int(stm),
                "value_i16": int(value_i16),
                "value_stm_db": int(value_i16) / 100.0,
                "source_cohort": str(cohort or ""),
                "label_source": "teacher_labels.value_i16",
            }
        )
    rng.shuffle(out)
    return out


def sample_json_outcome_rows(con: sqlite3.Connection, n: int) -> list[dict]:
    rows = con.execute(
        """
        SELECT p.pos_key, p.position_data, p.side_to_move, l.value_stm, l.source, l.n_samples
        FROM positions p
        JOIN labels l ON l.pos_key = p.pos_key
        WHERE l.source LIKE '%_outcome'
        ORDER BY RANDOM()
        LIMIT ?
        """,
        (n,),
    ).fetchall()
    out = []
    for pos_key, data, stm, value_stm, source, n_samples in rows:
        raw = bytes(data) if isinstance(data, bytes) else str(data).encode()
        out.append(
            {
                "storage": "json",
                "pos_key": str(pos_key),
                "position_data": raw,
                "side_to_move": int(stm),
                "value_stm": float(value_stm),
                "source": str(source),
                "n_samples": int(n_samples),
                "label_source": f"labels.value_stm ({source})",
            }
        )
    return out


def trace_game_outcome(games_con: sqlite3.Connection, pos_key: str) -> dict | None:
    row = games_con.execute(
        """
        SELECT g.game_id, g.outcome_p0, gm.move_num
        FROM game_moves gm
        JOIN games g ON g.game_id = gm.game_id
        WHERE gm.pos_key = ?
        LIMIT 1
        """,
        (pos_key,),
    ).fetchone()
    if not row:
        return None
    game_id, outcome_p0, move_num = row
    moves = [
        r[0]
        for r in games_con.execute(
            "SELECT move_alg FROM game_moves WHERE game_id=? ORDER BY move_num",
            (game_id,),
        )
    ]
    return {
        "game_id": game_id,
        "outcome_p0": int(outcome_p0),
        "move_num": int(move_num),
        "moves_prefix": moves[: int(move_num)],
        "moves_full": moves,
    }


def terminal_tests(model: HalfPW) -> list[dict]:
    """Synthetic near-terminal positions from replayed games."""
    tests: list[dict] = []
    if not GAMES_DB_PATH.is_file():
        return tests
    con = sqlite3.connect(str(GAMES_DB_PATH))
    games = con.execute(
        "SELECT game_id, outcome_p0 FROM games ORDER BY RANDOM() LIMIT 40"
    ).fetchall()
    for game_id, outcome_p0 in games:
        moves = [
            r[0]
            for r in con.execute(
                "SELECT move_alg FROM game_moves WHERE game_id=? ORDER BY move_num",
                (game_id,),
            )
        ]
        if len(moves) < 4:
            continue
        for ply in (len(moves) - 2, len(moves) - 1, len(moves)):
            prefix = moves[:ply]
            rec = engine_eval_json(prefix)
            if not rec:
                continue
            stm = int(rec.get("turn", ply % 2))
            outcome_stm = float(outcome_p0) if stm == 0 else float(-outcome_p0)
            target_stm = (outcome_stm + 1.0) / 2.0
            cp = int(rec.get("eval", 0))
            prob = cp_to_prob(cp)
            tests.append(
                {
                    "game_id": game_id,
                    "ply": ply,
                    "moves": prefix,
                    "side_to_move": stm,
                    "outcome_p0": int(outcome_p0),
                    "outcome_stm": outcome_stm,
                    "target_stm_hard": target_stm,
                    "engine_cp": cp,
                    "engine_prob": prob,
                    "terminal_ply": ply >= len(moves),
                }
            )
        if len(tests) >= 24:
            break
    con.close()
    return tests


def _rec_from_row(row: dict) -> dict | None:
    if row["storage"] == "teacher":
        return engine_eval_packed(row["packed_state"])
    rec = json.loads(row["position_data"].decode())
    if rec.get("eval") is None:
        trace = row.get("trace")
        if trace:
            return engine_eval_json(trace["moves_prefix"])
    return rec


def mirror_swap_tests(model: HalfPW, seed_rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    for row in seed_rows[:12]:
        rec_o = _rec_from_row(row)
        if not rec_o:
            continue
        me = int(rec_o.get("turn", row["side_to_move"]))
        v = float(row.get("value_stm_db") or row.get("value_stm"))
        target_o = (v + 1.0) / 2.0
        cp_o = model_eval_from_rec(rec_o, model)

        # Player/color swap: exchange pawn0/pawn1, flip turn, negate P0-outcome label.
        rec_swap = dict(rec_o)
        rec_swap["pawn0"], rec_swap["pawn1"] = int(rec_o["pawn1"]), int(rec_o["pawn0"])
        rec_swap["wl0"], rec_swap["wl1"] = int(rec_o.get("wl1", 0)), int(rec_o.get("wl0", 0))
        rec_swap["d0"], rec_swap["d1"] = int(rec_o.get("d1", 0)), int(rec_o.get("d0", 0))
        rec_swap["turn"] = 1 - me
        target_swap = (1.0 - target_o)
        cp_swap = model_eval_from_rec(rec_swap, model)
        out.append(
            {
                "test": "player_swap",
                "stm_orig": me,
                "stm_after_swap": 1 - me,
                "stored_value_stm": v,
                "target_orig": target_o,
                "expected_target_after_swap": target_swap,
                "complement_ok": abs(target_o + target_swap - 1.0) < 0.02,
                "engine_cp_orig": cp_o,
                "engine_cp_swapped": cp_swap,
                "eval_antisymmetric": abs(cp_o + cp_swap) < 80,
            }
        )

        # Pure board mirror (flip ranks, same player ids): mirror pawn cells only.
        rec_mirror = dict(rec_o)
        rec_mirror["pawn0"] = int(NET_MIRC[int(rec_o["pawn0"])])
        rec_mirror["pawn1"] = int(NET_MIRC[int(rec_o["pawn1"])])
        rec_mirror["turn"] = me
        cp_mirror = model_eval_from_rec(rec_mirror, model)
        out.append(
            {
                "test": "board_mirror_same_players",
                "stm": me,
                "target_orig": target_o,
                "target_after_mirror": target_o,
                "target_preserved": True,
                "engine_cp_orig": cp_o,
                "engine_cp_mirrored": cp_mirror,
            }
        )
    return out


def audit_records(rows: list[dict], model: HalfPW, games_con: sqlite3.Connection | None) -> tuple[list[dict], dict]:
    audited: list[dict] = []
    hy_stats = {h.name: {"target": [], "engine_cp": [], "engine_prob": []} for h in HYPOTHESES}

    for row in rows:
        item: dict[str, Any] = dict(row)
        rec = None
        cp = None
        if row["storage"] == "teacher":
            rec = engine_eval_packed(row["packed_state"])
            if rec:
                cp = int(rec.get("eval", 0))
                item["engine_turn"] = int(rec.get("turn", -1))
                item["turn_match"] = item["engine_turn"] == row["side_to_move"]
        else:
            rec = json.loads(row["position_data"].decode())
            cp = int(rec.get("eval", 0)) if rec.get("eval") is not None else None
            if cp is None:
                moves_row = trace_game_outcome(games_con, row["pos_key"]) if games_con else None
                if moves_row:
                    ev = engine_eval_json(moves_row["moves_prefix"])
                    if ev:
                        rec = ev
                        cp = int(ev.get("eval", 0))
            item["engine_turn"] = int(rec.get("turn", row["side_to_move"])) if rec else None
            if games_con:
                trace = trace_game_outcome(games_con, row["pos_key"])
                if trace:
                    stm = row["side_to_move"]
                    expected_stm = float(trace["outcome_p0"]) if stm == 0 else float(-trace["outcome_p0"])
                    item["trace"] = trace
                    item["expected_value_stm_from_game"] = expected_stm
                    item["stored_vs_game_delta"] = float(row["value_stm"]) - expected_stm

        if cp is None or rec is None:
            item["skip"] = "no_engine_eval"
            audited.append(item)
            continue

        item["engine_cp"] = cp
        item["engine_prob"] = cp_to_prob(cp)
        item["model_cp"] = model_eval_from_rec(rec, model)
        item["model_prob"] = cp_to_prob(item["model_cp"])

        stm = row["side_to_move"]
        raw = row["value_i16"] if row["storage"] == "teacher" else row["value_stm"]
        kind = "i16" if row["storage"] == "teacher" else "stm"
        item["current_mapping_target"] = (
            teacher_value_target(int(row["value_i16"]))
            if row["storage"] == "teacher"
            else (float(row["value_stm"]) + 1.0) / 2.0
        )
        for h in HYPOTHESES:
            t = h.target(value_raw=raw, stm=stm, value_kind=kind)
            item[f"target_{h.name}"] = t
            hy_stats[h.name]["target"].append(t)
            hy_stats[h.name]["engine_cp"].append(float(cp))
            hy_stats[h.name]["engine_prob"].append(cp_to_prob(cp))

        audited.append(item)

    correlations = {}
    for h in HYPOTHESES:
        stats = hy_stats[h.name]
        correlations[h.name] = {
            "description": h.description,
            "corr_target_engine_cp": corr(stats["target"], stats["engine_cp"]),
            "corr_target_engine_prob": corr(stats["target"], stats["engine_prob"]),
            "n": len(stats["target"]),
        }
    for subset_name, filt in (
        ("teacher_friend_only", lambda a: a.get("storage") == "teacher"),
        ("json_outcome_only", lambda a: a.get("storage") == "json"),
    ):
        sub = [a for a in audited if filt(a) and "engine_cp" in a]
        correlations[subset_name] = {
            "n": len(sub),
            "corr_current_target_engine_cp": corr(
                [a["current_mapping_target"] for a in sub],
                [float(a["engine_cp"]) for a in sub],
            ),
        }
    correlations["current_code_stm_normalized"] = correlations.get(StmNormalized().name)
    return audited, correlations


def pick_examples(audited: list[dict], n: int = N_EXAMPLES) -> list[dict]:
    usable = [a for a in audited if "engine_cp" in a and "trace" in a or a.get("storage") == "teacher"]
    usable.sort(key=lambda a: abs(a.get("stored_vs_game_delta", 0) if "stored_vs_game_delta" in a else 0), reverse=True)
    examples = []
    for a in usable[: n // 2]:
        examples.append({k: a[k] for k in a if k not in ("packed_state", "position_data")})
    teacher_sorted = sorted(
        [a for a in audited if a.get("storage") == "teacher" and "engine_cp" in a],
        key=lambda a: abs(a["engine_cp"]),
        reverse=True,
    )
    for a in teacher_sorted[: n - len(examples)]:
        slim = {k: a[k] for k in a if k not in ("packed_state", "position_data")}
        examples.append(slim)
    return examples[:n]


def main() -> int:
    labels_path = LABELS_DB_PATH
    games_con = sqlite3.connect(str(GAMES_DB_PATH)) if GAMES_DB_PATH.is_file() else None
    con = sqlite3.connect(str(labels_path))
    teacher_n = int(N_SAMPLE * 0.85)
    json_n = N_SAMPLE - teacher_n
    rows = sample_teacher_rows(con, teacher_n, seed=42) + sample_json_outcome_rows(con, json_n)
    con.close()

    model = HalfPW(WEIGHTS_BIN)
    model.eval()

    audited, correlations = audit_records(rows, model, games_con)
    examples = pick_examples(audited, N_EXAMPLES)
    terminals = terminal_tests(model)
    mirrors = mirror_swap_tests(model, [a for a in audited if "engine_cp" in a][:20])

    n_ok = sum(1 for a in audited if "engine_cp" in a)
    n_trace = sum(1 for a in audited if a.get("trace"))
    n_match = sum(1 for a in audited if a.get("stored_vs_game_delta") is not None and abs(a["stored_vs_game_delta"]) < 1e-6)

    best = max(
        (v for v in correlations.values() if isinstance(v, dict) and v.get("corr_target_engine_cp") is not None),
        key=lambda v: v["corr_target_engine_cp"],
        default=None,
    )
    best_name = None
    if best:
        for k, v in correlations.items():
            if v is best:
                best_name = k
                break

    report = {
        "audit": "label_orientation_e2e",
        "training_stopped": True,
        "n_sampled": len(rows),
        "n_with_engine_eval": n_ok,
        "label_provenance": {
            "teacher_parquet_sqlite": {
                "table": "teacher_labels.value_i16",
                "build_path": "teacher_dataset/build.py copies labels.value * 100 from teacher store",
                "teacher_store_friend": "friend_selfplay rootValue -> labels.value (normalized NN root, STM assumed)",
                "teacher_store_zeroink": "search.root_value float -> labels.value",
                "overnight_outcome_parquet": "sync_overnight_to_teacher float_stm_to_value_i16(outcome_stm)",
            },
            "labels_db_json": {
                "table": "labels.value_stm",
                "comment": "db_import.py: outcome_stm = outcome_p0 if stm==0 else -outcome_p0",
                "sources": "wallz_outcome, zeroink_outcome, pool_*_outcome, overnight_*_outcome",
            },
        },
        "current_mapping_equations": {
            "value_stm_json": "target_prob = (value_stm + 1.0) / 2.0",
            "value_i16_teacher": "target_prob = (value_i16 / 100.0 + 1.0) / 2.0  # teacher_value_target()",
            "trainer_parquet_path": "value_i16 STM -> outcome_p0 flip -> QuoridorDataset STM target (round-trip)",
            "streaming_db_loader": "same as above per storage_kind",
            "engine_eval": "centipawns from side-to-move perspective (search.rs evaluate, turn=self.g.turn)",
        },
        "hypothesis_correlations": correlations,
        "best_hypothesis_by_engine_cp_corr": best_name,
        "outcome_trace_checks": {
            "n_traced": n_trace,
            "n_exact_match_game_outcome": n_match,
            "fraction_exact": round(n_match / n_trace, 4) if n_trace else None,
        },
        "terminal_position_tests": terminals,
        "mirror_and_swap_tests": mirrors,
        "concrete_examples": examples,
        "verdict": None,
    }

    stm_corr = correlations.get(StmNormalized().name, {}).get("corr_target_engine_cp")
    inv_corr = correlations.get(InvertedStmNormalized().name, {}).get("corr_target_engine_cp")
    if stm_corr is not None and stm_corr > 0.15:
        report["verdict"] = "value_i16/value_stm appear STM-normalized; current mapping direction is correct"
        report["recommended_mapping"] = report["current_mapping_equations"]
    elif inv_corr is not None and inv_corr > 0.15 and (stm_corr or 0) < 0:
        report["verdict"] = "BUG: labels correlate with inverted engine eval — mapping sign likely wrong"
        report["corrected_mapping_equations"] = {
            "value_stm_json": "target_prob = (1.0 - value_stm) / 2.0  # invert STM",
            "value_i16_teacher": "target_prob = (1.0 - value_i16/100.0) / 2.0",
        }
    elif best_name == P0NormalizedToStm().name:
        report["verdict"] = "BUG: stored values appear P0-normalized, not STM"
        report["corrected_mapping_equations"] = {
            "value_stm_json": "stm_v = value if stm==0 else -value; target = (stm_v+1)/2",
            "value_i16_teacher": "same flip using side_to_move before teacher_value_target",
        }
    else:
        report["verdict"] = "INCONCLUSIVE or mixed cohort semantics — inspect concrete_examples and cohort splits"
        report["corrected_mapping_equations"] = "pending manual review"

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("audit", "n_sampled", "n_with_engine_eval", "hypothesis_correlations", "best_hypothesis_by_engine_cp_corr", "outcome_trace_checks", "verdict")}, indent=2))
    print(f"full report -> {OUT_PATH}")
    if games_con:
        games_con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
