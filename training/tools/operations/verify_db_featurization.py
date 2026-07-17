#!/usr/bin/env python3
"""Verify DB featurization matches engine eval-packed-batch (streaming canonical path)."""
from __future__ import annotations

import json
import random
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
TRAINING = REPO / "training"
sys.path.insert(0, str(TRAINING))

from build_feature_cache import FV_LEN, record_to_fv
from db_import import LABELS_DB_PATH
from label_perspective import LABEL_PERSPECTIVE_CONVENTION, packed_row_target_prob, value_i16_to_dataset_stm
from streaming_db_loader import LabeledPosition, _featurize_records, features_to_torch_batch
from titanium_training.data.eval_packed import eval_packed_batch_allow_errors
from titanium_training.models.eval_forward import record_to_trainer_batch
from titanium_training.paths import ENGINE_BIN, WEIGHTS_BIN
from titanium_training.training.trainer import HalfPW

OUT = REPO / "training" / "runs" / "db_featurization_verify.json"
N = 64


def main() -> int:
    if not ENGINE_BIN.is_file():
        print("engine missing")
        return 1
    con = sqlite3.connect(str(LABELS_DB_PATH))
    rows = con.execute(
        """
        SELECT p.packed_state, p.side_to_move, AVG(l.value_i16) / 100.0
        FROM teacher_positions p
        JOIN teacher_labels l ON l.position_key = p.position_key
        WHERE l.value_i16 IS NOT NULL
        GROUP BY p.position_key
        ORDER BY RANDOM()
        LIMIT ?
        """,
        (N,),
    ).fetchall()
    con.close()
    model = HalfPW(WEIGHTS_BIN)
    ok = 0
    for packed, stm, v in rows:
        rec = eval_packed_batch_allow_errors([(0, bytes(packed))])[0]
        if not rec.get("ok"):
            continue
        target = packed_row_target_prob(
            value_dataset_stm=float(v),
            engine_turn=int(rec["turn"]),
            dataset_side_to_move=int(stm),
        )
        labeled = LabeledPosition(
            position_id="verify",
            packed_state=bytes(packed),
            value_target=0.0,
            sample_weight=1.0,
            storage_kind="packed",
            dataset_side_to_move=int(stm),
            value_dataset_stm=float(v),
        )
        ids, features, targets, *_ = _featurize_records([labeled])
        if not ids:
            continue
        batch = features_to_torch_batch(features, ids)
        pred = int(model(batch).item())
        rust = int(rec.get("net_eval", rec["eval"]))
        fv = record_to_fv(rec, target)
        direct = record_to_fv(rec, targets[0])
        if (
            fv is not None
            and direct is not None
            and np.allclose(fv[1:], direct[1:], rtol=0, atol=1e-5)
            and abs(pred - rust) <= 1
            and abs(float(targets[0]) - target) < 1e-6
        ):
            ok += 1
    report = {
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "label_perspective_convention": LABEL_PERSPECTIVE_CONVENTION,
        "n_sampled": N,
        "n_passed": ok,
        "fv_len": FV_LEN,
        "passed": ok >= N - 2,
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
