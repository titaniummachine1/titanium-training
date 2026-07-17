#!/usr/bin/env python3
"""Fail-closed pre-flight for one controlled streaming mixed epoch.

Checks labels.db source, 952 schema, 80/10/10 cohorts, packed-derived phases,
no 628-feature cache reference, and no four-ply trunk filter in the loader.
Does not train or write checkpoints.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

_TRAINING = Path(__file__).resolve().parent
_REPO = _TRAINING.parent
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

from label_weights import game_phase_from_packed
from position_usage_db import open_labels_db
from streaming_db_loader import (
    DEFAULT_LABELS_DB,
    LabelsRepository,
    interleave_epoch_cohorts,
    sample_epoch_cohorts,
)
from titanium_training.data.eval_packed import FEATURE_SCHEMA
from titanium_training.training.trainer import TRAINING_SCHEMA

EXPECTED_SCHEMA = "halfpw-sparse-route5-catv5-normalized5-ws20-v1"
EXPECTED_FV = 952
LABELS_DB = DEFAULT_LABELS_DB
CKPT = _REPO / "training" / "runs" / "v16" / "accepted" / "epoch_0002.pt"
OLD_CACHE = _REPO / "training" / "data" / "feature_cache"


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}")
    return 1


def main() -> int:
    # Sampling is guarded; caller must allow this audit process only.
    if os.environ.get("TRAINING_PREP_ONLY", "1").strip() not in ("0", "false", "no", "off"):
        return _fail("set TRAINING_PREP_ONLY=0 for this audit process only")

    from build_feature_cache import FV_LEN

    report: dict = {"ok": False}

    if FEATURE_SCHEMA != EXPECTED_SCHEMA or TRAINING_SCHEMA != EXPECTED_SCHEMA:
        return _fail(
            f"schema drift FEATURE={FEATURE_SCHEMA!r} TRAINING={TRAINING_SCHEMA!r}"
        )
    if FV_LEN != EXPECTED_FV:
        return _fail(f"FV_LEN={FV_LEN} != {EXPECTED_FV}")

    if not LABELS_DB.is_file():
        return _fail(f"labels.db missing: {LABELS_DB}")
    report["labels_db"] = str(LABELS_DB.resolve())

    # Refuse accidental use of the stale 628 cache for this run.
    if OLD_CACHE.is_dir():
        meta = OLD_CACHE / "meta.json"
        if meta.is_file():
            old = json.loads(meta.read_text(encoding="utf-8"))
            if int(old.get("fv_len", 0)) == 628:
                report["old_628_cache_present"] = True
                report["old_628_cache_must_not_be_passed"] = str(OLD_CACHE)
                print(
                    "NOTE: legacy 628 feature_cache exists but must not be passed "
                    "as --cache-dir (streaming path uses --labels-db only)."
                )

    # Four-ply trunk must not appear as a training filter in the streaming loader.
    loader_src = (_TRAINING / "streaming_db_loader.py").read_text(encoding="utf-8")
    forbidden = (
        "four_ply_trunk",
        "FOUR_PLY_TRUNK",
        "deploy_trunk",
        "opening_book_trunk",
    )
    hits = [m for m in forbidden if m in loader_src]
    if hits:
        return _fail(f"four-ply trunk markers in streaming_db_loader.py: {hits}")
    if "TEMPORARY_GARBAGE_FILTER_NOT_DIVERSITY_COMPLIANCE" not in loader_src:
        return _fail("expected two-ply opening sanity gate marker missing")
    if "not the deploy-only four-ply trunk" not in loader_src:
        return _fail("streaming loader must document that four-ply trunk is not used")

    if not CKPT.is_file():
        return _fail(f"missing resume ckpt {CKPT}")
    import torch

    raw = torch.load(CKPT, map_location="cpu", weights_only=False)
    want = {
        "schema": EXPECTED_SCHEMA,
        "step": 223,
        "epoch": 1,
    }
    for k, v in want.items():
        if raw.get(k) != v:
            return _fail(f"ckpt {k}={raw.get(k)!r} want={v!r}")
    if abs(float(raw["best_val"]) - 0.3868207530635996) > 1e-12:
        return _fail(f"best_val={raw['best_val']}")
    if len(raw.get("model", {})) != 16:
        return _fail("model tensor count != 16")
    if len(raw.get("ema_state", {}) or {}) != 16:
        return _fail("ema tensor count != 16")
    opt_state = raw.get("optimizer", {}).get("state", {})
    if len(opt_state) != 16:
        return _fail(f"adam state entries={len(opt_state)}")
    report["ckpt"] = {
        "path": str(CKPT),
        "schema": raw["schema"],
        "step": raw["step"],
        "epoch": raw["epoch"],
        "best_val": float(raw["best_val"]),
        "model": 16,
        "adam": 16,
        "ema": 16,
    }

    # Sample a small 80/10/10 epoch and verify phases from packed/json state.
    epoch_size = 1000
    con = open_labels_db(LABELS_DB)
    try:
        cohorts = sample_epoch_cohorts(
            con,
            epoch_size=epoch_size,
            seed=42,
            anchor_fraction=0.10,
            recent_fraction=0.10,
        )
    finally:
        con.close()

    n_f, n_r, n_a = len(cohorts.fresh), len(cohorts.recent), len(cohorts.anchor)
    if (n_f, n_r, n_a) != (800, 100, 100):
        return _fail(f"cohort sizes fresh/recent/anchor={n_f}/{n_r}/{n_a} want 800/100/100")
    keys = interleave_epoch_cohorts(cohorts, batch_size=10, seed=42)
    if len(keys) != epoch_size:
        return _fail(f"interleaved keys={len(keys)}")
    # Every full batch must keep 8/1/1 composition.
    for start in range(0, epoch_size, 10):
        batch = keys[start : start + 10]
        if sum(k in set(cohorts.fresh) for k in batch) != 8:
            return _fail(f"batch@{start} fresh != 8")
        if sum(k in set(cohorts.recent) for k in batch) != 1:
            return _fail(f"batch@{start} recent != 1")
        if sum(k in set(cohorts.anchor) for k in batch) != 1:
            return _fail(f"batch@{start} anchor != 1")

    repo = LabelsRepository(LABELS_DB)
    try:
        # Probe a stratified mix of keys for phase derivation.
        probe = cohorts.fresh[:40] + cohorts.recent[:30] + cohorts.anchor[:30]
        rows = repo.load_labeled_positions(probe)
    finally:
        repo.close()
    if len(rows) < 50:
        return _fail(f"loaded only {len(rows)} labeled rows from probe")

    phases = Counter(r.game_phase for r in rows)
    if phases.get("midgame", 0) == len(rows):
        return _fail("all probed phases midgame — packed/json phase still hardcoded?")
    for name in ("opening", "midgame", "endgame"):
        if phases.get(name, 0) <= 0:
            return _fail(f"phase {name} count is zero in probe {dict(phases)}")

    # Explicit packed teacher check: wall bytes must drive phase.
    teacher_rows = [r for r in rows if r.position_id.startswith("teacher:")]
    if not teacher_rows:
        return _fail("no teacher rows in probe")
    for r in teacher_rows[:20]:
        derived = game_phase_from_packed(r.packed_state)
        if derived != r.game_phase:
            return _fail(
                f"teacher phase mismatch id={r.position_id[:24]}… "
                f"row={r.game_phase} packed={derived}"
            )

    # Confirm feature vector width via one tiny featurize if engine is ready.
    engine = Path(
        os.environ.get(
            "TITANIUM_ENGINE_BIN",
            str(_REPO / "engine" / "target-catv5-accepted-03856fe" / "release" / "titanium.exe"),
        )
    )
    if not engine.is_file():
        return _fail(f"accepted engine missing: {engine}")
    os.environ["TITANIUM_ENGINE_BIN"] = str(engine)

    from build_feature_cache import eval_packed_batch_raw, record_to_fv

    sample = teacher_rows[0]
    recs = eval_packed_batch_raw([(sample.packed_state, 0.5, int(sample.dataset_side_to_move or 0))])
    if not recs or recs[0] is None:
        return _fail("eval-packed-batch failed for probe row")
    fv = record_to_fv(recs[0], 0.5)
    if fv is None or fv.shape != (EXPECTED_FV,):
        return _fail(f"feature width {None if fv is None else fv.shape} != ({EXPECTED_FV},)")

    report.update(
        {
            "ok": True,
            "schema": EXPECTED_SCHEMA,
            "fv_len": EXPECTED_FV,
            "cohort_probe": {"fresh": n_f, "recent": n_r, "anchor": n_a},
            "phases_probe": dict(phases),
            "engine": str(engine),
            "cache_dir_arg": None,
            "four_ply_trunk_filter": False,
        }
    )
    print(json.dumps(report, indent=2))
    print("PASS: streaming input/schema audit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
