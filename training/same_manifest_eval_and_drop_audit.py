#!/usr/bin/env python3
"""Frozen validation-manifest same-semantics eval + cohort drop audit.

Reconstructs the seed-0 80/10/10 streaming split used by the mixed epoch,
featurizes the validation keys with the current pipeline, and scores both
parent and candidate networks with the current loss code.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

_TRAINING = Path(__file__).resolve().parent
_REPO = _TRAINING.parent
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

from position_usage_db import open_labels_db
from streaming_db_loader import (
    DEFAULT_LABELS_DB,
    EpochCohorts,
    LabelsRepository,
    FV_LEN,
    features_to_torch_batch,
    interleave_epoch_cohorts,
    sample_epoch_cohorts,
    _featurize_records,
)
from streaming_val_split import split_streaming_epoch_keys
from titanium_training.data.eval_packed import FEATURE_SCHEMA
from titanium_training.training.trainer import (
    TRAINING_SCHEMA,
    HalfPW,
    wdl_loss,
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def eval_net(model: HalfPW, features: np.ndarray, targets: np.ndarray, weights: np.ndarray, scale: float = 400.0) -> dict:
    model.eval()
    bs = 512
    total_w = 0.0
    total_loss = 0.0
    with torch.no_grad():
        for start in range(0, len(features), bs):
            end = min(len(features), start + bs)
            batch = features_to_torch_batch(
                features[start:end],
                [f"row{i}" for i in range(start, end)],
                sample_weights=weights[start:end],
            )
            batch["target"] = torch.from_numpy(targets[start:end].astype(np.float32))
            w = torch.from_numpy(weights[start:end].astype(np.float32))
            out = model(batch)
            loss = wdl_loss(out, batch["target"], scale, w)
            lw = float(w.sum().item())
            total_loss += float(loss.item()) * lw
            total_w += lw
    return {
        "loss": total_loss / max(total_w, 1e-12),
        "n_rows": int(len(features)),
        "weight_mass": total_w,
    }


def slice_metrics(model, features, targets, weights, labels, scale=400.0) -> dict:
    out = {}
    for name in sorted(set(labels)):
        idx = [i for i, lab in enumerate(labels) if lab == name]
        if not idx:
            continue
        out[name] = eval_net(
            model,
            features[idx],
            targets[idx],
            weights[idx],
            scale=scale,
        )
    return out


def main() -> int:
    os.environ.setdefault("TRAINING_PREP_ONLY", "0")
    run_dir = _REPO / "training" / "runs" / "catv5_normalized5_mixed80_10_10_epoch_from_e2_20260716"
    out_dir = run_dir / "same_manifest_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    parent_ckpt = _REPO / "training" / "runs" / "v16" / "accepted" / "epoch_0002.pt"
    parent_bin = _REPO / "training" / "runs" / "v16" / "accepted" / "epoch_0002.bin"
    cand_ckpt = run_dir / "ckpt_epoch0002.pt"
    engine = Path(
        os.environ.get(
            "TITANIUM_ENGINE_BIN",
            str(_REPO / "engine" / "target-catv5-accepted-03856fe" / "release" / "titanium.exe"),
        )
    )
    os.environ["TITANIUM_ENGINE_BIN"] = str(engine)

    if FEATURE_SCHEMA != TRAINING_SCHEMA:
        raise SystemExit(f"schema mismatch {FEATURE_SCHEMA} vs {TRAINING_SCHEMA}")

    # Reconstruct seed-0 cohorts + val split exactly as the epoch launch.
    epoch_size = 120000
    seed = 0
    batch = 512
    con = open_labels_db(DEFAULT_LABELS_DB)
    try:
        cohorts = sample_epoch_cohorts(
            con,
            epoch_size=epoch_size,
            seed=seed,
            anchor_fraction=0.10,
            recent_fraction=0.10,
        )
    finally:
        con.close()

    sampled = {
        "fresh": len(cohorts.fresh),
        "recent": len(cohorts.recent),
        "anchor": len(cohorts.anchor),
    }
    cohort_by_id = {
        **{k: "fresh" for k in cohorts.fresh},
        **{k: "recent" for k in cohorts.recent},
        **{k: "anchor" for k in cohorts.anchor},
    }

    split = {}
    for offset, (name, keys) in enumerate(
        (("fresh", cohorts.fresh), ("recent", cohorts.recent), ("anchor", cohorts.anchor))
    ):
        split[name] = split_streaming_epoch_keys(
            keys, labels_db=DEFAULT_LABELS_DB, val_fraction=0.05, seed=seed + offset
        )
    train_cohorts = EpochCohorts(
        fresh=split["fresh"][0], recent=split["recent"][0], anchor=split["anchor"][0]
    )
    val_keys = split["fresh"][1] + split["recent"][1] + split["anchor"][1]
    train_keys = interleave_epoch_cohorts(train_cohorts, batch_size=batch, seed=seed)

    # Featurize validation keys (and a train drop probe) with current pipeline.
    repo = LabelsRepository(DEFAULT_LABELS_DB)
    try:
        val_rows = repo.load_labeled_positions(val_keys)
        loaded_ids = {r.position_id for r in val_rows}
        load_drop = [k for k in val_keys if k not in loaded_ids]
        ok_ids, features, targets, weights, tiers, phases = _featurize_records(val_rows)
        feat_set = set(ok_ids)
        feat_drop = [r.position_id for r in val_rows if r.position_id not in feat_set]
    finally:
        repo.close()

    # Align cohort/phase labels to retained order
    retained_cohorts = [cohort_by_id.get(i, "unknown") for i in ok_ids]
    retained_phases = list(phases)
    retained_tiers = list(tiers)

    manifest = {
        "schema": FEATURE_SCHEMA,
        "fv_len": FV_LEN,
        "seed": seed,
        "epoch_size": epoch_size,
        "val_keys": val_keys,
        "retained_ids": ok_ids,
        "load_dropped_ids": load_drop,
        "featurize_dropped_ids": feat_drop,
    }
    manifest_blob = json.dumps(
        {
            "schema": FEATURE_SCHEMA,
            "fv_len": FV_LEN,
            "seed": seed,
            "val_keys": val_keys,
            "retained_ids": ok_ids,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    manifest_hash = _sha256_bytes(manifest_blob)
    (out_dir / "validation_manifest.json").write_text(
        json.dumps(
            {
                **{k: v for k, v in manifest.items() if k not in ("val_keys", "retained_ids", "load_dropped_ids", "featurize_dropped_ids")},
                "n_val_keys": len(val_keys),
                "n_retained": len(ok_ids),
                "n_load_dropped": len(load_drop),
                "n_featurize_dropped": len(feat_drop),
                "manifest_hash": manifest_hash,
                "val_keys_path": "validation_manifest_keys.json",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "validation_manifest_keys.json").write_text(
        json.dumps(
            {
                "manifest_hash": manifest_hash,
                "val_keys": val_keys,
                "retained_ids": ok_ids,
                "load_dropped_ids": load_drop,
                "featurize_dropped_ids": feat_drop,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    # Cohort drop audit for the full sampled epoch (train+val keys before featurize).
    # Train retained estimate: featurize a deterministic subsample of train keys is expensive;
    # instead report val drops fully + train sampled/split counts + optimized rows from diagnostics.
    weight_diag = json.loads((run_dir / "epoch_weight_diagnostics_0002.json").read_text(encoding="utf-8"))
    epoch_diag = json.loads((run_dir / "epoch_diagnostics_0002.json").read_text(encoding="utf-8"))

    def cohort_counts(keys: list[str]) -> dict[str, int]:
        c = Counter(cohort_by_id.get(k, "unknown") for k in keys)
        return {name: int(c.get(name, 0)) for name in ("fresh", "recent", "anchor", "unknown")}

    val_retained_by_cohort = Counter(retained_cohorts)
    val_sampled_by_cohort = cohort_counts(val_keys)
    phase_by_cohort: dict[str, dict[str, int]] = defaultdict(lambda: Counter())
    for cohort, phase in zip(retained_cohorts, retained_phases):
        phase_by_cohort[cohort][phase] += 1
    train_total = len(train_keys)
    final_mix = {
        "fresh": 100.0 * len(train_cohorts.fresh) / max(1, train_total),
        "recent": 100.0 * len(train_cohorts.recent) / max(1, train_total),
        "anchor": 100.0 * len(train_cohorts.anchor) / max(1, train_total),
    }
    # "exact" in this pipeline == frozen teacher/anchor cohort + exact-tier labels.
    exact_tier_names = {"external_teacher_anchor", "titanium_anchored"}
    drop_audit = {
        "sampled_total": sampled,
        "sampled_proportions_pct": {
            name: 100.0 * sampled[name] / max(1, sum(sampled.values()))
            for name in ("fresh", "recent", "anchor")
        },
        "after_val_split": {
            "train": {name: len(getattr(train_cohorts, name)) for name in ("fresh", "recent", "anchor")},
            "val": val_sampled_by_cohort,
            "train_rows": len(train_keys),
            "val_rows": len(val_keys),
            "final_80_10_10_proportions_pct": final_mix,
        },
        "validation_featurize": {
            "sampled": len(val_keys),
            "loaded": len(val_rows),
            "retained": len(ok_ids),
            "load_dropped": len(load_drop),
            "featurize_dropped": len(feat_drop),
            "drop_pct": 100.0 * (len(val_keys) - len(ok_ids)) / max(1, len(val_keys)),
            "retained_by_cohort": {k: int(v) for k, v in val_retained_by_cohort.items()},
            "sampled_by_cohort": val_sampled_by_cohort,
            "dropped_by_cohort": {
                name: int(val_sampled_by_cohort.get(name, 0) - val_retained_by_cohort.get(name, 0))
                for name in ("fresh", "recent", "anchor")
            },
            "drop_pct_by_cohort": {
                name: (
                    100.0
                    * (val_sampled_by_cohort.get(name, 0) - val_retained_by_cohort.get(name, 0))
                    / max(1, val_sampled_by_cohort.get(name, 0))
                )
                for name in ("fresh", "recent", "anchor")
            },
            "phase_distribution_retained": dict(Counter(retained_phases)),
            "phase_distribution_by_cohort_retained": {
                name: dict(phase_by_cohort.get(name, {})) for name in ("fresh", "recent", "anchor")
            },
            "tier_distribution_retained": dict(Counter(retained_tiers)),
        },
        "train_optimizer_rows_from_diagnostics": {
            "weight_diag_total_samples": weight_diag.get("total_samples"),
            "epoch_diag_n_samples_logged": epoch_diag.get("n_samples"),
            "phases": weight_diag.get("phases"),
            "tiers": {
                k: {"sample_count": v.get("sample_count"), "phases": v.get("phases")}
                for k, v in weight_diag.get("tiers", {}).items()
            },
            "note": (
                "optimized == retained through online train featurize "
                f"({weight_diag.get('total_samples')} of ~{train_total} train keys)"
            ),
        },
        "exact_label_definition": sorted(exact_tier_names),
        "note": (
            "Train-row featurize drops are observed online during the epoch; "
            "full re-featurize of ~104k train keys is intentionally not repeated here. "
            "Validation drop rates above are exact under the frozen manifest. "
            "Cohort 'anchor' is the exact/teacher cohort in this 80/10/10 experiment."
        ),
    }
    (out_dir / "cohort_drop_audit.json").write_text(json.dumps(drop_audit, indent=2) + "\n", encoding="utf-8")

    # Score parent + candidate (EMA if present else model).
    def load_scorer(ckpt_path: Path, arch_bin: Path, prefer_ema: bool = True) -> HalfPW:
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = HalfPW(arch_bin)
        if prefer_ema and raw.get("ema_state"):
            for name, param in model.named_parameters():
                param.data.copy_(raw["ema_state"][name])
        else:
            model.load_state_dict(raw["model"])
        model.eval()
        return model

    parent = load_scorer(parent_ckpt, parent_bin, prefer_ema=True)
    # Candidate arch from parent bin (same H=32 schema)
    cand = load_scorer(cand_ckpt, parent_bin, prefer_ema=True)

    exact_mask = [t in exact_tier_names for t in retained_tiers]
    exact_idx = [i for i, keep in enumerate(exact_mask) if keep]
    parent_overall = eval_net(parent, features, targets, weights)
    cand_overall = eval_net(cand, features, targets, weights)
    parent_exact = (
        eval_net(parent, features[exact_idx], targets[exact_idx], weights[exact_idx])
        if exact_idx
        else None
    )
    cand_exact = (
        eval_net(cand, features[exact_idx], targets[exact_idx], weights[exact_idx])
        if exact_idx
        else None
    )
    report = {
        "ok": True,
        "manifest_hash": manifest_hash,
        "feature_schema": FEATURE_SCHEMA,
        "fv_len": FV_LEN,
        "engine": str(engine),
        "n_val_keys": len(val_keys),
        "n_retained": len(ok_ids),
        "n_dropped": len(val_keys) - len(ok_ids),
        "n_exact_label_retained": len(exact_idx),
        "parent": {
            "ckpt": str(parent_ckpt),
            "bin": str(parent_bin),
            "sha256_ckpt": _sha256_file(parent_ckpt),
            "sha256_bin": _sha256_file(parent_bin),
            "overall": parent_overall,
            "value_loss": parent_overall["loss"],
            "policy_loss": None,
            "by_phase": slice_metrics(parent, features, targets, weights, retained_phases),
            "by_cohort": slice_metrics(parent, features, targets, weights, retained_cohorts),
            "by_tier": slice_metrics(parent, features, targets, weights, retained_tiers),
            "exact_label_subset": parent_exact,
        },
        "candidate": {
            "ckpt": str(cand_ckpt),
            "sha256_ckpt": _sha256_file(cand_ckpt),
            "overall": cand_overall,
            "value_loss": cand_overall["loss"],
            "policy_loss": None,
            "by_phase": slice_metrics(cand, features, targets, weights, retained_phases),
            "by_cohort": slice_metrics(cand, features, targets, weights, retained_cohorts),
            "by_tier": slice_metrics(cand, features, targets, weights, retained_tiers),
            "exact_label_subset": cand_exact,
        },
        "delta_candidate_minus_parent": cand_overall["loss"] - parent_overall["loss"],
        "policy_loss": None,
        "note": (
            "Value-only HalfPW; no policy head — total_loss == value_loss (WDL BCE). "
            "Exact-label subset = tiers in {external_teacher_anchor, titanium_anchored}. "
            "Cohorts map to fresh/recent/exact(anchor)."
        ),
    }
    (out_dir / "same_manifest_comparison.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "manifest_hash": manifest_hash,
        "n_retained": len(ok_ids),
        "parent_loss": parent_overall["loss"],
        "candidate_loss": cand_overall["loss"],
        "delta": report["delta_candidate_minus_parent"],
        "val_drop_pct_by_cohort": drop_audit["validation_featurize"]["drop_pct_by_cohort"],
        "out_dir": str(out_dir),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
