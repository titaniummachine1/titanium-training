#!/usr/bin/env python3
"""Root-cause investigation for H32 control retrain 0-112 vs frozen."""
from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[3]
TRAINING = REPO / "training"
sys.path.insert(0, str(TRAINING))

from titanium_training.data.split import deterministic_train_val_split
from titanium_training.data.teacher_value import (
    load_teacher_value_training_records,
    teacher_value_target,
)
from titanium_training.models.halfpw import Net
from titanium_training.paths import ENGINE_BIN, REPO_ROOT, WEIGHTS_BIN
from titanium_training.training.trainer import HalfPW, QuoridorDataset, wdl_loss

RUN_DIR = REPO / "training/runs/h32_control_retrain"
FROZEN_BIN = REPO / "engine/src/titanium/net_weights_frozen.bin"
TRAINED_BIN = RUN_DIR / "net_weights_control_best.bin"
CACHE_DIR = REPO / "training/data/feature_cache"
SCALE = 400.0
SEED = 0
MAX_SAMPLES = 200_000
VAL_FRAC = 0.05
MIN_VAL = 64


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def param_hash(model: HalfPW) -> str:
    parts = []
    for name, p in sorted(model.state_dict().items()):
        parts.append(p.detach().cpu().numpy().tobytes())
    return hashlib.sha256(b"".join(parts)).hexdigest()


def load_model(weights_path: Path) -> HalfPW:
    m = HalfPW(weights_path)
    m.eval()
    return m


def eval_val_loss(model: HalfPW, val_recs: list[dict]) -> dict:
    ds = QuoridorDataset(val_recs)
    dl = torch.utils.data.DataLoader(ds, batch_size=512, shuffle=False)
    total, n = 0.0, 0
    preds, labels = [], []
    with torch.no_grad():
        for batch in dl:
            out = model(batch)
            loss = wdl_loss(out, batch["target"], SCALE)
            total += loss.item() * len(batch["target"])
            n += len(batch["target"])
            preds.extend(out.tolist())
            labels.extend(batch["target"].tolist())
    preds_a = np.array(preds)
    labels_a = np.array(labels)
    pred_prob = 1 / (1 + np.exp(-preds_a / SCALE))
    return {
        "val_loss": total / n if n else float("inf"),
        "n": n,
        "pred_cp_mean": float(preds_a.mean()),
        "pred_cp_std": float(preds_a.std()),
        "pred_prob_mean": float(pred_prob.mean()),
        "label_mean": float(labels_a.mean()),
        "mae_prob": float(np.abs(pred_prob - labels_a).mean()),
        "corr_pred_label": float(np.corrcoef(preds_a, labels_a * 2 - 1)[0, 1])
        if len(preds_a) > 1
        else 0.0,
    }


def compare_state_dicts(a: HalfPW, b: HalfPW) -> dict:
    sa, sb = a.state_dict(), b.state_dict()
    max_delta = 0.0
    mismatched = []
    for k in sa:
        d = (sa[k].float() - sb[k].float()).abs()
        m = float(d.max())
        if m > 0:
            mismatched.append({"key": k, "max_abs": m, "mean_abs": float(d.mean())})
        max_delta = max(max_delta, m)
    return {"max_abs_delta": max_delta, "mismatched_layers": mismatched[:20], "n_mismatched": len(mismatched)}


def engine_eval_cp(moves: list[str], weights_path: Path | None) -> int:
    env = os.environ.copy()
    if weights_path:
        env["TITANIUM_NET_WEIGHTS_PATH"] = str(weights_path.resolve())
    else:
        env.pop("TITANIUM_NET_WEIGHTS_PATH", None)
    out = subprocess.run(
        [str(ENGINE_BIN), "eval", *moves, "--json"],
        capture_output=True,
        text=True,
        check=True,
        cwd=str(REPO_ROOT),
        env=env,
    )
    return int(json.loads(out.stdout.strip())["eval"])


FIXED_POSITIONS = [
    [],
    ["e2", "e8"],
    ["e2", "e8", "e3", "e7"],
    ["e2", "e8", "e3", "e7", "d3h", "f5v"],
    ["e2", "e8", "e3", "e7", "e4", "e6", "a3h", "d4v"],
    ["e2", "e8", "d2", "f8", "c4h", "g5h"],
    ["e2", "e8", "e3", "e7", "d3h", "f5v", "c2h"],
    ["e2", "e8", "e3", "e7", "e4", "e6", "e5", "d6", "f4h"],
    ["e2", "e8", "e3", "e7", "e4", "e6", "c6h", "f3v", "b5h"],
    ["e2", "e8", "e3", "e7", "e4", "e6", "e5", "d6", "f4h", "g3h"],
    ["d2", "e8", "e3", "d7"],
    ["e2", "d8", "d3", "e7"],
    ["e2", "e8", "f2", "e7", "g2"],
    ["e2", "e8", "c2", "f8"],
    ["e2", "e8", "e3", "e7", "f3", "d7"],
    ["e2", "e8", "e3", "e7", "d3", "f7"],
    ["e2", "e8", "e3", "e7", "e4", "e6", "d4"],
    ["e2", "e8", "e3", "e7", "e4", "e6", "f4"],
    ["a2", "e8", "b2", "e7"],
    ["i2", "e8", "h2", "e7"],
]


def position_outputs() -> list[dict]:
    rows = []
    for moves in FIXED_POSITIONS:
        row = {"moves": moves}
        for label, wp in [
            ("deployed_embedded", None),
            ("frozen", FROZEN_BIN),
            ("trained_control", TRAINED_BIN),
        ]:
            try:
                row[f"engine_{label}_cp"] = engine_eval_cp(moves, wp)
            except Exception as e:
                row[f"engine_{label}_cp"] = f"ERR:{e}"
        rows.append(row)
    return rows


def label_audit(val_recs: list[dict], n: int = 12) -> list[dict]:
    deployed = load_model(WEIGHTS_BIN)
    frozen = load_model(FROZEN_BIN)
    trained_ckpt = torch.load(RUN_DIR / "best.pt", weights_only=False, map_location="cpu")
    trained = load_model(WEIGHTS_BIN)
    trained.load_state_dict(trained_ckpt["model"])
    rng = np.random.default_rng(42)
    idxs = rng.choice(len(val_recs), size=min(n, len(val_recs)), replace=False)
    rows = []
    for i in idxs:
        r = val_recs[int(i)]
        ds = QuoridorDataset([r])
        batch = ds[0]
        b = {k: (v.unsqueeze(0) if hasattr(v, "unsqueeze") else v) for k, v in batch.items()}
        me = int(r["turn"])
        # recover value_i16 from outcome field
        outcome_p0 = float(r["outcome"])
        value_stm = outcome_p0 if me == 0 else -outcome_p0
        value_i16_est = int(round(value_stm * 100))
        target = float(batch["target"])
        rows.append(
            {
                "turn": me,
                "pawn0": r["pawn0"],
                "pawn1": r["pawn1"],
                "value_i16_est": value_i16_est,
                "target_prob": target,
                "deployed_cp": float(deployed(b).item()),
                "frozen_cp": float(frozen(b).item()),
                "trained_cp": float(trained(b).item()),
                "d_me": float(r["d0"] if me == 0 else r["d1"]),
                "d_opp": float(r["d1"] if me == 0 else r["d0"]),
            }
        )
    return rows


def mirror_swap_audit(val_recs: list[dict], n: int = 5) -> list[dict]:
    """Check sign flip when swapping colors on same geometry."""
    from titanium_training.models.halfpw import NET_MIRC

    deployed = load_model(WEIGHTS_BIN)
    rows = []
    for r in val_recs[:n]:
        me = int(r["turn"])
        ds0 = QuoridorDataset([r])
        b0 = {k: (v.unsqueeze(0) if hasattr(v, "unsqueeze") else v) for k, v in ds0[0].items()}
        cp0 = float(deployed(b0).item())
        # swap pawns and flip turn
        r2 = dict(r)
        r2["pawn0"], r2["pawn1"] = NET_MIRC[r["pawn1"]], NET_MIRC[r["pawn0"]]
        r2["turn"] = 1 - me
        r2["d0"], r2["d1"] = r["d1"], r["d0"]
        r2["wl0"], r2["wl1"] = r["wl1"], r["wl0"]
        r2["outcome"] = -float(r["outcome"])
        ds1 = QuoridorDataset([r2])
        b1 = {k: (v.unsqueeze(0) if hasattr(v, "unsqueeze") else v) for k, v in ds1[0].items()}
        cp1 = float(deployed(b1).item())
        rows.append(
            {
                "orig_turn": me,
                "cp_orig": cp0,
                "cp_swapped": cp1,
                "sum_near_zero": abs(cp0 + cp1) < 50,
                "target_orig": float(b0["target"]),
                "target_swapped": float(b1["target"]),
                "targets_mirror": abs(float(b0["target"]) + float(b1["target"]) - 1.0) < 0.01,
            }
        )
    return rows


def cache_vs_engine(n: int = 100) -> dict:
    from build_feature_cache import record_to_fv
    from titanium_training.data.eval_packed import eval_packed_batch_allow_errors
    from titanium_training.data.teacher_value import iter_value_only_rows

    meta = json.loads((CACHE_DIR / "meta.json").read_text())
    data = np.memmap(CACHE_DIR / "positions.bin", dtype="float32", mode="r", shape=(meta["n_total"], 628))
    val_idx = np.load(CACHE_DIR / "val_indices.npy")
    rng = np.random.default_rng(99)
    pick = rng.choice(val_idx, size=min(n, len(val_idx)), replace=False)

    scalar_keys = ["d_me", "d_opp", "w_me", "w_opp", "width_opp"]
    fv_slices = {
        "d_me": 1,
        "d_opp": 2,
        "w_me": 3,
        "w_opp": 4,
        "width_opp": 6,
        "target": 0,
    }
    max_scalar_diff = {k: 0.0 for k in scalar_keys}
    max_target_diff = 0.0
    mismatches = 0
    engine_fail = 0
    rows_sample = []

    dataset_dir = REPO / "training/data/teacher_dataset_good"
    # build position_key -> packed map (bounded scan)
    key_to_packed: dict[bytes, bytes] = {}
    for row in iter_value_only_rows(dataset_dir, max_scan=500_000):
        if row.get("_missing_position"):
            continue
        key_to_packed[bytes(row["position_key"])] = bytes(row["packed_state"])
        if len(key_to_packed) > 400_000:
            break

    for row_i in pick:
        fv_cached = np.array(data[int(row_i)])
        # We don't have position_key in cache row — re-featurize random val via dataset instead
        pass

    # Alternative: compare cache rows by re-running featurization pipeline on teacher val overlap
    records, _ = load_teacher_value_training_records(
        dataset_dir,
        max_samples=MAX_SAMPLES,
        min_samples=64,
        seed=SEED,
        coverage_min=0.999,
    )
    _, val_recs, _ = deterministic_train_val_split(
        records, val_fraction=VAL_FRAC, seed=SEED, min_val=MIN_VAL, min_train=1
    )
    rng2 = np.random.default_rng(99)
    sample_recs = [val_recs[i] for i in rng2.choice(len(val_recs), size=min(n, len(val_recs)), replace=False)]

    for rec in sample_recs:
        ds = QuoridorDataset([rec])
        batch = ds[0]
        target_direct = float(batch["target"])
        # engine fresh eval from packed would need packed_state — not in rec after featurize
        # Compare trainer batch scalars to what record holds
        me = int(rec["turn"])
        d_me = float(rec["d0"] if me == 0 else rec["d1"])
        d_opp = float(rec["d1"] if me == 0 else rec["d0"])
        w_me = float(rec["wl0"] if me == 0 else rec["wl1"])
        w_opp = float(rec["wl1"] if me == 0 else rec["wl0"])
        direct = {
            "d_me": d_me,
            "d_opp": d_opp,
            "w_me": w_me,
            "w_opp": w_opp,
            "width_opp": float(batch["width_opp"]),
            "target": target_direct,
        }
        rows_sample.append(direct)

    # Cache comparison: find cache rows with matching scalars+target (approximate)
    cache_scalar_diffs = []
    for rec in sample_recs[:min(20, len(sample_recs))]:
        ds = QuoridorDataset([rec])
        b = ds[0]
        target = float(b["target"])
        d_me = float(b["d_me"])
        # brute search cache val for close match on target+d_me (expensive but bounded)
        val_data = data[val_idx]
        diff = np.abs(val_data[:, 0] - target) + np.abs(val_data[:, 1] - d_me)
        best = int(val_idx[int(np.argmin(diff))])
        fv = data[best]
        scal_diff = {
            k: abs(float(fv[fv_slices[k]]) - float(b[k if k != "width_opp" else "width_opp"]))
            for k in ["d_me", "d_opp", "w_me", "w_opp", "width_opp", "target"]
        }
        cache_scalar_diffs.append({"cache_row": best, **scal_diff, "max": max(scal_diff.values())})

    return {
        "note": "Control train used live Parquet featurization, not cache. Cache compare uses nearest-neighbor on val cache by target+d_me.",
        "direct_featurization_samples": rows_sample[:5],
        "cache_nearest_scalar_diffs": cache_scalar_diffs,
        "cache_meta_schema": meta.get("schema"),
        "cache_fv_len": meta.get("fv_len"),
    }


def epoch_diagnostics_summary() -> list[dict]:
    rows = []
    for p in sorted(RUN_DIR.glob("epoch_diagnostics_*.json")):
        d = json.loads(p.read_text())
        rows.append(
            {
                "epoch": d["epoch"],
                "val_loss": d["val_loss"],
                "train_loss_end": d["train_loss_end"],
                "pred_mean": d.get("pred_mean"),
                "label_mean": d.get("label_mean"),
                "mae": d.get("mae"),
                "grad_norm_max": d.get("grad_norm_max"),
                "param_norm": d.get("param_norm"),
                "update_over_param_norm": d.get("update_over_param_norm"),
            }
        )
    return rows


def run_match_harness(games: int = 20) -> dict:
    results = {}
    trained = TRAINED_BIN.resolve()
    for name, env_extra, a, b in [
        (
            "control_vs_control",
            {"TITANIUM_NET_WEIGHTS_PATH": str(trained)},
            "titanium-v15",
            "titanium-v15",
        ),
        (
            "frozen_vs_frozen",
            {},
            "titanium-v15-frozen",
            "titanium-v15-frozen",
        ),
        (
            "deployed_vs_frozen",
            {},
            "titanium-v15",
            "titanium-v15-frozen",
        ),
    ]:
        env = os.environ.copy()
        env.update(env_extra)
        env.pop("TITANIUM_NET_WEIGHTS_PATH", None) if name == "deployed_vs_frozen" else None
        if name == "control_vs_control":
            env["TITANIUM_NET_WEIGHTS_PATH"] = str(trained)
        proc = subprocess.run(
            [
                str(ENGINE_BIN),
                "match",
                "--games",
                str(games),
                "--time",
                "2",
                "--openings",
                "book",
                "--a",
                a,
                "--b",
                b,
                "--no-early-stop",
            ],
            capture_output=True,
            text=True,
            cwd=str(REPO),
            env=env,
            timeout=600,
        )
        summary = [
            ln
            for ln in (proc.stdout + proc.stderr).splitlines()
            if "wins" in ln.lower() or "score" in ln.lower() or "STRENGTH" in ln
        ]
        results[name] = {
            "exit_code": proc.returncode,
            "weights_env": env.get("TITANIUM_NET_WEIGHTS_PATH"),
            "weights_sha256": sha256(Path(env["TITANIUM_NET_WEIGHTS_PATH"]))
            if env.get("TITANIUM_NET_WEIGHTS_PATH") and Path(env["TITANIUM_NET_WEIGHTS_PATH"]).is_file()
            else None,
            "summary": summary[-5:],
        }
    return results


def main() -> int:
    report: dict = {"investigation": "h32_control_0-112"}

    report["weight_files"] = {
        "deployed": {"path": str(WEIGHTS_BIN), "sha256": sha256(WEIGHTS_BIN)},
        "frozen": {"path": str(FROZEN_BIN), "sha256": sha256(FROZEN_BIN)},
        "trained_control": {"path": str(TRAINED_BIN), "sha256": sha256(TRAINED_BIN)},
        "deployed_eq_frozen": sha256(WEIGHTS_BIN) == sha256(FROZEN_BIN),
        "control_run_starting_sha": json.loads((RUN_DIR / "control_provenance.json").read_text())[
            "starting_weights_sha256"
        ],
    }

    # 1. Initialization
    epoch0 = load_model(WEIGHTS_BIN)
    frozen_model = load_model(FROZEN_BIN)
    ckpt1 = torch.load(RUN_DIR / "ckpt_epoch0001.pt", weights_only=False, map_location="cpu")
    ckpt20 = torch.load(RUN_DIR / "best.pt", weights_only=False, map_location="cpu")
    ep1 = load_model(WEIGHTS_BIN)
    ep1.load_state_dict(ckpt1["model"])
    ep20 = load_model(WEIGHTS_BIN)
    ep20.load_state_dict(ckpt20["model"])

    report["initialization"] = {
        "control_loaded_path": str(WEIGHTS_BIN),
        "note": "Control used deployed net_weights.bin, NOT net_weights_frozen.bin",
        "epoch0_param_hash": param_hash(epoch0),
        "frozen_param_hash": param_hash(frozen_model),
        "epoch0_vs_deployed_bin": compare_state_dicts(epoch0, epoch0),
        "epoch0_vs_frozen": compare_state_dicts(epoch0, frozen_model),
        "epoch1_vs_epoch0": compare_state_dicts(ep1, epoch0),
        "epoch20_vs_epoch0": compare_state_dicts(ep20, epoch0),
        "ckpt1_epoch_field": ckpt1.get("epoch"),
        "ckpt20_epoch_field": ckpt20.get("epoch"),
    }

    # Reload val set
    records, meta = load_teacher_value_training_records(
        REPO / "training/data/teacher_dataset_good",
        max_samples=MAX_SAMPLES,
        min_samples=64,
        seed=SEED,
        coverage_min=0.999,
    )
    train_recs, val_recs, split_meta = deterministic_train_val_split(
        records, val_fraction=VAL_FRAC, seed=SEED, min_val=MIN_VAL, min_train=1
    )
    report["val_set"] = {"n_val": len(val_recs), "split": split_meta}

    # 2. Baseline loss
    report["baseline_loss"] = {
        "frozen_val": eval_val_loss(frozen_model, val_recs),
        "deployed_epoch0_val": eval_val_loss(epoch0, val_recs),
        "epoch1_val_from_diagnostics": json.loads(
            (RUN_DIR / "epoch_diagnostics_0001.json").read_text()
        )["val_loss"],
        "epoch20_val_from_diagnostics": json.loads(
            (RUN_DIR / "epoch_diagnostics_0020.json").read_text()
        )["val_loss"],
        "epoch1_val_recomputed": eval_val_loss(ep1, val_recs),
        "epoch20_val_recomputed": eval_val_loss(ep20, val_recs),
    }

    # 3. Label orientation
    report["label_audit"] = label_audit(val_recs, n=12)
    report["mirror_swap_audit"] = mirror_swap_audit(val_recs, n=8)

    # 4. Cache compatibility
    report["cache_vs_direct"] = cache_vs_engine(100)

    # 5. Optimization stability
    report["epoch_diagnostics"] = epoch_diagnostics_summary()

    # 6. Position outputs
    report["fixed_position_engine_evals"] = position_outputs()

    # 7. Match harness
    report["match_harness_20g"] = run_match_harness(20)

    # Match weight SHA from original 112 run
    report["original_match_112"] = {
        "trained_weights_sha256": sha256(TRAINED_BIN),
        "log_path": str(RUN_DIR / "match_vs_frozen_112.txt"),
        "result": "A 0 - 112 B (trained via TITANIUM_NET_WEIGHTS_PATH vs frozen embedded)",
    }

    out = RUN_DIR / "root_cause_investigation.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
