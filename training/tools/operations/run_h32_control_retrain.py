#!/usr/bin/env python3
"""Orchestrate H32 control retrain: metadata, smoke, train, post-eval (no deploy)."""
from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
TRAINING = REPO / "training"
sys.path.insert(0, str(TRAINING))

from titanium_training.paths import ENGINE_BIN, REPO_ROOT, WEIGHTS_BIN
from titanium_training.validation.export_parity import verify_export_parity

SMOKE_CFG = TRAINING / "configs" / "value_nnue_control_cache_smoke.yaml"
TRAIN_CFG = TRAINING / "configs" / "value_nnue_control_retrain.yaml"
START_WEIGHTS = WEIGHTS_BIN
FROZEN_WEIGHTS = REPO / "engine" / "src" / "titanium" / "net_weights_frozen.bin"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _run(cmd: list[str], *, env: dict | None = None, cwd: Path | None = None) -> None:
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    run_env = os.environ.copy()
    training = str(TRAINING)
    existing = run_env.get("PYTHONPATH", "")
    run_env["PYTHONPATH"] = training if not existing else f"{training}{os.pathsep}{existing}"
    if env:
        run_env.update(env)
    subprocess.run(cmd, cwd=str(cwd or REPO), env=run_env, check=True)


def _load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def record_provenance(out_path: Path) -> dict:
    cfg = _load_yaml(TRAIN_CFG)
    cache_meta = {}
    cm = TRAINING / "data" / "feature_cache" / "meta.json"
    if cm.is_file():
        cache_meta = json.loads(cm.read_text(encoding="utf-8"))
    manifest_path = REPO / str(cfg["teacher_dataset"]) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stamp = json.loads((TRAINING / "data" / "engine_stamp.json").read_text(encoding="utf-8"))
    prov = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "control_run": "h32_corrected_trainer_forward",
        "config_path": str(TRAIN_CFG.relative_to(REPO)),
        "config": cfg,
        "dataset_manifest_sha256": manifest.get("manifest_hash"),
        "dataset_manifest_path": str(manifest_path.relative_to(REPO)),
        "feature_cache_meta": cache_meta,
        "random_seed": cfg.get("seed", 0),
        "starting_weights_path": str(START_WEIGHTS.relative_to(REPO)),
        "starting_weights_sha256": _sha256(START_WEIGHTS),
        "frozen_baseline_sha256": _sha256(FROZEN_WEIGHTS) if FROZEN_WEIGHTS.is_file() else None,
        "engine_stamp": stamp,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(prov, indent=2), encoding="utf-8")
    return prov


def weight_drift_report(start: Path, trained: Path) -> dict:
    a = np.array(list(struct.unpack(f"<{len(start.read_bytes()) // 8}d", start.read_bytes())))
    b = np.array(list(struct.unpack(f"<{len(trained.read_bytes()) // 8}d", trained.read_bytes())))
    n = min(len(a), len(b))
    diff = b[:n] - a[:n]
    return {
        "start_sha256": _sha256(start),
        "trained_sha256": _sha256(trained),
        "f64_count": n,
        "max_abs_delta": float(np.max(np.abs(diff))),
        "mean_abs_delta": float(np.mean(np.abs(diff))),
        "l2_delta": float(np.linalg.norm(diff)),
    }


def collect_epoch_losses(run_dir: Path) -> list[dict]:
    rows = []
    for p in sorted(run_dir.glob("epoch_diagnostics_*.json")):
        rows.append(json.loads(p.read_text(encoding="utf-8")))
    return rows


def post_train_eval(run_dir: Path, export_bin: Path) -> dict:
    report: dict = {}
    ckpt = run_dir / "best.pt"
    if not ckpt.is_file():
        ckpt = max(run_dir.glob("ckpt_epoch*.pt"), key=lambda p: p.stat().st_mtime, default=None)
    if ckpt is None or not ckpt.is_file():
        raise FileNotFoundError(f"no checkpoint in {run_dir}")

    _run(
        [
            sys.executable,
            str(TRAINING / "nnue_cli.py"),
            "export",
            "--checkpoint",
            str(ckpt),
            "--output",
            str(export_bin),
        ]
    )
    parity = verify_export_parity(ckpt, export_bin)
    report["export_parity"] = {
        "passed": parity.passed,
        "max_parity_error_cp": parity.max_parity_error,
        "details": parity.details,
    }

    _run([sys.executable, str(TRAINING / "titanium_training" / "validation" / "parity_check.py")])
    _run([sys.executable, "-m", "pytest", str(TRAINING / "tests" / "test_trainer_scalar_parity.py"), "-q"])

    report["weight_drift"] = weight_drift_report(START_WEIGHTS, export_bin)

    bench_bin = REPO / "engine" / "target" / "release" / "search_bench.exe"
    if bench_bin.is_file():
        env = os.environ.copy()
        env["TITANIUM_NET_WEIGHTS_PATH"] = str(export_bin.resolve())
        for label, extra in [
            ("baseline_embedded", {}),
            ("trained_weights", {"TITANIUM_NET_WEIGHTS_PATH": str(export_bin.resolve())}),
        ]:
            e = env.copy()
            e.update(extra)
            if label == "baseline_embedded":
                e.pop("TITANIUM_NET_WEIGHTS_PATH", None)
            proc = subprocess.run(
                [str(bench_bin), "time", "--sec", "2", "--runs", "3"],
                capture_output=True,
                text=True,
                cwd=str(REPO),
                env=e,
            )
            report[f"search_bench_{label}"] = {
                "exit_code": proc.returncode,
                "stdout_tail": proc.stdout.strip().splitlines()[-5:],
                "stderr_tail": proc.stderr.strip().splitlines()[-5:],
            }

        instr_bin = REPO / "engine" / "target" / "release" / "search_bench.exe"
        if instr_bin.is_file():
            e = env.copy()
            e["TITANIUM_NET_WEIGHTS_PATH"] = str(export_bin.resolve())
            proc = subprocess.run(
                [str(instr_bin), "instr", "--sec", "2", "--runs", "1"],
                capture_output=True,
                text=True,
                cwd=str(REPO),
                env=e,
            )
            report["nnue_instr"] = {
                "exit_code": proc.returncode,
                "stdout": proc.stdout.strip()[-4000:],
            }

    match_env = os.environ.copy()
    match_env["TITANIUM_NET_WEIGHTS_PATH"] = str(export_bin.resolve())
    match_log = run_dir / "match_vs_frozen_112.txt"
    proc = subprocess.run(
        [
            str(ENGINE_BIN),
            "match",
            "--games",
            "112",
            "--time",
            "5",
            "--openings",
            "book",
            "--a",
            "titanium-v15",
            "--b",
            "titanium-v15-frozen",
            "--no-early-stop",
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO),
        env=match_env,
    )
    match_log.write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
    report["match_112"] = {
        "exit_code": proc.returncode,
        "log": str(match_log.relative_to(REPO)),
        "summary_lines": [ln for ln in (proc.stdout + proc.stderr).splitlines() if "STRENGTH" in ln or "Elo" in ln or "score" in ln.lower()][-10:],
    }
    return report


def main() -> int:
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("--phase", choices=["preflight", "smoke", "train", "post", "all"], default="all")
    args = ap.parse_args()

    run_dir = REPO / "training" / "runs" / "h32_control_retrain"
    smoke_dir = REPO / "training" / "runs" / "h32_control_cache_smoke"
    prov_path = run_dir / "control_provenance.json"
    report_path = run_dir / "control_report.json"
    export_bin = run_dir / "net_weights_control_best.bin"

    if args.phase in ("preflight", "all"):
        _run([sys.executable, "-m", "titanium_training.validation.engine_identity", "--write"])
        record_provenance(prov_path)
        _run([sys.executable, str(TRAINING / "nnue_cli.py"), "preflight"])

    if args.phase in ("smoke", "all"):
        smoke_dir.mkdir(parents=True, exist_ok=True)
        _run([sys.executable, str(TRAINING / "nnue_cli.py"), "train", "--config", str(SMOKE_CFG)])
        smoke_ckpt = smoke_dir / "best.pt"
        if not smoke_ckpt.is_file():
            smoke_ckpt = max(smoke_dir.glob("ckpt_epoch*.pt"), key=lambda p: p.stat().st_mtime)
        smoke_export = smoke_dir / "net_weights_smoke_export.bin"
        _run(
            [
                sys.executable,
                str(TRAINING / "nnue_cli.py"),
                "export",
                "--checkpoint",
                str(smoke_ckpt),
                "--output",
                str(smoke_export),
            ]
        )
        smoke_parity = verify_export_parity(smoke_ckpt, smoke_export)
        if not smoke_parity.passed:
            raise SystemExit(f"smoke export parity failed: max_err={smoke_parity.max_parity_error}")

    if args.phase in ("train", "all"):
        run_dir.mkdir(parents=True, exist_ok=True)
        if not prov_path.is_file():
            record_provenance(prov_path)
        _run([sys.executable, str(TRAINING / "nnue_cli.py"), "train", "--config", str(TRAIN_CFG)])

    if args.phase in ("post", "all"):
        epochs = collect_epoch_losses(run_dir)
        best = min(epochs, key=lambda r: r["val_loss"]) if epochs else {}
        post = post_train_eval(run_dir, export_bin)
        report = {
            "epochs": epochs,
            "best_epoch_by_val_loss": best.get("epoch"),
            "best_val_loss": best.get("val_loss"),
            "post_eval": post,
        }
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
