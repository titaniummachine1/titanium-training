#!/usr/bin/env python3
"""Bounded teacher-value smoke: promoted dataset + packed-state HalfPW path."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from titanium_training.paths import ACTIVE_TEACHER_DATASET, REPO_ROOT, TRAINING_ROOT

sys.path.insert(0, str(REPO_ROOT / "scripts" / "lib"))
from bundle_lib import verify_active_manifest, verify_provenance  # noqa: E402


def _fail(msg: str) -> int:
    print(f"SMOKE-TEACHER FAIL: {msg}", file=sys.stderr)
    return 1


def _pass(msg: str) -> None:
    print(f"SMOKE-TEACHER PASS: {msg}")


def _clean_smoke_outputs(ckpt_dir: Path) -> None:
    for pattern in (
        "best.pt",
        "ckpt_step*.pt",
        "ckpt_epoch*.pt",
        "epoch_diagnostics_*.json",
        "net_weights_teacher_smoke.bin",
        "net_weights_best.bin",
    ):
        for path in ckpt_dir.glob(pattern):
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(TRAINING_ROOT / "configs" / "value_nnue_smoke.yaml"))
    args = ap.parse_args()

    import yaml

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    run_dir = Path(cfg.get("out_dir", TRAINING_ROOT / "runs" / "smoke_teacher_latest"))
    if not run_dir.is_absolute():
        run_dir = (REPO_ROOT / run_dir).resolve()
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    _clean_smoke_outputs(ckpt_dir)

    expected_sha = cfg.get("active_manifest_sha256")
    print("=== Phase 1: dataset identity ===")
    errors = verify_active_manifest(root=REPO_ROOT)
    errors.extend(verify_provenance(root=REPO_ROOT))
    if errors:
        return _fail("; ".join(errors))
    manifest = json.loads((ACTIVE_TEACHER_DATASET / "manifest.json").read_text(encoding="utf-8"))
    if expected_sha and manifest.get("manifest_hash") != expected_sha:
        return _fail("manifest hash mismatch")
    _pass(f"manifest {manifest.get('manifest_hash', '')[:16]}…")

    print("=== Phase 2: featurize + train ===")
    t0 = time.perf_counter()
    train_cmd = [
        sys.executable,
        str(TRAINING_ROOT / "titanium_training" / "training" / "trainer.py"),
        "--data",
        str(cfg.get("data", "training/data/teacher_dataset")),
        "--out-dir",
        str(ckpt_dir),
        "--cpu",
        "--epochs",
        str(cfg.get("epochs", 2)),
        "--batch",
        str(cfg.get("batch", 32)),
        "--lr",
        str(cfg.get("lr", 5e-4)),
        "--checkpoint-steps",
        str(cfg.get("checkpoint_steps", 32)),
        "--val-split",
        str(cfg.get("val_split", 0.08)),
        "--max-samples",
        str(cfg.get("max_samples", 4096)),
        "--seed",
        str(cfg.get("seed", 0)),
        "--min-val",
        str(cfg.get("min_val", 64)),
        "--coverage-min",
        str(cfg.get("coverage_min", 1.0)),
    ]
    env = dict(__import__("os").environ)
    env["PYTHONPATH"] = str(TRAINING_ROOT)
    rc = subprocess.call(train_cmd, cwd=str(REPO_ROOT), env=env)
    if rc != 0:
        return _fail(f"trainer exited {rc}")
    meta_path = ckpt_dir / "run_metadata.json"
    if not meta_path.is_file():
        return _fail("missing run_metadata.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("synthetic_fallback_used"):
        return _fail("synthetic fallback flagged true")
    if meta.get("featurization_mode") != "packed-state-direct":
        return _fail(f"unexpected featurization mode: {meta.get('featurization_mode')}")
    if int(meta.get("validation_count", 0)) <= 0:
        return _fail("validation_count is zero")
    if float(meta.get("coverage_percentage", 0)) < 100.0:
        return _fail(f"coverage below 100%: {meta.get('coverage_percentage')}")
    _pass(
        f"train {meta.get('train_count')} val {meta.get('validation_count')} "
        f"({time.perf_counter()-t0:.1f}s)"
    )

    ckpts = sorted(ckpt_dir.glob("ckpt_step*.pt")) or sorted(ckpt_dir.glob("ckpt_epoch*.pt"))
    if not ckpts:
        return _fail("no checkpoint written")
    initial_ckpt = ckpts[0]

    print("=== Phase 3: resume ===")
    resume_cmd = train_cmd + ["--resume", "--ckpt", str(ckpts[-1])]
    rc = subprocess.call(resume_cmd, cwd=str(REPO_ROOT), env=env)
    if rc != 0:
        return _fail(f"resume exited {rc}")
    ckpts = sorted(ckpt_dir.glob("ckpt_step*.pt")) or sorted(ckpt_dir.glob("ckpt_epoch*.pt"))
    _pass("resume OK")

    print("=== Phase 4: export + parity ===")
    export_out = ckpt_dir / "net_weights_teacher_smoke.bin"
    rc = subprocess.call(
        [
            sys.executable,
            str(TRAINING_ROOT / "nnue_cli.py"),
            "export",
            "--checkpoint",
            str(ckpts[-1]),
            "--output",
            str(export_out),
        ],
        cwd=str(REPO_ROOT),
        env=env,
    )
    if rc != 0 or not export_out.is_file():
        return _fail("export failed")

    from titanium_training.validation.export_parity import verify_export_parity

    parity = verify_export_parity(ckpts[-1], export_out)
    if not parity.passed:
        return _fail(
            f"export parity failed (max_err={parity.max_parity_error}): "
            + "; ".join(parity.details[:5])
        )
    _pass(f"export parity max_err={parity.max_parity_error}")

    cfg_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    run_meta = {
        "run_type": "smoke_teacher",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(Path(args.config).as_posix()),
        "config_sha256": cfg_hash,
        "dataset_path": meta.get("dataset_path"),
        "dataset_manifest_sha256": meta.get("dataset_manifest_sha256"),
        "featurization_mode": meta.get("featurization_mode"),
        "engine_commit": meta.get("engine_commit"),
        "feature_schema": meta.get("feature_schema"),
        "candidate_labels": meta.get("candidate_labels"),
        "featurized_rows": meta.get("featurized_samples"),
        "coverage_percentage": meta.get("coverage_percentage"),
        "train_count": meta.get("train_count"),
        "validation_count": meta.get("validation_count"),
        "target_definition": meta.get("target_definition"),
        "feature_source": meta.get("feature_source"),
        "exported_net_parity_status": "PASS",
        "maximum_parity_error": parity.max_parity_error,
        "synthetic_fallback_used": False,
        "initial_checkpoint": str(initial_ckpt.relative_to(REPO_ROOT)).replace("\\", "/"),
        "checkpoint": str(ckpts[-1].relative_to(REPO_ROOT)).replace("\\", "/"),
        "export_path": str(export_out.relative_to(REPO_ROOT)).replace("\\", "/"),
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    print(f"\nSMOKE-TEACHER COMPLETE — run dir: {run_dir.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
