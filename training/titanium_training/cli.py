#!/usr/bin/env python3
"""Canonical Titanium value-NNUE training interface."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from titanium_training.paths import REPO_ROOT, TRAINING_ROOT

sys.path.insert(0, str(REPO_ROOT / "scripts" / "lib"))

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    training = str(TRAINING_ROOT)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = training if not existing else f"{training}{os.pathsep}{existing}"
    return env


def _run_training_script(script: Path, cmd: list[str]) -> int:
    return subprocess.call(cmd, cwd=str(REPO_ROOT), env=_subprocess_env())


def _load_config(path: Path | None) -> dict:
    if path is None:
        return {}
    if yaml is None:
        raise SystemExit("PyYAML required for --config (pip install pyyaml)")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"invalid config: {path}")
    return data


def cmd_doctor(_args) -> int:
    script = REPO_ROOT / "scripts" / "maintenance" / "repository_doctor.py"
    return subprocess.call([sys.executable, str(script)], cwd=str(REPO_ROOT))


def cmd_verify_dataset(_args) -> int:
    from repo_constants import ACTIVE_MANIFEST_SHA256, ACTIVE_TEACHER_DATASET  # noqa: E402
    from bundle_lib import verify_active_manifest, verify_provenance  # noqa: E402

    errors = verify_active_manifest(root=REPO_ROOT)
    errors.extend(verify_provenance(root=REPO_ROOT))
    manifest_path = ACTIVE_TEACHER_DATASET / "manifest.json"
    if not manifest_path.is_file():
        print("FAIL: active manifest missing")
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(f"Active dataset: {ACTIVE_TEACHER_DATASET.relative_to(REPO_ROOT)}")
    print(f"Manifest SHA256: {manifest.get('manifest_hash')}")
    print(f"Expected SHA256: {ACTIVE_MANIFEST_SHA256}")
    if errors:
        for err in errors:
            print(f"FAIL: {err}")
        return 1
    print("PASS: active teacher dataset verified")
    return 0


def cmd_smoke(args) -> int:
    cfg = _load_config(Path(args.config) if args.config else TRAINING_ROOT / "configs" / "smoke.yaml")
    script = TRAINING_ROOT / "titanium_training" / "validation" / "smoke.py"
    cmd = [sys.executable, str(script)]
    if cfg.get("max_samples"):
        cmd += ["--max-samples", str(cfg["max_samples"])]
    if cfg.get("max_steps"):
        cmd += ["--max-steps", str(cfg["max_steps"])]
    if cfg.get("out_dir"):
        cmd += ["--out-dir", str(cfg["out_dir"])]
    return _run_training_script(script, cmd)


def _trainer_script() -> Path:
    return TRAINING_ROOT / "titanium_training" / "training" / "trainer.py"


def cmd_train(args) -> int:
    cfg = _load_config(Path(args.config) if args.config else None)
    data = cfg.get("teacher_dataset") or cfg.get("data") or str(
        TRAINING_ROOT / "data" / "canonical" / "game_store.db"
    )
    out_dir = cfg.get("out_dir") or str(
        TRAINING_ROOT / "runs" / f"value_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    cmd = [
        sys.executable,
        str(_trainer_script()),
        "--data",
        str(data),
        "--out-dir",
        str(out_dir),
        "--resume",
        "--cpu",
        "--epochs",
        str(cfg.get("epochs", 1)),
        "--batch",
        str(cfg.get("batch", 512)),
        "--lr",
        str(cfg.get("lr", 1e-3)),
        "--checkpoint-steps",
        str(cfg.get("checkpoint_steps", 1000)),
    ]
    if cfg.get("micro"):
        cmd.append("--micro")
    if cfg.get("max_samples"):
        cmd += ["--max-samples", str(cfg["max_samples"])]
    if cfg.get("seed") is not None:
        cmd += ["--seed", str(cfg["seed"])]
    if cfg.get("val_split") is not None:
        cmd += ["--val-split", str(cfg["val_split"])]
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "resolved_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return _run_training_script(_trainer_script(), cmd)


def cmd_smoke_teacher(args) -> int:
    cfg = _load_config(Path(args.config) if args.config else TRAINING_ROOT / "configs" / "value_nnue_smoke.yaml")
    script = TRAINING_ROOT / "titanium_training" / "validation" / "smoke_teacher.py"
    cmd = [sys.executable, str(script), "--config", str(args.config or TRAINING_ROOT / "configs" / "value_nnue_smoke.yaml")]
    return _run_training_script(script, cmd)


def cmd_resume(args) -> int:
    ckpt = Path(args.checkpoint)
    out_dir = ckpt.parent
    cfg_path = out_dir.parent / "resolved_config.json"
    data = TRAINING_ROOT / "data" / "canonical" / "game_store.db"
    if cfg_path.is_file():
        data = Path(json.loads(cfg_path.read_text(encoding="utf-8")).get("data", data))
    cmd = [
        sys.executable,
        str(_trainer_script()),
        "--data",
        str(data),
        "--out-dir",
        str(out_dir),
        "--resume",
        "--ckpt",
        str(ckpt),
        "--cpu",
        "--epochs",
        "1",
    ]
    return _run_training_script(_trainer_script(), cmd)


def cmd_export(args) -> int:
    import torch
    from titanium_training.training.trainer import HalfPW, WEIGHTS

    ckpt = Path(args.checkpoint)
    out = Path(args.output) if args.output else ckpt.parent / "net_weights_export.bin"
    payload = torch.load(ckpt, weights_only=False)
    model = HalfPW(WEIGHTS)
    model.load_state_dict(payload["model"])
    model.save_weights(out)
    print(f"Exported -> {out}")
    return 0


def cmd_preflight(_args) -> int:
    script = TRAINING_ROOT / "titanium_training" / "validation" / "preflight.py"
    return _run_training_script(script, [sys.executable, str(script)])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="Run repository doctor")
    sub.add_parser("verify-dataset", help="Verify active teacher dataset identity")
    sub.add_parser("preflight", help="Engine parity + eval-batch preflight")

    smoke = sub.add_parser("smoke", help="End-to-end value-NNUE smoke")
    smoke.add_argument("--config", default=str(TRAINING_ROOT / "configs" / "smoke.yaml"))

    smoke_teacher = sub.add_parser("smoke-teacher", help="Teacher-value dataset smoke (real Parquet path)")
    smoke_teacher.add_argument("--config", default=str(TRAINING_ROOT / "configs" / "value_nnue_smoke.yaml"))

    train = sub.add_parser("train", help="Start value-NNUE training (game-store WDL path)")
    train.add_argument("--config", required=True)

    resume = sub.add_parser("resume", help="Resume from checkpoint")
    resume.add_argument("--checkpoint", required=True)

    export = sub.add_parser("export", help="Export engine-format weights from checkpoint")
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--output")

    args = ap.parse_args()
    handlers = {
        "doctor": cmd_doctor,
        "verify-dataset": cmd_verify_dataset,
        "preflight": cmd_preflight,
        "smoke": cmd_smoke,
        "smoke-teacher": cmd_smoke_teacher,
        "train": cmd_train,
        "resume": cmd_resume,
        "export": cmd_export,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
