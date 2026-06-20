#!/usr/bin/env python3
"""Canonical Titanium value-NNUE training interface.

Thin wrapper around existing training modules — one documented entrypoint per action.

Examples:
    python training/nnue_cli.py doctor
    python training/nnue_cli.py verify-dataset
    python training/nnue_cli.py smoke --config training/configs/smoke.yaml
    python training/nnue_cli.py train --config training/configs/value_nnue_local.yaml
    python training/nnue_cli.py resume --checkpoint training/runs/<run_id>/checkpoints/best.pt
    python training/nnue_cli.py export --checkpoint training/runs/<run_id>/checkpoints/best.pt
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAINING = ROOT / "training"
sys.path.insert(0, str(TRAINING))
sys.path.insert(0, str(ROOT / "scripts" / "lib"))

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore


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
    script = ROOT / "scripts" / "maintenance" / "repository_doctor.py"
    return subprocess.call([sys.executable, str(script)], cwd=str(ROOT))


def cmd_verify_dataset(_args) -> int:
    from repo_constants import ACTIVE_MANIFEST_SHA256, ACTIVE_TEACHER_DATASET  # noqa: E402
    from bundle_lib import verify_active_manifest, verify_provenance  # noqa: E402

    errors = verify_active_manifest(root=ROOT)
    errors.extend(verify_provenance(root=ROOT))
    manifest_path = ACTIVE_TEACHER_DATASET / "manifest.json"
    if not manifest_path.is_file():
        print("FAIL: active manifest missing")
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(f"Active dataset: {ACTIVE_TEACHER_DATASET.relative_to(ROOT)}")
    print(f"Manifest SHA256: {manifest.get('manifest_hash')}")
    print(f"Expected SHA256: {ACTIVE_MANIFEST_SHA256}")
    if errors:
        for err in errors:
            print(f"FAIL: {err}")
        return 1
    print("PASS: active teacher dataset verified")
    return 0


def cmd_smoke(args) -> int:
    cfg = _load_config(Path(args.config) if args.config else TRAINING / "configs" / "smoke.yaml")
    script = TRAINING / "value_nnue_smoke.py"
    cmd = [sys.executable, str(script)]
    if cfg.get("max_samples"):
        cmd += ["--max-samples", str(cfg["max_samples"])]
    if cfg.get("max_steps"):
        cmd += ["--max-steps", str(cfg["max_steps"])]
    if cfg.get("out_dir"):
        cmd += ["--out-dir", str(cfg["out_dir"])]
    return subprocess.call(cmd, cwd=str(ROOT))


def cmd_train(args) -> int:
    cfg = _load_config(Path(args.config) if args.config else None)
    data = cfg.get("data") or str(TRAINING / "data" / "canonical" / "game_store.db")
    out_dir = cfg.get("out_dir") or str(TRAINING / "runs" / f"value_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    cmd = [
        sys.executable,
        str(TRAINING / "train.py"),
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
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "resolved_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return subprocess.call(cmd, cwd=str(ROOT))


def cmd_resume(args) -> int:
    ckpt = Path(args.checkpoint)
    out_dir = ckpt.parent
    cfg_path = out_dir.parent / "resolved_config.json"
    data = TRAINING / "data" / "canonical" / "game_store.db"
    if cfg_path.is_file():
        data = Path(json.loads(cfg_path.read_text(encoding="utf-8")).get("data", data))
    cmd = [
        sys.executable,
        str(TRAINING / "train.py"),
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
    return subprocess.call(cmd, cwd=str(ROOT))


def cmd_export(args) -> int:
    import torch

    ckpt = Path(args.checkpoint)
    out = Path(args.output) if args.output else ckpt.parent / "net_weights_export.bin"
    payload = torch.load(ckpt, weights_only=False)
    sys.path.insert(0, str(TRAINING))
    from train import HalfPW, WEIGHTS

    model = HalfPW(WEIGHTS)
    model.load_state_dict(payload["model"])
    model.save_weights(out)
    print(f"Exported -> {out}")
    return 0


def cmd_preflight(_args) -> int:
    return subprocess.call([sys.executable, str(TRAINING / "validate_train_ready.py")], cwd=str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="Run repository doctor")
    sub.add_parser("verify-dataset", help="Verify active teacher dataset identity")
    sub.add_parser("preflight", help="Engine parity + eval-batch preflight")

    smoke = sub.add_parser("smoke", help="End-to-end value-NNUE smoke")
    smoke.add_argument("--config", default=str(TRAINING / "configs" / "smoke.yaml"))

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
        "train": cmd_train,
        "resume": cmd_resume,
        "export": cmd_export,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
