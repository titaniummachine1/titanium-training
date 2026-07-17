"""Run exactly one supervised EMA continuation in a quarantine directory."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--background", action="store_true")
    args = ap.parse_args()
    out = ROOT / "training/runs/oracle_horizon_pilot_v1/continuation_e3"
    source = ROOT / "training/runs/oracle_horizon_pilot_v1/cycle1/labels_primary.jsonl"
    parent_bin = ROOT / "training/runs/v16/accepted/epoch_0003.bin"
    parent_pt = ROOT / "training/runs/v16/accepted/epoch_0003.pt"
    before = {"bin": sha256(parent_bin), "pt": sha256(parent_pt)}
    if before["bin"] != "869ad228cfea8bb8964d98d05d6cf5e67a21b27661a36259a3976f60d486be56":
        raise RuntimeError("accepted epoch3 bin sha mismatch; refusing continuation")
    build = [sys.executable, str(ROOT / "training/oracle_horizon/build_continuation_manifest.py"),
             "--source", str(source), "--out-dir", str(out)]
    subprocess.run(build, cwd=ROOT, check=True)
    audit = json.loads((out / "PRETRAIN_AUDIT.json").read_text(encoding="utf-8"))
    if audit.get("status") != "PASS":
        raise RuntimeError("PRETRAIN_AUDIT is not PASS")
    manifest = json.loads((out / "TRAIN_MANIFEST.json").read_text(encoding="utf-8"))
    env = os.environ.copy()
    env.update({
        "TRAINING_PREP_ONLY": "0", "PYTHONPATH": str(ROOT / "training"),
        "TITANIUM_BOOK_MODE": "off",
        "TITANIUM_ENGINE_BIN": str(ROOT / "engine/target-catv5-accepted-03856fe/release/titanium.exe"),
        "RUSTFLAGS": "-C target-cpu=native",
    })
    cmd = [
        sys.executable, str(ROOT / "training/titanium_training/training/trainer.py"),
        "--labels-db", "training/data/canonical/labels.db",
        "--weights", "training/runs/v16/accepted/epoch_0003.bin",
        "--ckpt", "training/runs/v16/accepted/epoch_0003.pt", "--resume",
        "--epochs", "1", "--batch", "512", "--lr", "0.0002", "--weight-decay", "0.00001",
        "--grad-clip", "1.0", "--stream-epoch-size", str(manifest["stream_epoch_size"]),
        "--stream-featurize-chunk", "2048", "--stream-anchor-fraction", "0.10",
        "--stream-oracle-jsonl", str(out / "train_oracle.jsonl"), "--stream-oracle-fraction", "0.10",
        "--mirror-prob", "0.5", "--ema-decay", "0.99", "--val-split", "0.05",
        "--checkpoint-steps", "999999", "--patience", "0", "--cpu", "--defer-usage-commit",
        "--log-every", "10", "--log-interval-sec", "30", "--out-dir", str(out),
        # Streaming NNUE path: same as continuous_pool / start_local_game_pool_detached.
        # Python HCE vs engine parity is not a gate for this recipe (export parity still runs).
        "--no-parity",
    ]
    log = out / "continuation.log"
    out.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        (out / "CONTINUATION_PID").write_text(str(proc.pid) + "\n", encoding="utf-8")
        print(json.dumps({"pid": proc.pid, "log": str(log), "out_dir": str(out)}))
        if args.background:
            return 0
        rc = proc.wait()
    after = {"bin": sha256(parent_bin), "pt": sha256(parent_pt)}
    if after != before:
        raise RuntimeError("accepted epoch3 artifact changed during continuation")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
