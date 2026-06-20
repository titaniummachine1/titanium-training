#!/usr/bin/env python3
"""Bounded value-NNUE smoke: teacher dataset + micro-train + resume + export.

Does not start a multi-hour campaign. Safeguards:
  - max_samples caps loader/dataset checks
  - max_steps caps train.py steps via micro mode + tiny game subset
  - separate output directory under training/runs/
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAINING = ROOT / "training"
sys.path.insert(0, str(TRAINING))
sys.path.insert(0, str(ROOT / "scripts" / "lib"))

from repo_constants import ACTIVE_MANIFEST_SHA256, ACTIVE_TEACHER_DATASET  # noqa: E402
from bundle_lib import verify_active_manifest, verify_provenance  # noqa: E402
from position_store_config import GAME_STORE_DB  # noqa: E402


def _fail(msg: str) -> int:
    print(f"SMOKE FAIL: {msg}", file=sys.stderr)
    return 1


def _pass(msg: str) -> None:
    print(f"SMOKE PASS: {msg}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-samples", type=int, default=256, help="Cap loader/policy lookups")
    ap.add_argument("--max-steps", type=int, default=4, help="Target train steps (micro mode)")
    ap.add_argument("--out-dir", default="", help="Smoke run directory")
    args = ap.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.out_dir) if args.out_dir else TRAINING / "runs" / f"smoke_{stamp}"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "max_samples": args.max_samples,
        "max_steps": args.max_steps,
        "active_manifest_sha256": ACTIVE_MANIFEST_SHA256,
    }
    (run_dir / "smoke_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("=== Phase 1: active teacher dataset ===")
    errors = verify_active_manifest(root=ROOT)
    errors.extend(verify_provenance(root=ROOT))
    if errors:
        return _fail("; ".join(errors))
    manifest = json.loads((ACTIVE_TEACHER_DATASET / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("manifest_hash") != ACTIVE_MANIFEST_SHA256:
        return _fail("manifest hash changed")
    _pass(f"manifest {ACTIVE_MANIFEST_SHA256[:16]}…")

    print("=== Phase 2: artifact + loader smoke ===")
    from teacher_dataset.verify_artifacts import verify_candidate_artifacts
    from teacher_dataset.loader_smoke import run_loader_smoke_audit

    t0 = time.perf_counter()
    artifact = verify_candidate_artifacts(ACTIVE_TEACHER_DATASET, root=ROOT, sample_policy_records=min(100, args.max_samples))
    if not artifact.passed:
        return _fail(f"artifact verification: {artifact.to_dict()}")
    loader = run_loader_smoke_audit(ACTIVE_TEACHER_DATASET, root=ROOT)
    if not loader.passed:
        return _fail(f"loader smoke: {loader.to_dict()}")
    _pass(f"dataset IO checks ({time.perf_counter() - t0:.1f}s)")

    print("=== Phase 3: micro-train (game-store WDL path) ===")
    if not GAME_STORE_DB.is_file():
        return _fail(f"missing game store for train smoke: {GAME_STORE_DB}")

    from datagen import expand_games
    from position_store_lib import connect_db, moves_from_u8_blob

    conn = connect_db(GAME_STORE_DB)
    rows = conn.execute(
        "SELECT g.game_id, g.result, g.source, gp.packed_u8_move_sequence "
        "FROM games g JOIN game_paths gp ON gp.game_id=g.game_id "
        "WHERE g.result IS NOT NULL "
        "ORDER BY g.game_id DESC LIMIT 2"
    ).fetchall()
    conn.close()
    if not rows:
        return _fail("game store has no completed games for micro-train")

    games = [
        (moves_from_u8_blob(row["packed_u8_move_sequence"]), int(row["result"]), str(row["source"]))
        for row in rows
    ]
    records = expand_games(games, min_ply=0, max_ply=40, sample_rate=1.0)
    if not records:
        return _fail("eval-batch produced no training records for smoke games")

    jsonl_path = run_dir / "smoke_records.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for rec in records[: max(16, args.max_samples // 8)]:
            handle.write(json.dumps(rec) + "\n")

    train_cmd = [
        sys.executable,
        str(TRAINING / "train.py"),
        "--data",
        str(jsonl_path),
        "--out-dir",
        str(ckpt_dir),
        "--micro",
        "--epochs",
        "1",
        "--batch",
        "8",
        "--checkpoint-steps",
        str(max(1, args.max_steps)),
        "--cpu",
        "--val-split",
        "0",
    ]
    rc = subprocess.call(train_cmd, cwd=str(ROOT))
    if rc != 0:
        return _fail(f"train.py exited {rc}")
    ckpts = sorted(ckpt_dir.glob("ckpt_step*.pt")) or sorted(ckpt_dir.glob("ckpt_epoch*.pt"))
    if not ckpts:
        return _fail("no checkpoint written")
    _pass(f"checkpoint {ckpts[-1].name}")

    print("=== Phase 4: resume ===")
    resume_cmd = train_cmd + ["--resume", "--ckpt", str(ckpts[-1])]
    rc = subprocess.call(resume_cmd, cwd=str(ROOT))
    if rc != 0:
        return _fail(f"resume exited {rc}")
    _pass("resume OK")

    print("=== Phase 5: export ===")
    export_out = ckpt_dir / "net_weights_smoke.bin"
    rc = subprocess.call(
        [sys.executable, str(TRAINING / "nnue_cli.py"), "export", "--checkpoint", str(ckpts[-1]), "--output", str(export_out)],
        cwd=str(ROOT),
    )
    if rc != 0 or not export_out.is_file():
        return _fail("export failed")
    _pass(f"export -> {export_out.name}")

    meta["finished_at"] = datetime.now(timezone.utc).isoformat()
    meta["manifest_sha256_after"] = manifest.get("manifest_hash")
    meta["checkpoint"] = str(ckpts[-1].relative_to(ROOT))
    (run_dir / "smoke_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nSMOKE COMPLETE — run dir: {run_dir.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
