#!/usr/bin/env python3
"""Audit + preflight for controlled continuous_pool restart."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_TRAINING = _REPO / "training"
sys.path.insert(0, str(_TRAINING))

import numpy as np

GAMES_DB = _TRAINING / "data" / "canonical" / "games.db"
LABELS_DB = _TRAINING / "data" / "canonical" / "labels.db"
CACHE_DIR = _TRAINING / "data" / "feature_cache"
GOOD = _TRAINING / "data" / "teacher_dataset_good"
LEGACY = _TRAINING / "data" / "teacher_dataset"
QUARANTINE = _TRAINING / "data" / "quarantine"
RUN_DIR = _TRAINING / "runs" / "value_oracle"
ENGINE_WEIGHTS = _REPO / "engine" / "src" / "titanium" / "net_weights.bin"
FROZEN = _REPO / "engine" / "src" / "titanium" / "net_weights_frozen.bin"
BEST = RUN_DIR / "net_weights_best.bin"
STATE = _TRAINING / "data" / "overnight_logs" / "continuous_pool_state.json"
FV_LEN = 547


def sha16(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def sha_full(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def quarantine_corrupt_legacy_teacher() -> dict:
    """Move corrupt teacher_dataset parquet to quarantine if unreadable."""
    QUARANTINE.mkdir(parents=True, exist_ok=True)
    out = {"quarantined": [], "good_ok": False, "legacy_ok": False}
    try:
        import pyarrow.parquet as pq

        good_lbl = GOOD / "labels" / "part-00000.parquet"
        if good_lbl.is_file():
            pq.read_schema(good_lbl)
            out["good_ok"] = True
        leg_lbl = LEGACY / "labels" / "part-00000.parquet"
        if leg_lbl.is_file():
            try:
                pq.read_schema(leg_lbl)
                out["legacy_ok"] = True
            except Exception as exc:
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                qdir = QUARANTINE / f"teacher_dataset_corrupt_{stamp}"
                qdir.mkdir(parents=True, exist_ok=True)
                manifest = {
                    "reason": str(exc),
                    "excluded_from_training": True,
                    "source": str(LEGACY),
                    "quarantined_at": stamp,
                }
                for sub in ("labels", "positions", "observations"):
                    src = LEGACY / sub
                    if src.is_dir():
                        for f in src.glob("*.parquet"):
                            dest = qdir / sub
                            dest.mkdir(exist_ok=True)
                            f.rename(dest / f.name)
                            out["quarantined"].append(str(f))
                (qdir / "QUARANTINE_MANIFEST.json").write_text(
                    json.dumps(manifest, indent=2), encoding="utf-8"
                )
                out["legacy_ok"] = False
    except Exception as exc:
        out["error"] = str(exc)
    return out


def audit_datasets() -> dict:
    import pyarrow.parquet as pq

    def scan_parquet(path: Path) -> dict:
        if not path.is_file():
            return {"exists": False}
        try:
            t = pq.read_table(path, columns=["value_i16", "source_cohort", "position_key"])
            vals = t.column("value_i16").to_pylist()
            nulls = sum(v is None for v in vals)
            nonfin = 0
            for v in vals:
                if v is not None and not (-100 <= int(v) <= 100):
                    nonfin += 1
            return {
                "exists": True,
                "rows": t.num_rows,
                "null_labels": nulls,
                "out_of_range": nonfin,
                "mtime": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
                "readable": True,
            }
        except Exception as exc:
            return {"exists": True, "readable": False, "error": str(exc)}

    games = {}
    if GAMES_DB.is_file():
        con = sqlite3.connect(GAMES_DB)
        games["total"] = con.execute("SELECT COUNT(*) FROM games").fetchone()[0]
        games["by_source"] = dict(con.execute("SELECT source, COUNT(*) FROM games GROUP BY source").fetchall())
        games["pool_games"] = con.execute("SELECT COUNT(*) FROM games WHERE game_id LIKE 'pool_%'").fetchone()[0]
        games["mtime"] = datetime.fromtimestamp(GAMES_DB.stat().st_mtime, tz=timezone.utc).isoformat()
        con.close()

    return {
        "games_db": str(GAMES_DB),
        "games": games,
        "good_labels": scan_parquet(GOOD / "labels" / "part-00000.parquet"),
        "legacy_labels": scan_parquet(LEGACY / "labels" / "part-00000.parquet"),
        "quarantine": quarantine_corrupt_legacy_teacher(),
    }


def audit_cache() -> dict:
    meta_path = CACHE_DIR / "meta.json"
    if not meta_path.is_file():
        return {"ok": False, "reason": "meta.json missing"}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    pos = CACHE_DIR / "positions.bin"
    ti = CACHE_DIR / "train_indices.npy"
    vi = CACHE_DIR / "val_indices.npy"
    ok = pos.is_file() and ti.is_file() and vi.is_file()
    from build_feature_cache import check_fingerprint

    fp_ok, fp_reason = check_fingerprint(CACHE_DIR)
    return {
        "cache_dir": str(CACHE_DIR),
        "meta": meta,
        "files_ok": ok,
        "fingerprint_ok": fp_ok,
        "fingerprint_reason": fp_reason,
        "positions_bytes": pos.stat().st_size if pos.is_file() else 0,
        "mtime": datetime.fromtimestamp(meta_path.stat().st_mtime, tz=timezone.utc).isoformat(),
    }


def audit_checkpoints() -> dict:
    ckpts = sorted(RUN_DIR.glob("ckpt_epoch*.pt"))
    state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.is_file() else {}
    return {
        "deployed_engine": {"path": str(ENGINE_WEIGHTS), "sha256": sha_full(ENGINE_WEIGHTS)},
        "frozen": {"path": str(FROZEN), "sha256": sha_full(FROZEN)},
        "best": {"path": str(BEST), "sha256": sha_full(BEST)},
        "checkpoints": [str(p) for p in ckpts],
        "latest_ckpt": str(ckpts[-1]) if ckpts else None,
        "state": state,
        "deploy_matches_best": sha_full(ENGINE_WEIGHTS) == sha_full(BEST) if BEST.is_file() else None,
    }


def run_preflight() -> dict:
    results: dict = {"steps": [], "ok": True}
    errors: list[str] = []

    def step(name: str, fn):
        try:
            fn()
            results["steps"].append({"name": name, "ok": True})
        except Exception as exc:
            results["steps"].append({"name": name, "ok": False, "error": str(exc)})
            errors.append(f"{name}: {exc}")
            results["ok"] = False

    # second instance guard
    def test_lock():
        import subprocess
        import time
        from pool_lock import acquire_pool_lock, release_pool_lock

        release_pool_lock()
        holder = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import time,sys; from pathlib import Path; "
                f"sys.path.insert(0, r'{_TRAINING}'); "
                "from pool_lock import acquire_pool_lock; acquire_pool_lock(); time.sleep(20)",
            ],
            cwd=str(_REPO),
        )
        try:
            time.sleep(1.5)
            try:
                acquire_pool_lock()
                raise AssertionError("second acquire should fail")
            except RuntimeError:
                pass
        finally:
            holder.terminate()
            holder.wait(timeout=10)
            release_pool_lock()

    step("single_instance_lock", test_lock)

    # cache batch load
    def test_cache():
        meta = json.loads((CACHE_DIR / "meta.json").read_text(encoding="utf-8"))
        assert meta["fv_len"] == FV_LEN
        n = meta["n_total"]
        mm = np.memmap(CACHE_DIR / "positions.bin", dtype="float32", mode="r", shape=(n, FV_LEN))
        batch = mm[:32]
        assert np.isfinite(batch).all()

    step("cache_load_batch", test_cache)

    # one train step
    def test_train_step():
        import torch
        from titanium_training.training.trainer import CachedDataset, HalfPW, wdl_loss

        torch.set_num_threads(1)
        ds = CachedDataset(CACHE_DIR, "train")
        sample = ds[0]
        model = HalfPW(str(BEST if BEST.is_file() else ENGINE_WEIGHTS))
        opt = torch.optim.Adam(model.parameters(), lr=5e-4)
        batch = {k: v.unsqueeze(0) if hasattr(v, "unsqueeze") else v for k, v in sample.items()}
        out = model(batch)
        loss = wdl_loss(out, batch["target"], 400.0)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite loss")
        opt.zero_grad()
        loss.backward()
        opt.step()

    step("forward_backward_step", test_train_step)

    # validation subset
    def test_val_subset():
        import torch
        from titanium_training.training.trainer import CachedDataset, HalfPW, wdl_loss

        ds = CachedDataset(CACHE_DIR, "val")
        assert len(ds) > 0
        model = HalfPW(str(BEST if BEST.is_file() else ENGINE_WEIGHTS))
        model.eval()
        losses = []
        for i in range(min(16, len(ds))):
            sample = ds[i]
            batch = {k: v.unsqueeze(0) if hasattr(v, "unsqueeze") else v for k, v in sample.items()}
            with torch.no_grad():
                out = model(batch)
                loss = wdl_loss(out, batch["target"], 400.0)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite val loss at {i}")
            losses.append(float(loss))
        results["val_loss_mean"] = sum(losses) / len(losses)

    step("validation_subset", test_val_subset)

    # short self-play + persist
    def test_selfplay_persist():
        from self_play_overnight import DEFAULT_CURRENT, play_one_game
        from db_import import GAMES_DB_PATH, LABELS_DB_PATH, open_db, GAMES_SCHEMA, LABELS_SCHEMA, write_batch

        gid = f"preflight_{int(datetime.now(timezone.utc).timestamp())}"
        r = play_one_game(gid, 0.5, DEFAULT_CURRENT, DEFAULT_CURRENT, False, True)
        assert r and r.get("moves"), "no moves from self-play"
        games_db = open_db(GAMES_DB_PATH, GAMES_SCHEMA)
        labels_db = open_db(LABELS_DB_PATH, LABELS_SCHEMA)
        try:
            write_batch(games_db, labels_db, [(gid, r["moves"], r["outcome_p0"], None, "overnight_selfplay")], 512, 1)
        finally:
            games_db.close()
            labels_db.close()
        con = sqlite3.connect(GAMES_DB)
        assert con.execute("SELECT 1 FROM games WHERE game_id=?", (gid,)).fetchone()
        con.close()

    step("selfplay_persist_roundtrip", test_selfplay_persist)

    # weight export size + temp copy
    def test_weights():
        import shutil
        import tempfile
        from titanium_training.training.trainer import HalfPW, NET_WEIGHT_F64S

        path = BEST if BEST.is_file() else ENGINE_WEIGHTS
        nbytes = path.stat().st_size
        expected = NET_WEIGHT_F64S * 8
        if nbytes != expected:
            raise RuntimeError(f"weight size {nbytes} != trainer NET_WEIGHT_F64S*8 ({expected})")
        HalfPW(str(path))  # schema/load round-trip
        tmp = Path(tempfile.mkdtemp(prefix="preflight_weights_"))
        try:
            dest = tmp / "export_test.bin"
            shutil.copy2(path, dest)
            HalfPW(str(dest))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    step("weight_schema_size", test_weights)

    # no orphan titanium
    def test_no_titanium():
        if sys.platform == "win32":
            out = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq titanium.exe"],
                capture_output=True,
                text=True,
            )
            if "titanium.exe" in out.stdout.lower():
                raise RuntimeError("orphan titanium.exe processes present")

    step("no_orphan_titanium", test_no_titanium)

    results["errors"] = errors
    return results


def main() -> int:
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "datasets": audit_datasets(),
        "cache": audit_cache(),
        "checkpoints": audit_checkpoints(),
        "preflight": run_preflight(),
        "effective_config": {
            "workers": 4,
            "time_sec": 2.0,
            "batch_size": 512,
            "epochs_per_trigger": 1,
            "lr": 0.0005,
            "optimizer": "Adam",
            "train_trigger": "positions_since_epoch >= 40000 (fallback games >= 512)",
            "cache_dir": str(CACHE_DIR),
            "teacher_dataset": "teacher_dataset_good via titanium_training.paths",
            "promotion": "maybe_deploy_after_train gate (no blind deploy)",
            "initial_epoch": False,
            "cache_rebuild": "only when fingerprint invalid",
            "log": str(_TRAINING / "data" / "overnight_logs" / "continuous_pool.log"),
            "lock": str(_TRAINING / "data" / "overnight_logs" / "continuous_pool.lock.json"),
        },
    }
    out_path = _TRAINING / "data" / "overnight_logs" / "controlled_restart_audit.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["preflight"]["ok"] and report["cache"]["fingerprint_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
