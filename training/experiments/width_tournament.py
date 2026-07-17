#!/usr/bin/env python3
"""Width comparison: does raw NPS (narrow net) beat capacity (wide net)?

Trains three width variants for exactly one epoch on the SAME fresh,
never-visited data slice (2,526,565 positions available -- this experiment
uses a plain slice of that pool, so it doesn't touch the position_usage
visit-cap at all: every position used here gets exactly 1 visit, same as a
normal single training cycle):

  h32  -- from engine/src/titanium/net_weights_frozen.bin (ACE v13's OWN
          weights, already h=32 and independently strong per this session's
          earlier match results -- net2net only widens, it can't cleanly
          shrink our own h=96 net down to 32, so this is the correct base,
          not a hack).
  h96  -- from runs/v16/accepted/epoch_0042.bin (current real base).
  h128 -- from the net2net-widened epoch_0042 (96->128), reusing the same
          widened blob already generated earlier this session for the NPS
          benchmark.

Then a real ~400-game round-robin tournament among the three trained
variants, using streaming_epoch_validation.py's real EngineSession-based
match harness (per-side independent weight files, same mechanism as every
other real strength-gate match in this pipeline) -- NOT a custom/simplified
match implementation.

Isolated: writes only under training/runs/width_tournament/. Never touches
position_usage, the accepted checkpoint chain, or deployed engine weights.
Nothing gets auto-promoted; this reports results for a human decision.
"""
from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

from typing import Any

import numpy as np

_TRAINING = Path(__file__).resolve().parents[1]
_REPO = _TRAINING.parent
sys.path.insert(0, str(_TRAINING))

from db_import import LABELS_DB_PATH  # noqa: E402
from build_feature_cache import record_to_fv, FV_LEN, make_fingerprint  # noqa: E402
from label_perspective import json_row_target_prob  # noqa: E402
from titanium_training.validation.engine_identity import load_expected_stamp  # noqa: E402

EXP_DIR = _TRAINING / "runs" / "width_tournament"
CACHE_DIR = EXP_DIR / "cache"
TRAINER = _TRAINING / "titanium_training" / "training" / "trainer.py"

FROZEN_BASE = _REPO / "engine" / "src" / "titanium" / "net_weights_frozen.bin"
H96_BASE = _TRAINING / "runs" / "v16" / "accepted" / "epoch_0042.bin"
H128_BASE = Path(
    r"C:\Users\TERMIN~1\AppData\Local\Temp\claude\C--gitProjects-Quoridor-best-AI"
    r"\e26da05a-f0f3-496a-9733-175af874ee17\scratchpad\test_h128.bin"
)

N_TRAIN_POSITIONS = 100_000  # matches the coordinator's own default --epoch-size
TOURNAMENT_GAMES_PER_PAIR = 134  # 3 pairs * 134 = 402, close to the requested 400
SEED = 20260710
VAL_FRACTION = 0.05


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(EXP_DIR / "tournament.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def featurize() -> tuple[int, np.ndarray]:
    """Same cache-fingerprint fix as catheat_ab_retrain.py (meta.json must
    carry the real schema/engine-sha/label-perspective fields or trainer.py's
    check_fingerprint() rejects it instantly -- learned the hard way earlier
    this session). Only samples from genuinely never-visited positions, so
    this experiment can never conflict with the visit-cap concern."""
    meta_path = CACHE_DIR / "meta.json"
    if meta_path.is_file():
        n_existing = json.loads(meta_path.read_text())["n_total"]
        log(f"featurize: cache already built, n_total={n_existing} -- skipping")
        return n_existing, np.load(CACHE_DIR / "train_indices.npy")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(LABELS_DB_PATH))
    cur = con.cursor()
    cur.execute("BEGIN")
    cur.execute(
        """
        SELECT p.pos_key, p.position_data, l.value_stm
        FROM positions p
        JOIN labels l ON p.pos_key = l.pos_key
        JOIN position_usage u ON u.pos_key = p.pos_key
        WHERE u.training_visits = 0
        ORDER BY RANDOM()
        LIMIT ?
        """,
        (N_TRAIN_POSITIONS,),
    )
    rows = cur.fetchall()
    con.execute("COMMIT")
    n_total = len(rows)
    log(f"featurize: {n_total} never-visited rows sampled")

    tmp_bin = CACHE_DIR / "positions.bin.tmp"
    mm = np.memmap(tmp_bin, dtype="float32", mode="w+", shape=(n_total, FV_LEN))
    pos_keys: list[str] = []
    n_bad = 0
    for i, (pos_key, blob, value_stm) in enumerate(rows):
        fv = None
        try:
            rec = json.loads(blob)
            target = json_row_target_prob(float(value_stm))
            fv = record_to_fv(rec, target)
        except Exception:
            fv = None
        if fv is None:
            n_bad += 1
            fv = np.zeros(FV_LEN, dtype=np.float32)
        mm[i] = fv
        pos_keys.append(pos_key)
        if (i + 1) % 20000 == 0:
            log(f"featurize: {i+1}/{n_total}")
    mm.flush()
    del mm
    con.close()
    tmp_bin.replace(CACHE_DIR / "positions.bin")

    np.save(CACHE_DIR / "row_position_keys.npy", np.array(pos_keys, dtype="<U16"))
    stamp = load_expected_stamp() or {}
    n_val = int(n_total * VAL_FRACTION)
    fp = make_fingerprint(stamp, n_total, n_total - n_val, n_val, manifest_hash="width_tournament")
    meta_path.write_text(json.dumps(fp, indent=2))

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(n_total).astype(np.int32)
    np.save(CACHE_DIR / "val_indices.npy", idx[:n_val])
    train_idx = idx[n_val:]
    np.save(CACHE_DIR / "train_indices.npy", train_idx)

    log(f"featurize done: {n_total} rows ({n_bad} unparseable -> zero row)")
    return n_total, train_idx


def train_one_epoch(name: str, weights_in: Path, train_idx: np.ndarray) -> Path:
    out_dir = EXP_DIR / name
    result_bin = out_dir / "net_weights_best.bin"
    if result_bin.is_file():
        log(f"train[{name}]: already done -> {result_bin}")
        return result_bin
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(CACHE_DIR / "train_indices.npy", train_idx)
    cmd = [
        sys.executable, str(TRAINER),
        "--cache-dir", str(CACHE_DIR),
        "--out-dir", str(out_dir),
        "--weights", str(weights_in),
        "--epochs", "1",
        "--batch", "512",
        "--lr", "0.0005",
        "--checkpoint-steps", "999999",
        "--patience", "0",
        "--cpu",
        "--log-every", "1000",
        "--log-interval-sec", "30",
    ]
    log(f"train[{name}]: starting weights_in={weights_in.name}")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(_TRAINING), capture_output=True, text=True)
    (out_dir / "stdout.log").write_text(proc.stdout)
    (out_dir / "stderr.log").write_text(proc.stderr)
    if proc.returncode != 0 or not result_bin.is_file():
        raise RuntimeError(f"trainer failed for {name} (rc={proc.returncode}); see {out_dir}/stderr.log")
    log(f"train[{name}]: done in {time.time()-t0:.0f}s -> {result_bin}")
    return result_bin


def tournament(bins: dict[str, Path]) -> dict[str, Any]:
    from streaming_epoch_validation import _match_candidate_vs_parent

    names = list(bins.keys())
    results: dict[str, Any] = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            key = f"{a}_vs_{b}"
            log(f"tournament: {key} ({TOURNAMENT_GAMES_PER_PAIR} games)")
            r = _match_candidate_vs_parent(
                candidate_bin=bins[a],
                parent_bin=bins[b],
                games=TOURNAMENT_GAMES_PER_PAIR,
                time_sec=1.0,
            )
            results[key] = r
            log(f"tournament: {key} -> {json.dumps({k: r.get(k) for k in ('score','wins','draws','losses','games')})}")
    return results


def main() -> int:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    log("width tournament starting")
    log(f"bases: h32={FROZEN_BASE} h96={H96_BASE} h128={H128_BASE}")
    for p in (FROZEN_BASE, H96_BASE, H128_BASE):
        if not p.is_file():
            log(f"FATAL: base weights missing: {p}")
            return 1

    n_total, train_idx = featurize()

    bins = {
        "h32": train_one_epoch("h32", FROZEN_BASE, train_idx),
        "h96": train_one_epoch("h96", H96_BASE, train_idx),
        "h128": train_one_epoch("h128", H128_BASE, train_idx),
    }

    results = tournament(bins)
    (EXP_DIR / "results.json").write_text(json.dumps(results, indent=2, default=str))
    log("=== FINAL RESULTS ===")
    for k, r in results.items():
        log(f"  {k}: score={r.get('score')} w/d/l={r.get('wins')}/{r.get('draws')}/{r.get('losses')}")
    log("width tournament complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
