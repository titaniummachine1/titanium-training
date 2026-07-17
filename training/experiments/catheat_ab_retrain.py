#!/usr/bin/env python3
"""A/B retrain experiment after the cat_heat gradient-flow fix.

Question: after unblocking cat_heat's gradient (it was structurally stuck at
zero across all 36 prior accepted epochs), is it better to digest the full
corpus as many small sequential epochs (matching the live coordinator's normal
incremental cadence) or as one single big epoch over everything at once?

Isolated from production: reads labels.db read-only for featurization, writes
only under training/runs/catheat_ab_experiment/. Never touches position_usage,
the accepted checkpoint chain, or BEST_WEIGHTS/ENGINE_WEIGHTS. Both arms warm-
start from the SAME current accepted weights (epoch 36) -- not from random
init -- so the comparison isolates "how to feed cat_heat its first real
gradient", not "does 36 epochs of prior work matter" (it does; discarding it
was explicitly rejected earlier in this conversation).

Data pool: all (positions JOIN labels) rows in labels.db, i.e. the full
corpus with position_usage.retired ignored entirely (position_usage is a
SQLite-table-level tracking mechanism specific to the live --labels-db
streaming path; this script uses the separate --cache-dir path, which has its
own independent, freshly-built retirement state -- so "ignore retirement" is
free here, no DB mutation needed, sidestepping the earlier-blocked bulk
UPDATE entirely).
"""
from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_TRAINING = Path(__file__).resolve().parents[1]
_REPO = _TRAINING.parent
sys.path.insert(0, str(_TRAINING))

from db_import import LABELS_DB_PATH  # noqa: E402
from build_feature_cache import record_to_fv, FV_LEN, make_fingerprint  # noqa: E402
from label_perspective import json_row_target_prob  # noqa: E402
from titanium_training.validation.engine_identity import load_expected_stamp  # noqa: E402

EXP_DIR = _TRAINING / "runs" / "catheat_ab_experiment"
CACHE_DIR = EXP_DIR / "cache"
BASE_WEIGHTS = _TRAINING / "runs" / "v16" / "accepted" / "epoch_0036.bin"
TRAINER = _TRAINING / "titanium_training" / "training" / "trainer.py"
N_CHUNKS = 30
VAL_FRACTION = 0.05
SEED = 20260708

LOG = EXP_DIR / "experiment.log"


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def featurize() -> int:
    meta_path = CACHE_DIR / "meta.json"
    if meta_path.is_file():
        n_existing = json.loads(meta_path.read_text())["n_total"]
        log(f"featurize: cache already built, n_total={n_existing} -- skipping")
        return n_existing

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(LABELS_DB_PATH))
    cur = con.cursor()
    # labels.db is live -- oracle_importer/coordinator keep writing to it while
    # this runs. A bare COUNT(*) then a separate SELECT can race: rows land
    # between the two, so the SELECT returns more rows than the memmap was
    # sized for (this crashed the first run with an off-by-one IndexError at
    # row 1,211,989). A single read transaction gives both queries the same
    # consistent snapshot (WAL mode: readers see the DB as of BEGIN, even
    # while writers continue) -- count and rows-returned are then guaranteed
    # to match exactly.
    cur.execute("BEGIN")
    cur.execute("SELECT COUNT(*) FROM positions p JOIN labels l ON p.pos_key = l.pos_key")
    n_total = cur.fetchone()[0]
    log(f"featurize: {n_total} rows to process (snapshot-consistent)")

    tmp_bin = CACHE_DIR / "positions.bin.tmp"
    mm = np.memmap(tmp_bin, dtype="float32", mode="w+", shape=(n_total, FV_LEN))

    cur.execute("SELECT p.pos_key, p.position_data, l.value_stm FROM positions p JOIN labels l ON p.pos_key = l.pos_key")
    BATCH = 20000
    i = 0
    n_bad = 0
    n_overflow = 0
    pos_keys: list[str] = []
    t0 = time.time()
    while True:
        rows = cur.fetchmany(BATCH)
        if not rows:
            break
        for pos_key, blob, value_stm in rows:
            if i >= n_total:
                # Defense in depth only -- should be unreachable given the
                # snapshot transaction above, but fail loud-and-safe instead
                # of crashing if it ever does happen.
                n_overflow += 1
                continue
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
            i += 1
        if i % (BATCH * 5) == 0 or i == n_total:
            elapsed = time.time() - t0
            rate = i / max(elapsed, 1e-6)
            log(f"featurize: {i}/{n_total} ({elapsed:.0f}s, {rate:.0f} rows/s, bad={n_bad})")
    con.execute("COMMIT")
    if n_overflow:
        log(f"featurize: WARNING {n_overflow} rows beyond snapshot count were dropped (unexpected)")
    mm.flush()
    del mm
    con.close()
    tmp_bin.replace(CACHE_DIR / "positions.bin")

    # trainer.py's --cache-dir loader validates a full fingerprint (schema,
    # engine sha, label-perspective convention) via check_fingerprint(), plus
    # a row_position_keys.npy sidecar whose length must match n_total exactly
    # -- a bare {"n_total":...} meta.json (what the first two runs shipped)
    # fails this immediately with "schema mismatch: cache=None current=...",
    # which killed the run silently (no retry loop) right after featurize
    # finished both times. Build a real, compliant fingerprint instead.
    np.save(CACHE_DIR / "row_position_keys.npy", np.array(pos_keys, dtype="<U16"))
    stamp = load_expected_stamp() or {}
    n_val_est = int(i * VAL_FRACTION)
    fp = make_fingerprint(
        stamp, i, i - n_val_est, n_val_est,
        manifest_hash="catheat_ab_experiment",
    )
    meta_path.write_text(json.dumps(fp, indent=2))
    log(f"featurize done: {i} rows ({n_bad} unparseable -> zero row) in {time.time()-t0:.0f}s")
    return i


def build_splits(n_total: int) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n_total).astype(np.int32)
    n_val = int(n_total * VAL_FRACTION)
    val_idx = perm[:n_val]
    train_pool = perm[n_val:]
    chunks = np.array_split(train_pool, N_CHUNKS)
    np.save(CACHE_DIR / "val_indices.npy", val_idx)
    log(f"splits: n_val={len(val_idx)} n_train_pool={len(train_pool)} n_chunks={N_CHUNKS} "
        f"(~{len(train_pool)//N_CHUNKS}/chunk)")
    return train_pool, chunks, val_idx


def run_trainer(weights_in: Path, out_dir: Path, train_indices: np.ndarray) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(CACHE_DIR / "train_indices.npy", train_indices.astype(np.int32))
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
        "--log-every", "500",
        "--log-interval-sec", "30",
    ]
    env_desc = f"weights_in={weights_in.name} out_dir={out_dir.name} n_train={len(train_indices)}"
    log(f"trainer: starting {env_desc}")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(_REPO), capture_output=True, text=True)
    elapsed = time.time() - t0
    (out_dir / "stdout.log").write_text(proc.stdout)
    (out_dir / "stderr.log").write_text(proc.stderr)
    if proc.returncode != 0:
        log(f"trainer: FAILED rc={proc.returncode} ({env_desc}) -- see {out_dir}/stderr.log")
        raise RuntimeError(f"trainer failed for {out_dir}")
    result_bin = out_dir / "net_weights_best.bin"
    log(f"trainer: done {env_desc} in {elapsed:.0f}s -> {result_bin.name}")
    return result_bin


def run_experiment_a(train_pool_chunks: list[np.ndarray]) -> Path:
    log("=== experiment A: many small sequential epochs (simulated incremental arrival) ===")
    a_dir = EXP_DIR / "A_incremental"
    current = BASE_WEIGHTS
    for i, chunk in enumerate(train_pool_chunks):
        step_dir = a_dir / f"step_{i:02d}"
        current = run_trainer(current, step_dir, chunk)
    final = a_dir / "final.bin"
    final.write_bytes(current.read_bytes())
    log(f"experiment A final -> {final}")
    return final


def run_experiment_b(train_pool: np.ndarray) -> Path:
    log("=== experiment B: one big epoch over the full pool at once ===")
    b_dir = EXP_DIR / "B_bigepoch"
    result = run_trainer(BASE_WEIGHTS, b_dir, train_pool)
    final = b_dir / "final.bin"
    final.write_bytes(result.read_bytes())
    log(f"experiment B final -> {final}")
    return final


def compare(a_final: Path, b_final: Path) -> None:
    log("=== comparison: direct match A vs B, and each vs baseline epoch 36 ===")
    from streaming_epoch_validation import _match_candidate_vs_parent

    results = {
        "A_vs_baseline": _match_candidate_vs_parent(candidate_bin=a_final, parent_bin=BASE_WEIGHTS, games=60),
        "B_vs_baseline": _match_candidate_vs_parent(candidate_bin=b_final, parent_bin=BASE_WEIGHTS, games=60),
        "A_vs_B": _match_candidate_vs_parent(candidate_bin=a_final, parent_bin=b_final, games=80),
    }
    (EXP_DIR / "comparison_result.json").write_text(json.dumps(results, indent=2, default=str))
    log(f"comparison written -> {EXP_DIR / 'comparison_result.json'}")
    for k, v in results.items():
        log(f"  {k}: score={v.get('score')} w/d/l={v.get('wins')}/{v.get('draws')}/{v.get('losses')} "
            f"(games={v.get('games')})")


def main() -> int:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    log(f"catheat A/B retrain experiment starting. base={BASE_WEIGHTS}")
    n_total = featurize()
    train_pool, chunks, val_idx = build_splits(n_total)
    b_final = run_experiment_b(train_pool)
    a_final = run_experiment_a(chunks)
    compare(a_final, b_final)
    log("experiment complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
