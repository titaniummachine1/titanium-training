"""Guards for lean HalfPW NNUE fine-tuning alongside overnight game generation.

OVERNIGHT STABLE SELF-IMPROVEMENT LOOP
──────────────────────────────────────
  Search/BFS  = fixed truth (do not retrain online)
  NNUE        = eval prior (slow drift OK)
  Promote if: (eval_drift > 2cp OR move_delta > 5%) AND Elo drop < 12
  Monitor: plateau_probe.py + nnue_train.log (Elo is secondary)

Architecture note (2026-06): engine + trainer share per-player field planes
(goal_inv, pawn_fwd, corridor_delta, path_cross, choke, contested).
Zero-init new planes; fine-tune only. Do not increase H or blob overnight.

Artifact budget (minimax / lean):
  - Soft warn: 500 MB under training/checkpoints + snapshots
  - Hard stop: 1 GB — refuse to train, prune old ckpts first
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "training" / "data"
CKPT_DIR = ROOT / "training" / "checkpoints"
SNAP_DIR = CKPT_DIR / "snapshots"
ELO_HISTORY = DATA_DIR / "nnue_elo_history.json"
GUARD_STATE = DATA_DIR / "nnue_guard_state.json"
NNUE_LOG = DATA_DIR / "nnue_train.log"

SOFT_CAP_BYTES = 500 * 1024 * 1024
HARD_CAP_BYTES = 1024 * 1024 * 1024

# HalfPW inference blob (leaf eval) — 11 field planes, ~552 KB weights.
NET_WEIGHT_F64S = 16 + 32 + 32 + 9 * 128 * 32 + 81 * 32 * 2 + 81 * 32 * 11
HALFPW_WEIGHT_BYTES = NET_WEIGHT_F64S * 8
HALFPW_L3_BUDGET_BYTES = 576 * 1024  # target: entire net + accum fits typical L3
MAIN_NET_SOFT_CAP_BYTES = SOFT_CAP_BYTES  # trainer checkpoints, not leaf weights

# titanium-v15@5s vs ace-v13-ti-pure@5s — pre-retrain ~48% (README baseline)
SELF_PAIR_ENGINE_A = "titanium-v15"
SELF_PAIR_ENGINE_B = "ace-v13-ti-pure"
SELF_PAIR_TC = "5s"
MAX_PRETRAIN_WIN_RATE = 0.58  # warn only for micro-train; batch mode uses stricter rule
MAX_PRETRAIN_WIN_RATE_BATCH = 0.70  # block full-DB epoch only when clearly dominating
MIN_GAMES_SELF_CHECK = 8
MIN_GAMES_BATCH_BLOCK = 24

CATCH_UP_MAX_GAMES = 8  # CLI --catch-up only; pool mode never blocks on backlog

# Copy trained weights into engine + rebuild so next titanium spawn picks up new eval.
DEPLOY_EVERY_GAMES = int(os.environ.get("NNUE_DEPLOY_EVERY", "32"))
MIN_DEPLOY_INTERVAL_SEC = float(os.environ.get("NNUE_DEPLOY_INTERVAL_SEC", "1800"))
ENGINE_DIR = ROOT / "engine"
TITANIUM_BIN = ENGINE_DIR / "target" / "release" / "titanium.exe"


def pool_quiet() -> bool:
    return os.environ.get("NNUE_POOL_QUIET") == "1"


def nnue_log(msg: str, *, force_stdout: bool = False) -> None:
    """Append to nnue_train.log; stdout only when not in overnight pool mode."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"{ts} {msg}\n"
    with open(NNUE_LOG, "a", encoding="utf-8") as f:
        f.write(line)
    if force_stdout or not pool_quiet():
        print(msg, flush=True)


def nnue_train_log_path() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return NNUE_LOG

ELO_DROP_SNAPSHOT = 12  # centipawn-scale rating points from manifest ladder
ELO_HISTORY_LEN = 64


def dir_size_bytes(path: Path) -> int:
    path = Path(path)
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def artifact_usage() -> dict:
    ckpt = dir_size_bytes(CKPT_DIR)
    db = dir_size_bytes(DATA_DIR / "all_games.db")
    return {
        "checkpoints_bytes": ckpt,
        "db_bytes": db,
        "total_bytes": ckpt + db,
    }


def prune_snapshots(snap_dir: Path | None = None, keep: int = 8) -> int:
    """Drop old Elo-drop snapshot dirs; keep newest `keep`."""
    snap_dir = Path(snap_dir or SNAP_DIR)
    if not snap_dir.exists():
        return 0
    dirs = sorted(
        (p for p in snap_dir.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
    )
    freed = 0
    for p in dirs[:-keep]:
        for f in p.rglob("*"):
            if f.is_file():
                freed += f.stat().st_size
        shutil.rmtree(p, ignore_errors=True)
    return freed


def prune_checkpoints(out_dir: Path | None = None, keep_step: int = 2) -> int:
    """Delete old ckpt_step*.pt; keep best.pt, net_weights_best.bin, latest N step ckpts."""
    out_dir = Path(out_dir or CKPT_DIR)
    if not out_dir.exists():
        return 0
    freed = 0
    steps = sorted(out_dir.glob("ckpt_step*.pt"), key=lambda p: p.stat().st_mtime)
    for p in steps[:-keep_step]:
        freed += p.stat().st_size
        p.unlink(missing_ok=True)
    epochs = sorted(out_dir.glob("ckpt_epoch*.pt"), key=lambda p: p.stat().st_mtime)
    for p in epochs[:-1]:
        freed += p.stat().st_size
        p.unlink(missing_ok=True)
    return freed


def enforce_artifact_cap(out_dir: Path | None = None) -> tuple[bool, str]:
    """Prune then verify under HARD_CAP. Returns (ok, message)."""
    out_dir = Path(out_dir or CKPT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    SNAP_DIR.mkdir(parents=True, exist_ok=True)

    usage = artifact_usage()
    if usage["checkpoints_bytes"] > SOFT_CAP_BYTES:
        freed = prune_checkpoints(out_dir)
        usage = artifact_usage()
        msg = f"pruned {freed} bytes from checkpoints"
        if usage["checkpoints_bytes"] > SOFT_CAP_BYTES:
            snap_freed = prune_snapshots()
            usage = artifact_usage()
            msg += f"; pruned {snap_freed} bytes from snapshots"
    else:
        msg = "artifact cap ok"

    if usage["checkpoints_bytes"] > HARD_CAP_BYTES:
        return False, f"HARD_CAP exceeded ({usage['checkpoints_bytes'] / 1e6:.0f} MB > 1 GB). {msg}"
    if usage["checkpoints_bytes"] > SOFT_CAP_BYTES:
        return True, f"WARN soft cap ({usage['checkpoints_bytes'] / 1e6:.0f} MB). {msg}"
    return True, msg


def _load_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def current_v15_rating(manifest: dict | None = None) -> int | None:
    from manifest import CURRENT_ENGINE, entity_label, load_manifest

    manifest = manifest or load_manifest()
    ent = entity_label(CURRENT_ENGINE, "5s")
    info = manifest.get("global_ratings", {}).get(ent)
    if not info:
        return None
    return int(info.get("rating", 0))


def record_elo_sample(manifest: dict | None = None) -> int | None:
    """Append ladder rating sample; return rating if recorded."""
    from manifest import load_manifest

    manifest = manifest or load_manifest()
    rating = current_v15_rating(manifest)
    if rating is None:
        return None

    hist = _load_json(ELO_HISTORY, {"samples": []})
    hist["samples"].append({
        "ts": datetime.now(timezone.utc).isoformat(),
        "rating": rating,
    })
    hist["samples"] = hist["samples"][-ELO_HISTORY_LEN:]
    _save_json(ELO_HISTORY, hist)
    return rating


def elo_drop_detected(threshold: int = ELO_DROP_SNAPSHOT) -> tuple[bool, int]:
    """True if latest rating dropped >= threshold from recent peak."""
    hist = _load_json(ELO_HISTORY, {"samples": []})
    samples = hist.get("samples", [])
    if len(samples) < 2:
        return False, 0
    ratings = [s["rating"] for s in samples if "rating" in s]
    if len(ratings) < 2:
        return False, 0
    peak = max(ratings[:-1])
    latest = ratings[-1]
    drop = peak - latest
    return drop >= threshold, drop


def snapshot_weights(reason: str, weights_src: Path | None = None) -> Path:
    """Copy net_weights + best.pt into timestamped snapshot dir (turning-point archive)."""
    from manifest import load_manifest

    weights_src = Path(weights_src or ROOT / "engine" / "src" / "acev13" / "net_weights.bin")
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_reason = "".join(c if c.isalnum() or c in "-_" else "_" for c in reason)[:48]
    dest = SNAP_DIR / f"{ts}_{safe_reason}"
    dest.mkdir(parents=True, exist_ok=True)

    if weights_src.exists():
        shutil.copy2(weights_src, dest / "net_weights.bin")
    best_pt = CKPT_DIR / "best.pt"
    if best_pt.exists():
        shutil.copy2(best_pt, dest / "best.pt")
    best_bin = CKPT_DIR / "net_weights_best.bin"
    if best_bin.exists():
        shutil.copy2(best_bin, dest / "net_weights_best.bin")

    manifest = load_manifest()
    meta = {
        "reason": reason,
        "ts": ts,
        "v15_rating": current_v15_rating(manifest),
        "elo_history": _load_json(ELO_HISTORY, {}),
    }
    (dest / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    nnue_log(f"NNUE snapshot -> {dest}")
    return dest


def self_matchup_win_rate(manifest: dict | None = None) -> tuple[float, int] | None:
    """Win rate of CURRENT_ENGINE vs ace-v13-ti-pure @ 5s from manifest matchups."""
    from manifest import load_manifest, matchup_key

    manifest = manifest or load_manifest()
    key = matchup_key(SELF_PAIR_ENGINE_A, SELF_PAIR_ENGINE_B, SELF_PAIR_TC, SELF_PAIR_TC)
    m = manifest.get("matchups", {}).get(key)
    if not m:
        return None
    aw, bw = m.get("a_wins", 0), m.get("b_wins", 0)
    n = aw + bw
    if n < MIN_GAMES_SELF_CHECK:
        return None
    return aw / n, n


def pretrain_sanity_ok(manifest: dict | None = None, *, batch: bool = False) -> tuple[bool, str]:
    """
    Artifact cap always. Win-rate gate only for full-DB batch epochs (not per-game micro).
    Games still accumulate in DB regardless.
    """
    from manifest import load_manifest
    from engine_identity import assert_engine_ready

    manifest = manifest or load_manifest()
    try:
        stamp = assert_engine_ready(write_if_missing=True, parity=True)
    except Exception as e:
        return False, f"engine validation failed: {e}"

    cap_ok, cap_msg = enforce_artifact_cap()
    if not cap_ok:
        return False, cap_msg

    if batch:
        wr = self_matchup_win_rate(manifest)
        if wr is not None:
            rate, n = wr
            if n >= MIN_GAMES_BATCH_BLOCK and rate > MAX_PRETRAIN_WIN_RATE_BATCH:
                return False, (
                    f"v15 vs ti-pure {rate:.1%} ({n}g) > {MAX_PRETRAIN_WIN_RATE_BATCH:.0%} "
                    "— skip batch train"
                )

    return True, f"{cap_msg}; engine {stamp['sha256'][:12]}"


def micro_train_warning(manifest: dict | None = None) -> str | None:
    """Optional one-line hint when ahead of baseline but micro-train still runs."""
    wr = self_matchup_win_rate(manifest)
    if wr is None:
        return None
    rate, n = wr
    if n >= MIN_GAMES_SELF_CHECK and rate > MAX_PRETRAIN_WIN_RATE:
        return f"note: v15 vs ti-pure {rate:.1%} ({n}g) — micro-train continues, deploy only if Elo holds"
    return None


def mark_games_processed_through(game_id: int) -> None:
    """Advance queue cursor without training (e.g. skip stale catch-up backlog)."""
    state = load_guard_state()
    state["last_trained_game_id"] = max(int(game_id), state.get("last_trained_game_id", 0))
    save_guard_state(state)


def post_train_check(manifest: dict | None = None) -> None:
    """After training: record Elo, snapshot immediately on drop."""
    from manifest import load_manifest

    manifest = manifest or load_manifest()
    record_elo_sample(manifest)
    dropped, drop = elo_drop_detected()
    if dropped:
        snapshot_weights(f"elo_drop_{drop}")


def count_db_games(db_path: Path | None = None) -> int:
    import sqlite3
    from datagen import DB_PATH

    db_path = Path(db_path or DB_PATH)
    if not db_path.exists():
        return 0
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute("SELECT COUNT(*) FROM games").fetchone()
        return int(row[0]) if row else 0
    finally:
        conn.close()


def load_guard_state() -> dict:
    return _load_json(GUARD_STATE, {
        "last_train_game_count": 0,
        "last_trained_game_id": 0,
        "last_train_ts": 0,
        "train_runs": 0,
        "games_trained": 0,
        "games_since_deploy": 0,
        "last_deploy_ts": 0,
        "deploy_runs": 0,
    })


def save_guard_state(state: dict) -> None:
    _save_json(GUARD_STATE, state)


def should_run_training_cycle(
    min_new_games: int = 32,
    min_interval_sec: float = 600,
) -> tuple[bool, str]:
    """Whether background trainer should run one epoch now."""
    state = load_guard_state()
    now = time.time()
    games = count_db_games()
    new_games = games - state.get("last_train_game_count", 0)

    if new_games < min_new_games:
        return False, f"only {new_games} new games since last train (need {min_new_games})"

    if now - state.get("last_train_ts", 0) < min_interval_sec:
        return False, f"interval guard ({min_interval_sec}s since last train)"

    ok, msg = pretrain_sanity_ok(batch=False)
    if not ok:
        return False, msg

    return True, f"{new_games} new games, {msg}"


def mark_game_trained(game_id: int) -> None:
    state = load_guard_state()
    state["last_trained_game_id"] = max(int(game_id), state.get("last_trained_game_id", 0))
    state["last_train_game_count"] = count_db_games()
    state["last_train_ts"] = time.time()
    state["train_runs"] = state.get("train_runs", 0) + 1
    state["games_trained"] = state.get("games_trained", 0) + 1
    save_guard_state(state)


def mark_training_done() -> None:
    state = load_guard_state()
    state["last_train_game_count"] = count_db_games()
    state["last_train_ts"] = time.time()
    state["train_runs"] = state.get("train_runs", 0) + 1
    save_guard_state(state)


def net_weights_size_ok(path: Path | None = None) -> bool:
    """net_weights.bin must stay tiny (minimax leaf eval, L3-friendly)."""
    path = Path(path or ROOT / "engine" / "src" / "acev13" / "net_weights.bin")
    if not path.exists():
        return False
    return path.stat().st_size == HALFPW_WEIGHT_BYTES


def assert_leaf_net_budget(extra_bytes: int = 0) -> tuple[bool, str]:
    """Reject architecture expansions that blow L3 budget at search leaves."""
    total = HALFPW_WEIGHT_BYTES + extra_bytes
    if total > HALFPW_L3_BUDGET_BYTES:
        return False, f"leaf net {total} B exceeds L3 budget {HALFPW_L3_BUDGET_BYTES} B"
    return True, f"leaf net {total} B (budget {HALFPW_L3_BUDGET_BYTES} B)"


def spawn_low_priority(
    cmd: list[str],
    cwd: Path | None = None,
    *,
    timeout_sec: float | None = None,
) -> subprocess.CompletedProcess:
    """Run training subprocess below normal priority so engine slots keep CPU."""
    import platform

    timeout_sec = timeout_sec if timeout_sec is not None else float(
        os.environ.get("NNUE_TRAIN_TIMEOUT_SEC", "600")
    )
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    kwargs: dict = {"cwd": str(cwd or ROOT), "env": env, "timeout": timeout_sec}
    if pool_quiet():
        log_path = nnue_train_log_path()
        log_f = open(log_path, "a", encoding="utf-8")
        log_f.write(f"\n--- {' '.join(cmd)} ---\n")
        log_f.flush()
        kwargs["stdout"] = log_f
        kwargs["stderr"] = subprocess.STDOUT
    else:
        kwargs["capture_output"] = False

    if platform.system() == "Windows":
        kwargs["creationflags"] = subprocess.BELOW_NORMAL_PRIORITY_CLASS  # type: ignore[attr-defined]
    else:
        def _nice():
            import os as _os
            _os.nice(15)
        kwargs["preexec_fn"] = _nice

    try:
        return subprocess.run(cmd, **kwargs)
    except subprocess.TimeoutExpired as e:
        nnue_log(f"train subprocess timeout after {timeout_sec:.0f}s: {' '.join(cmd)}")
        if e.process is not None:
            try:
                e.process.kill()
            except Exception:
                pass
        return subprocess.CompletedProcess(cmd, returncode=124)
    finally:
        if pool_quiet() and "stdout" in kwargs and hasattr(kwargs["stdout"], "close"):
            kwargs["stdout"].close()


def deploy_best_weights_to_engine() -> Path | None:
    """Copy net_weights_best.bin -> engine/src/acev13/net_weights.bin if present."""
    src = CKPT_DIR / "net_weights_best.bin"
    dest = ROOT / "engine" / "src" / "acev13" / "net_weights.bin"
    if not src.exists():
        return None
    shutil.copy2(src, dest)
    return dest


def rebuild_titanium_release() -> tuple[bool, str]:
    """Rebuild titanium.exe so include_bytes! embeds the deployed net_weights.bin."""
    import platform

    env = os.environ.copy()
    env.setdefault("RUSTFLAGS", "-C target-cpu=native")
    cmd = ["cargo", "build", "--release", "-p", "titanium"]
    kwargs: dict = {"cwd": str(ENGINE_DIR), "env": env}
    if pool_quiet():
        log_path = nnue_train_log_path()
        log_f = open(log_path, "a", encoding="utf-8")
        log_f.write(f"\n--- {' '.join(cmd)} (deploy rebuild) ---\n")
        log_f.flush()
        kwargs["stdout"] = log_f
        kwargs["stderr"] = subprocess.STDOUT
    else:
        kwargs["capture_output"] = True
        kwargs["text"] = True

    if platform.system() == "Windows":
        kwargs["creationflags"] = subprocess.BELOW_NORMAL_PRIORITY_CLASS  # type: ignore[attr-defined]
    else:
        def _nice():
            import os as _os
            _os.nice(10)
        kwargs["preexec_fn"] = _nice

    try:
        r = subprocess.run(cmd, **kwargs)
    finally:
        if pool_quiet() and "stdout" in kwargs and hasattr(kwargs["stdout"], "close"):
            kwargs["stdout"].close()

    if r.returncode != 0:
        return False, f"cargo build failed (exit {r.returncode})"
    if not TITANIUM_BIN.exists():
        return False, f"missing binary after build: {TITANIUM_BIN}"
    return True, f"rebuilt {TITANIUM_BIN.name}"


def maybe_deploy_after_train(*, force: bool = False) -> tuple[bool, str]:
    """Stable overnight: promote only if drift/move gate passes + no Elo crash."""
    state = load_guard_state()
    state["games_since_deploy"] = state.get("games_since_deploy", 0) + 1
    save_guard_state(state)

    now = time.time()
    if not force and DEPLOY_EVERY_GAMES <= 0:
        return False, "auto-deploy disabled (NNUE_DEPLOY_EVERY=0)"
    if not force:
        if state["games_since_deploy"] < DEPLOY_EVERY_GAMES:
            return False, f"deploy in {DEPLOY_EVERY_GAMES - state['games_since_deploy']} train(s)"
        if now - state.get("last_deploy_ts", 0) < MIN_DEPLOY_INTERVAL_SEC:
            return False, "deploy interval guard"

    dropped, drop = elo_drop_detected()
    if dropped and not force:
        return False, f"deploy skipped: Elo drop {drop} (snapshot on disk)"

    from plateau_probe import evaluate_promotion_gate

    promote, gate_reason, already_staged = evaluate_promotion_gate(force=force)
    if not promote:
        nnue_log(f"deploy held: {gate_reason}")
        return False, gate_reason

    if not already_staged:
        dest = deploy_best_weights_to_engine()
        if dest is None:
            return False, "no net_weights_best.bin to deploy"
        if not net_weights_size_ok(dest):
            return False, f"deployed blob size {dest.stat().st_size} != expected {HALFPW_WEIGHT_BYTES}"
        ok, rebuild_msg = rebuild_titanium_release()
        if not ok:
            return False, rebuild_msg
    else:
        rebuild_msg = "staged by move probe"

    from engine_identity import assert_engine_ready

    assert_engine_ready(write_if_missing=True, parity=True)

    state = load_guard_state()
    state["games_since_deploy"] = 0
    state["last_deploy_ts"] = now
    state["deploy_runs"] = state.get("deploy_runs", 0) + 1
    save_guard_state(state)
    msg = f"promoted ({gate_reason}) + {rebuild_msg}"
    nnue_log(f"NNUE {msg}")
    return True, msg
