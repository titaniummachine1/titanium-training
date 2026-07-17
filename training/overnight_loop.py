#!/usr/bin/env python3
"""Continuous overnight training loop until interrupted.

Each cycle:
  1. Train 1 epoch (win=+1 / loss=-1 labels, position usage retires after 5 epochs)
  2. Deploy best weights to engine live blob
  3. Elo probe: N games current vs previous (512 until epoch 3, then 1024)
  4. Self-play data batch (70% same-net / 30% current vs previous), 4s/move default
  5. Sync new games -> teacher parquet, rebuild feature cache
  6. Rotate previous weights checkpoint

Run: python training/overnight_loop.py --from-frozen
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_TRAINING = _REPO / "training"
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

CACHE_DIR = _TRAINING / "data" / "feature_cache"
RUN_DIR = _TRAINING / "runs" / "value_oracle"
LOG_DIR = _TRAINING / "data" / "overnight_logs"
STATE_PATH = LOG_DIR / "loop_state.json"
FROZEN = _REPO / "engine" / "src" / "titanium" / "net_weights_frozen.bin"
BEST = RUN_DIR / "net_weights_best.bin"
PREVIOUS = RUN_DIR / "net_weights_previous.bin"

_stop = False


def _handle_sigint(_sig, _frame) -> None:
    global _stop
    _stop = True
    log("Interrupt received — finishing current step then exiting.")


def log(msg: str) -> None:
    print(msg, flush=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with (LOG_DIR / "loop.log").open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def run(cmd: list[str]) -> int:
    log(f"$ {' '.join(cmd)}")
    env = {**dict(__import__("os").environ), "PYTHONPATH": str(_TRAINING)}
    return subprocess.call(cmd, cwd=str(_REPO), env=env)


def load_state() -> dict:
    if STATE_PATH.is_file():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"epoch": 0, "cycles": 0}


def save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def ensure_previous_from_frozen() -> None:
    if not PREVIOUS.is_file() and FROZEN.is_file():
        shutil.copy2(FROZEN, PREVIOUS)
        log(f"Initialized previous weights from frozen -> {PREVIOUS.name}")


def rotate_previous() -> None:
    if BEST.is_file():
        shutil.copy2(BEST, PREVIOUS)
        log(f"Rotated previous weights from best -> {PREVIOUS.name}")


def games_for_epoch(epoch: int, eval_games: int, data_games_small: int, data_games_large: int) -> tuple[int, int]:
    eval_n = eval_games if epoch >= 3 else min(eval_games, 512)
    data_n = data_games_large if epoch >= 3 else data_games_small
    return eval_n, data_n


def train_one_epoch(*, from_frozen: bool, resume: bool) -> int:
    trainer = _TRAINING / "titanium_training" / "training" / "trainer.py"
    cmd = [
        sys.executable, str(trainer),
        "--cache-dir", str(CACHE_DIR),
        "--out-dir", str(RUN_DIR),
        "--epochs", "1",
        "--batch", "512",
        "--lr", "0.0005",
        "--checkpoint-steps", "999999",
        "--val-split", "0.05",
        "--patience", "0",
        "--cpu",
    ]
    if resume and not from_frozen:
        ckpts = sorted(RUN_DIR.glob("ckpt_epoch*.pt"))
        if ckpts:
            cmd.extend(["--resume", "--ckpt", str(ckpts[-1])])
    rc = run(cmd)
    if rc != 0:
        return rc
    from revert_checkpoint import export_checkpoint

    latest = sorted(RUN_DIR.glob("ckpt_epoch*.pt"))[-1]
    export_checkpoint(latest, deploy_engine=True)
    return 0


def run_selfplay(*, n_games: int, threads: int, time_sec: float, current: Path, previous: Path,
                 same_net_pct: float, stream_db: bool, tag: str) -> dict:
    from self_play_overnight import run_batch, run_batch_streaming

    if stream_db:
        stats, _ = run_batch_streaming(
            n_games=n_games,
            threads=threads,
            time_sec=time_sec,
            current=current,
            previous=previous,
            p_same_net=same_net_pct,
            seed=int(datetime.now(timezone.utc).timestamp()) % 1_000_000,
        )
    else:
        stats, _ = run_batch(
            n_games=n_games,
            threads=threads,
            time_sec=time_sec,
            current=current,
            previous=previous,
            p_same_net=same_net_pct,
            write_db=True,
        )
    report = {
        "tag": tag,
        "games": stats.games,
        "mixed_games": stats.mixed_games,
        "current_wins": stats.current_wins,
        "current_losses": stats.current_losses,
        "current_win_rate": stats.current_win_rate,
        "pseudo_elo_vs_previous": (
            round(-400 * math.log10(1 / stats.current_win_rate - 1), 1)
            if 0 < stats.current_win_rate < 1 and stats.mixed_games > 0
            else None
        ),
        "saturated": stats.saturated(min_games=max(32, n_games // 10)),
    }
    out = LOG_DIR / f"selfplay_{tag}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"Self-play [{tag}]: {json.dumps(report)}")
    return report


def sync_and_rebuild_cache() -> int:
    rc = run([sys.executable, str(_TRAINING / "sync_overnight_to_teacher.py")])
    if rc != 0:
        return rc
    usage = CACHE_DIR / "usage_counts.npy"
    if usage.is_file():
        usage.unlink()
        log("Reset usage_counts.npy after teacher append (cache size will change)")
    return run([
        sys.executable, str(_TRAINING / "build_feature_cache.py"),
        "--cache-dir", str(CACHE_DIR),
        "--force",
    ])


def ensure_feature_cache() -> int:
    from build_feature_cache import check_fingerprint
    ok, reason = check_fingerprint(CACHE_DIR)
    if ok:
        return 0
    log(f"Feature cache missing/stale ({reason}) — building from teacher dataset...")
    return run([
        sys.executable, str(_TRAINING / "build_feature_cache.py"),
        "--cache-dir", str(CACHE_DIR),
        "--force",
    ])


def main() -> int:
    global _stop
    signal.signal(signal.SIGINT, _handle_sigint)
    signal.signal(signal.SIGTERM, _handle_sigint)

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--time", type=float, default=4.0, help="Seconds per move in self-play")
    ap.add_argument("--eval-games", type=int, default=512, help="Current vs previous eval games (1024 from epoch 3)")
    ap.add_argument("--data-games", type=int, default=512, help="Self-play data games epochs 1-2")
    ap.add_argument("--data-games-large", type=int, default=1024, help="Self-play data games from epoch 3+")
    ap.add_argument("--same-net-pct", type=float, default=0.7, help="Fraction same-net games (rest = current vs previous)")
    ap.add_argument("--from-frozen", action="store_true", help="Restore live+best from frozen before loop")
    ap.add_argument("--archive-corrupted-ckpts", action="store_true")
    ap.add_argument("--max-cycles", type=int, default=0, help="Stop after N cycles (0 = forever)")
    args = ap.parse_args()

    if args.from_frozen:
        revert_cmd = [sys.executable, str(_TRAINING / "revert_to_frozen.py")]
        if args.archive_corrupted_ckpts:
            revert_cmd.append("--archive-ckpts")
        rc = run(revert_cmd)
        if rc != 0:
            return rc
        ensure_previous_from_frozen()

    state = load_state()
    resume = state["epoch"] > 0 or any(RUN_DIR.glob("ckpt_epoch*.pt"))

    rc = ensure_feature_cache()
    if rc != 0:
        return rc

    log(f"=== overnight loop start epoch={state['epoch']} resume={resume} ===")

    while not _stop:
        state["epoch"] += 1
        state["cycles"] += 1
        epoch = state["epoch"]
        save_state(state)

        eval_n, data_n = games_for_epoch(epoch, args.eval_games, args.data_games, args.data_games_large)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        log(f"\n--- cycle {state['cycles']} epoch {epoch} @ {stamp} ---")
        log(f"eval_games={eval_n} data_games={data_n} threads={args.threads} time={args.time}s")

        if not BEST.is_file():
            log("ERROR: net_weights_best.bin missing — run with --from-frozen first")
            return 1
        if not PREVIOUS.is_file():
            ensure_previous_from_frozen()

        rc = train_one_epoch(from_frozen=False, resume=resume)
        resume = True
        if rc != 0:
            log(f"TRAIN FAILED rc={rc}")
            return rc

        if _stop:
            break

        eval_report = run_selfplay(
            n_games=eval_n,
            threads=args.threads,
            time_sec=args.time,
            current=BEST,
            previous=PREVIOUS,
            same_net_pct=0.0,
            stream_db=False,
            tag=f"eval_e{epoch:04d}",
        )
        if eval_report.get("saturated"):
            log("SATURATED: current net losing to previous — stopping loop.")
            (RUN_DIR / "SATURATED.txt").write_text(json.dumps(eval_report, indent=2), encoding="utf-8")
            return 2

        if _stop:
            break

        run_selfplay(
            n_games=data_n,
            threads=args.threads,
            time_sec=args.time,
            current=BEST,
            previous=PREVIOUS,
            same_net_pct=args.same_net_pct,
            stream_db=True,
            tag=f"data_e{epoch:04d}",
        )

        if _stop:
            break

        rc = sync_and_rebuild_cache()
        if rc != 0:
            log(f"SYNC/CACHE FAILED rc={rc}")
            return rc

        rotate_previous()

        from position_usage import status as usage_status
        if CACHE_DIR.is_dir():
            log(f"Usage: {usage_status(CACHE_DIR)}")

        if args.max_cycles and state["cycles"] >= args.max_cycles:
            log(f"Reached max cycles ({args.max_cycles}).")
            break

    log("Loop stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
