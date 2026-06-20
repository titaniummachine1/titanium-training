#!/usr/bin/env python3
"""Per-game or batch HalfPW training — low CPU priority, guarded artifacts.

After each finished game the overnight pool enqueues a db_id here. One background
worker drains the queue so engine slots never wait on training.

Pool mode (NNUE_POOL_QUIET=1): all output -> training/data/nnue_train.log only.

    python training/run_nnue_cycle.py --game-id 123
    python training/run_nnue_cycle.py --catch-up
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

from tools.datagen.datagen import DB_PATH, game_source_tag, max_game_id, untrained_game_ids  # noqa: E402
from tools.maintenance.manifest import load_manifest  # noqa: E402
from titanium_training.training.guards import (  # noqa: E402
    CATCH_UP_MAX_GAMES,
    CKPT_DIR,
    DEPLOY_EVERY_GAMES,
    enforce_artifact_cap,
    mark_game_trained,
    mark_games_processed_through,
    maybe_deploy_after_train,
    micro_train_warning,
    nnue_log,
    post_train_check,
    pretrain_sanity_ok,
    prune_checkpoints,
    record_elo_sample,
    should_run_training_cycle,
    snapshot_weights,
    spawn_low_priority,
)

_warned_micro = False


def _log(msg: str) -> None:
    nnue_log(msg)


def sync_backlog_cursor() -> int:
    """Mark existing DB rows processed without training — pool UI stays clean."""
    from titanium_training.training.guards import load_guard_state

    after = load_guard_state().get("last_trained_game_id", 0)
    pending = untrained_game_ids(DB_PATH, after)
    if not pending:
        return 0
    mark_games_processed_through(pending[-1])
    _log(
        f"backlog: marked {len(pending)} game(s) processed "
        f"(ids {pending[0]}–{pending[-1]}); micro-train on new games only"
    )
    return len(pending)


def _train_cmd(*, game_ids: list[int] | None, micro: bool) -> list[str]:
    cmd = [
        sys.executable,
        "-u",
        str(ROOT / "training" / "train.py"),
        "--data", str(DB_PATH),
        "--out-dir", str(CKPT_DIR),
        "--resume",
        "--cpu",
        "--epochs", "1",
    ]
    if game_ids:
        cmd += ["--game-ids", ",".join(str(i) for i in game_ids)]
    if micro:
        cmd += ["--micro", "--batch", "64"]
    return cmd


def startup_train_catch_up(*, max_games: int | None = None) -> int:
    """On pool start: train pending games (never silently skip the backlog)."""
    import os
    from titanium_training.training.guards import load_guard_state

    if os.environ.get("NNUE_SKIP_BACKLOG") == "1":
        return sync_backlog_cursor()

    cap = max_games if max_games is not None else CATCH_UP_MAX_GAMES
    after = load_guard_state().get("last_trained_game_id", 0)
    pending = untrained_game_ids(DB_PATH, after)
    if not pending:
        return 0

    to_train = pending if len(pending) <= cap else pending[-cap:]
    _log(f"startup: micro-train {len(to_train)} pending game(s) ids {to_train[0]}-{to_train[-1]}")
    rc = 0
    for gid in to_train:
        r = run_on_game(gid)
        if r != 0:
            _log(f"startup: stopped at game {gid} (rc={r}); will retry on next restart")
            rc = r
            break
    return rc


def run_on_game(
    game_id: int,
    *,
    dry_run: bool = False,
    snapshot_every: int = 50,
) -> int:
    """Micro-train on one DB game row (non-blocking for match workers)."""
    global _warned_micro

    ok, pre_msg = pretrain_sanity_ok(batch=False)
    if not ok:
        _log(f"game {game_id} blocked: {pre_msg}; left pending")
        return 1

    cap_ok, cap_msg = enforce_artifact_cap()
    if not cap_ok:
        _log(f"game {game_id} blocked: {cap_msg}")
        return 1

    src = game_source_tag(game_id)
    _log(f"game {game_id} source={src or '?'}")

    if not _warned_micro:
        hint = micro_train_warning(load_manifest())
        if hint:
            _log(hint)
            _warned_micro = True

    from titanium_training.training.guards import load_guard_state
    state_runs = load_guard_state().get("games_trained", 0)

    if state_runs == 0 or game_id % snapshot_every == 0:
        snapshot_weights(f"pre_game_{game_id}")

    _log(f"micro-train game_id={game_id} ({pre_msg})")
    record_elo_sample(load_manifest())

    if dry_run:
        _log(f"game {game_id} dry-run: training state unchanged")
        return 0

    cmd = _train_cmd(game_ids=[game_id], micro=True)
    try:
        rc = spawn_low_priority(cmd, cwd=ROOT).returncode
    except Exception as e:
        _log(f"train.py failed on game {game_id}: {e}")
        return 1
    if rc != 0:
        _log(f"train.py exited {rc} on game {game_id}")
        return rc

    prune_checkpoints(keep_step=1)
    enforce_artifact_cap()
    mark_game_trained(game_id)
    post_train_check(load_manifest())
    try:
        from titanium_training.training.plateau_probe import maybe_trainer_probe
        maybe_trainer_probe(game_id, every=8)
    except Exception as e:
        _log(f"probe trainer skip: {e}")
    deployed, deploy_msg = maybe_deploy_after_train()
    if deployed:
        _log(deploy_msg)
        try:
            from titanium_training.training.plateau_probe import record_engine_probe
            from titanium_training.training.guards import load_guard_state
            record_engine_probe(deploy_run=load_guard_state().get("deploy_runs"))
        except Exception as e:
            _log(f"probe deploy skip: {e}")
    _log(f"game {game_id} done")
    return 0


def run_catch_up(*, dry_run: bool = False, max_games: int = CATCH_UP_MAX_GAMES) -> int:
    """CLI catch-up: train recent pending games (not used during live pool)."""
    from titanium_training.training.guards import load_guard_state

    after = load_guard_state().get("last_trained_game_id", 0)
    pending = untrained_game_ids(DB_PATH, after)
    if not pending:
        return 0

    if len(pending) > max_games:
        to_skip = pending[:-max_games]
        if to_skip:
            mark_games_processed_through(to_skip[-1])
            _log(
                f"catch-up: {len(to_skip)} older game(s) marked processed "
                f"(ids {to_skip[0]}–{to_skip[-1]})"
            )
        pending = pending[-max_games:]

    _log(f"catch-up: micro-train {len(pending)} game(s) (ids {pending[0]}–{pending[-1]})")
    rc = 0
    last_done = None
    for gid in pending:
        r = run_on_game(gid, dry_run=dry_run)
        if r != 0:
            rc = r
            _log(f"catch-up: stopped at game {gid} (rc={r}); later games left pending")
            break
        last_done = gid
    if last_done is not None:
        _log(f"catch-up done through id {last_done}")
    return rc


def run_one_epoch(
    *,
    epochs: int = 1,
    batch: int = 512,
    checkpoint_steps: int = 5000,
    dry_run: bool = False,
    force: bool = False,
    min_new_games: int = 32,
) -> int:
    """Full-DB epoch (legacy batch mode). Per-game path uses run_on_game."""
    if not force:
        ok, reason = should_run_training_cycle(min_new_games=min_new_games)
        if not ok:
            _log(f"cycle skipped: {reason}")
            return 0

    cap_ok, cap_msg = enforce_artifact_cap()
    if not cap_ok:
        _log(f"cycle blocked: {cap_msg}")
        return 1

    ok, pre_msg = pretrain_sanity_ok(batch=True)
    if not ok:
        _log(f"cycle blocked: {pre_msg}")
        return 1

    _log(f"batch cycle starting: {pre_msg}")
    record_elo_sample(load_manifest())
    if dry_run:
        return 0

    snapshot_weights("pre_train_epoch")
    cmd = _train_cmd(game_ids=None, micro=False)
    cmd += ["--epochs", str(epochs), "--batch", str(batch),
            "--checkpoint-steps", str(checkpoint_steps)]
    rc = spawn_low_priority(cmd, cwd=ROOT).returncode
    if rc != 0:
        return rc

    prune_checkpoints(keep_step=1)
    enforce_artifact_cap()
    from titanium_training.training.guards import mark_training_done
    mark_training_done()
    mark_games_processed_through(max_game_id(DB_PATH))
    post_train_check(load_manifest())
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--game-id", type=int, default=None, help="Train one DB row")
    ap.add_argument("--catch-up", action="store_true", help="Train recent pending games")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--min-new-games", type=int, default=32)
    ap.add_argument("--checkpoint-steps", type=int, default=5000)
    args = ap.parse_args()

    if args.game_id is not None:
        sys.exit(run_on_game(args.game_id, dry_run=args.dry_run))
    if args.catch_up:
        sys.exit(run_catch_up(dry_run=args.dry_run))
    sys.exit(run_one_epoch(
        epochs=args.epochs,
        batch=args.batch,
        checkpoint_steps=args.checkpoint_steps,
        dry_run=args.dry_run,
        force=args.force,
        min_new_games=args.min_new_games,
    ))


if __name__ == "__main__":
    main()
