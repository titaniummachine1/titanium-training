"""Auto-widen the live NNUE (Net2WiderNet) when training plateaus.

Hooked into training_coordinator.py's accept/quarantine bookkeeping. Primary
plateau signal is `consecutive_quarantines` (the coordinator's own existing
measure of "candidate stopped beating its parent in real matches") -- that's
a held-out-game signal, stronger than a raw loss trend on its own. But match
plateau alone can't tell "out of capacity" apart from "targets exhausted or
noisy", and widening into the latter just adds memorization capacity for no
strength gain. `widen_signal()` below adds a second, independent check on the
train/val loss relationship (see its docstring) before a widen is allowed to
fire, plus a cooldown so a freshly-widened net gets real training time before
it can trigger another expansion.

Once triggered, widening happens off `latest_accepted()` — i.e. always the
strongest net currently in the chain, never an in-flight/quarantined
candidate — and the widened blob is appended to the chain via the normal
`accept_checkpoint` path (function-preserving widening changes the net's
output by only a symmetry-breaking noise term, so no strength-gate match is
required before promoting it; the very next training cycle rebases from it
automatically through `resolve_latest_accepted_weights()`).
"""
from __future__ import annotations

import json
import math
import os
import statistics
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

_TRAINING = Path(__file__).resolve().parent
_REPO = _TRAINING.parent

from streaming_checkpoint_chain import (
    LOG_DIR,
    RUN_DIR,
    accept_checkpoint,
    latest_accepted,
    load_chain,
    resolve_accepted_weights,
)

# Matches engine/src/titanium/net.rs::MAX_NET_H — the widest net the engine's
# fixed-size hot-path arrays can hold without a rebuild. Hard architectural
# ceiling, not an operational target -- see WIDEN_CAP below for the actual
# current growth limit.
MAX_NET_H = 256
H_HEADER_LEN = 8

WIDEN_DIR = RUN_DIR / "net2net_widened"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_flag(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


PLATEAU_QUARANTINE_THRESHOLD = _env_int("NET2NET_PLATEAU_QUARANTINES", 6)
AUTO_WIDEN_ENABLED = _env_flag("NET2NET_AUTO_WIDEN", True)

EPOCH_REPORTS_DIR = LOG_DIR / "epoch_reports"

# 2026-07-10: joint widen gate, added on top of the raw consecutive_quarantines
# check. epsilon is a RELATIVE loss-improvement threshold (fraction of the
# oldest loss in the lookback window), not an absolute one -- loss scale
# drifts across training, so a fixed absolute delta would silently mean
# different things at different points in the run. 0.003-0.005 chosen as a
# starting range: small enough that real ongoing learning (typically several
# times that per window) isn't mistaken for a plateau, large enough that
# normal per-cycle training noise doesn't look like "still improving" and
# block a genuine plateau forever. Configurable since the right value is an
# empirical question, not something to get right on the first guess.
WIDEN_EPSILON = _env_float("NET2NET_WIDEN_EPSILON", 0.003)
# Epochs that must elapse after a widen before another one can fire -- a
# freshly-widened net has new never-trained units and will legitimately
# quarantine a few times while it adapts; without this, that adaptation noise
# alone would look like another plateau and trigger a second expansion before
# the first one has demonstrated anything.
WIDEN_COOLDOWN_EPOCHS = _env_int("NET2NET_WIDEN_COOLDOWN_EPOCHS", 6)
# How many recent per-cycle epoch reports to inspect for the loss-trend check.
WIDEN_LOOKBACK_EPOCHS = _env_int("NET2NET_WIDEN_LOOKBACK_EPOCHS", 6)
# Minimum number of successfully-parsed reports required in that window --
# below this, there isn't enough signal to claim a plateau either way.
WIDEN_MIN_VAL_OBSERVATIONS = _env_int("NET2NET_WIDEN_MIN_VAL_OBSERVATIONS", 4)

# Operational cap on auto-widening, separate from the architectural MAX_NET_H.
# Set 2026-07-10: measured h=128 vs h=96 at ~9.3% NPS cost (239K -> 217K nps,
# single position, single-threaded 10s search) -- real but modest, judged
# acceptable for one more widen step. Capped here (rather than left to run up
# to MAX_NET_H unattended) because widening is only worth it once the current
# width has actually been trained to a real plateau, not applied repeatedly
# just because one plateau triggered it. Raise this, informed by a fresh NPS
# measurement at the new width, once h=128 itself plateaus.
WIDEN_CAP = _env_int("NET2NET_WIDEN_CAP", 128)


def _read_h(bin_path: Path) -> int:
    with bin_path.open("rb") as f:
        header = f.read(H_HEADER_LEN)
    (h,) = struct.unpack("<Q", header)
    return h


def _load_recent_epoch_reports(n: int) -> list[dict[str, Any]] | None:
    """Last `n` per-cycle reports (oldest first), or None if the directory
    is missing/unreadable or any report in the window fails to parse.
    Deliberately fails closed -- a missing/corrupt report means "cannot
    confirm a plateau", not "plateaued"."""
    if not EPOCH_REPORTS_DIR.is_dir():
        return None
    try:
        files = sorted(
            EPOCH_REPORTS_DIR.glob("epoch_*.json"),
            key=lambda p: int(p.stem.split("_")[-1]),
        )
    except (ValueError, OSError):
        return None
    if not files:
        return None
    reports = []
    for f in files[-n:]:
        try:
            reports.append(json.loads(f.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            return None
    return reports


def _extract_losses(reports: list[dict[str, Any]]) -> list[tuple[float, float]] | None:
    """(train_loss_end, val_loss) per report, oldest first. None (fail
    closed) if any report is missing or has a non-finite loss field."""
    out = []
    for r in reports:
        training = r.get("training") or {}
        train_loss = training.get("train_loss_end")
        val_loss = training.get("val_loss")
        if not isinstance(train_loss, (int, float)) or not isinstance(val_loss, (int, float)):
            return None
        if not (math.isfinite(train_loss) and math.isfinite(val_loss)):
            return None
        out.append((float(train_loss), float(val_loss)))
    return out


def _relative_improvement(oldest: float, best_recent: float) -> float | None:
    if oldest == 0:
        return None
    return (oldest - best_recent) / oldest


class _ChainReadError(Exception):
    """Chain is missing, corrupt, or contains a malformed/conflicting widen
    entry -- distinct from "chain read fine, no widen entry exists". Callers
    must treat this as "cannot confirm cooldown is clear" and fail closed,
    never as "never widened"."""


def _last_widen_epoch_from_chain() -> int | None:
    """Cooldown anchor, read from the accepted chain rather than state.json.
    accept_checkpoint() writes weights + chain entry together inside a single
    call, before maybe_auto_widen() returns -- state.json is written later by
    the caller's own loop, with real logic (logging, further state mutations)
    running in between. A crash in that window would leave state.json without
    last_widen_epoch while the chain already has the widened entry; reading
    the chain directly means the cooldown survives that crash instead of
    silently resetting to "never widened".

    Returns None only when the chain parses cleanly and genuinely contains no
    widen entry. Raises _ChainReadError on anything else (missing/corrupt
    JSON, non-list epochs, non-dict entries, a widen entry with a malformed
    epoch field) -- these must NOT be treated as "no cooldown", and must NOT
    silently fall through to an older, possibly-stale widen entry either.
    Takes max() over any widen entries found rather than trusting append
    order, so it doesn't depend on the chain never being reordered.
    """
    try:
        chain = load_chain()
    except (OSError, ValueError) as exc:  # json.JSONDecodeError is a ValueError
        raise _ChainReadError(f"chain unreadable: {exc}") from exc

    epochs = chain.get("epochs")
    if epochs is None:
        return None
    if not isinstance(epochs, list):
        raise _ChainReadError(f"chain 'epochs' is not a list: {type(epochs).__name__}")

    widen_epochs: list[int] = []
    for entry in epochs:
        if not isinstance(entry, dict):
            raise _ChainReadError("chain contains a non-dict epoch entry")
        validation = entry.get("validation")
        if not isinstance(validation, dict) or validation.get("reason") != "net2net_auto_widen":
            continue
        raw_epoch = entry.get("epoch")
        try:
            widen_epochs.append(int(raw_epoch))
        except (TypeError, ValueError):
            raise _ChainReadError(f"widen entry has malformed epoch field: {raw_epoch!r}")

    return max(widen_epochs) if widen_epochs else None


def widen_signal(state: dict[str, Any], *, completed_cycles: int) -> dict[str, Any]:
    """Independent joint gate evaluated on top of the raw consecutive_quarantines
    plateau check. Distinguishes a genuine capacity/optimization plateau from
    three lookalikes that widening would NOT help (or would actively hurt):

      Case A -- train improving, val flat:      overfitting; more width just
                                                 adds memorization capacity.
      Case B -- train flat, val flat:           capacity/optimization
                                                 plateau; widening is plausible.
      Case C -- train and val both improving:   still learning; not done yet.
      Case D -- train flat, val worsening:      data/distribution problem,
                                                 not a capacity problem.

    Only Case B allows a widen -- and even then this means "widening is
    permissible", not "widening is the correct diagnosis": flat/flat is also
    consistent with poor optimization, saturated targets, or inadequate
    features, which more width does not fix either. Automation should stop at
    "permissible", the same level of confidence this gate is built to.

    "flat"/"improving"/"worsening" are judged by WIDEN_EPSILON, a *relative*
    improvement between the median of the window's first half and the median
    of its second half (median rather than min/oldest-single-point so one
    noisy observation on either end can't flip the verdict).

    Fails closed (allowed=False) on cooldown, a missing/corrupt/conflicting
    chain, missing epoch reports, missing/non-finite loss fields, or a
    degenerate (zero) baseline loss. Never raises.
    """
    try:
        last_widen_epoch = _last_widen_epoch_from_chain()
    except _ChainReadError as exc:
        return {"allowed": False, "reason": "chain_unreadable_or_conflicting", "detail": str(exc)}
    epochs_since_widen = (
        completed_cycles - last_widen_epoch if last_widen_epoch is not None else None
    )
    if epochs_since_widen is not None and epochs_since_widen < WIDEN_COOLDOWN_EPOCHS:
        return {
            "allowed": False,
            "reason": "widen_cooldown",
            "epochs_since_widen": epochs_since_widen,
            "last_widen_epoch": last_widen_epoch,
        }

    reports = _load_recent_epoch_reports(WIDEN_LOOKBACK_EPOCHS)
    if reports is None or len(reports) < WIDEN_MIN_VAL_OBSERVATIONS:
        return {
            "allowed": False,
            "reason": "insufficient_epoch_reports",
            "n_reports": 0 if reports is None else len(reports),
        }

    losses = _extract_losses(reports)
    if losses is None:
        return {"allowed": False, "reason": "malformed_epoch_report_losses"}

    report_ids = [r.get("epoch") for r in reports]
    mid = len(losses) // 2
    first_half, second_half = losses[:mid], losses[mid:]
    median_train_first = statistics.median(t for t, _ in first_half)
    median_val_first = statistics.median(v for _, v in first_half)
    median_train_second = statistics.median(t for t, _ in second_half)
    median_val_second = statistics.median(v for _, v in second_half)

    train_rel = _relative_improvement(median_train_first, median_train_second)
    val_rel = _relative_improvement(median_val_first, median_val_second)
    if train_rel is None or val_rel is None:
        return {"allowed": False, "reason": "degenerate_loss_baseline"}

    train_improving = train_rel >= WIDEN_EPSILON
    val_improving = val_rel >= WIDEN_EPSILON
    val_worsening = val_rel <= -WIDEN_EPSILON

    if train_improving and not val_improving:
        case, allowed = "A_overfitting", False
    elif not train_improving and not val_improving and not val_worsening:
        case, allowed = "B_plateau", True
    elif train_improving and val_improving:
        case, allowed = "C_still_learning", False
    elif not train_improving and val_worsening:
        case, allowed = "D_distribution_problem", False
    else:
        case, allowed = "ambiguous", False

    return {
        "allowed": allowed,
        "reason": case,
        "report_ids": report_ids,
        "median_train_loss_first_half": median_train_first,
        "median_train_loss_second_half": median_train_second,
        "median_val_loss_first_half": median_val_first,
        "median_val_loss_second_half": median_val_second,
        "train_rel_improvement": train_rel,
        "val_rel_improvement": val_rel,
        "epsilon": WIDEN_EPSILON,
        "n_reports": len(reports),
    }


def maybe_auto_widen(state: dict[str, Any], *, completed_cycles: int) -> dict[str, Any] | None:
    """Widen the accepted net in place if training has plateaued.

    Returns a result dict (for logging/state bookkeeping) if it widened,
    else None. Never raises — a failed widen attempt just logs and leaves
    state alone so the coordinator's normal loop continues unaffected.
    """
    if not AUTO_WIDEN_ENABLED:
        return None
    if state.get("net2net_maxed"):
        return None
    consecutive_quarantines = int(state.get("consecutive_quarantines", 0))
    if consecutive_quarantines < PLATEAU_QUARANTINE_THRESHOLD:
        return None

    signal = widen_signal(state, completed_cycles=completed_cycles)
    if not signal["allowed"]:
        return {"widened": False, "reason": f"widen_signal:{signal['reason']}", "signal": signal}

    last = latest_accepted()
    if last is None:
        return None
    try:
        accepted_bin = resolve_accepted_weights(last)
    except FileNotFoundError:
        return None

    old_h = _read_h(accepted_bin)
    if old_h >= WIDEN_CAP:
        state["net2net_maxed"] = True
        return {"widened": False, "reason": "already at WIDEN_CAP", "old_h": old_h, "widen_cap": WIDEN_CAP}

    # 2026-07-10: was old_h * 2 (doubling); switched to +50% per step so each
    # widen is a smaller, cheaper-to-verify jump instead of a size-quadrupling
    # leap across two widens -- still gated by WIDEN_CAP/MAX_NET_H below.
    new_h = min(round(old_h * 1.5), WIDEN_CAP, MAX_NET_H)
    next_epoch = completed_cycles + 1

    WIDEN_DIR.mkdir(parents=True, exist_ok=True)
    out_bin = WIDEN_DIR / f"epoch_{next_epoch:04d}_h{old_h}to{new_h}.bin"

    cmd = [
        sys.executable,
        str(_TRAINING / "tools" / "net2net_widen.py"),
        "--in-bin", str(accepted_bin),
        "--out-bin", str(out_bin),
        "--new-h", str(new_h),
        "--seed", str(next_epoch),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_TRAINING)
    proc = subprocess.run(cmd, cwd=str(_REPO), capture_output=True, text=True, env=env)
    if proc.returncode != 0 or not out_bin.is_file():
        return {
            "widened": False,
            "reason": "net2net_widen.py failed",
            "returncode": proc.returncode,
            "stderr_tail": proc.stderr[-2000:],
        }

    # No "passed": True here -- this is not a strength-gate result, it's a
    # function-preserving transform (see module docstring: widening changes
    # output by only a symmetry-breaking noise term). Claiming a pass with no
    # match run is exactly the fabricated-validation pattern that caused the
    # epoch-37 incident; the honest record is the widen_signal that permitted
    # this and the fact that no gate applies here by construction.
    accept_checkpoint(
        weights_path=out_bin,
        epoch=next_epoch,
        validation={
            "gate": "not_applicable_function_preserving_widen",
            "reason": "net2net_auto_widen",
            "widen_signal": signal,
            "source_epoch": last.get("epoch"),
            "old_h": old_h,
            "new_h": new_h,
            "trigger_consecutive_quarantines": consecutive_quarantines,
        },
    )
    state["last_widen_epoch"] = next_epoch
    return {"widened": True, "old_h": old_h, "new_h": new_h, "epoch": next_epoch, "source_epoch": last.get("epoch")}
