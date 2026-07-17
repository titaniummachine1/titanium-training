#!/usr/bin/env python3
"""One-off experiment: optimizer + LR population trained from the current
accepted weights, then a local Swiss tournament (candidates + historical
accepted epochs as references) to see which one is actually strongest -- not
just which one has the lowest training loss.

Population (see CANDIDATES_SPEC): 3 AdamW candidates across an LR ladder,
plus 2 Muon-hybrid candidates (Muon on exactly-2D hidden matrices, AdamW on
biases/norms/3D embeddings) at LRs chosen by calibrate_muon_lr() below --
Muon's update-norm geometry differs from AdamW's, so "same numeric LR" is
not a comparable step size and the two Muon LRs are picked to (a) roughly
match AdamW-control's update/weight-norm ratio and (b) a bolder ~2x step,
not guessed blind.

This is a manual, one-time run. It does NOT wire into training_coordinator.py
and does NOT auto-promote a winner -- it reports full standings (aggregate +
head-to-head vs every entrant, including each historical reference
individually) and leaves the promotion decision to a human.

Tracks immediate fitness only (this one epoch's 300-game tournament). Lineage
fitness (does the winning method/optimizer keep winning over several
subsequent generations, respawning ALL variants fresh from each new accepted
parent rather than permanently locking in one optimizer) is not automated
here -- re-run this same script pointed at the new winner to continue the
lineage manually.

Usage:
    python training/tools/lr_sweep_tournament.py

Before running: stop local_game_pool and pause training_coordinator
(TRAINING_PAUSED.json), so nothing else competes for the trainer lock or CPU.
"""
from __future__ import annotations

import json
import random
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

_TRAINING = Path(__file__).resolve().parents[1]
_REPO = _TRAINING.parent
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

from db_import import LABELS_DB_PATH
from engine_session import EngineSession
from streaming_checkpoint_chain import (
    accepted_snapshot_path,
    load_chain,
    resolve_latest_accepted_weights,
)

RUN_DIR = _TRAINING / "runs" / "v16"
EXPERIMENT_DIR = RUN_DIR / "evo_experiment"
CALIBRATION_DIR = EXPERIMENT_DIR / "calibration"
REPORT_PATH = EXPERIMENT_DIR / "lr_sweep_report.json"
TRAINER = _TRAINING / "titanium_training" / "training" / "trainer.py"

WEIGHT_DECAY = "0.00001"
STREAM_MAX_POSITIONS = "16384"
CALIBRATION_POSITIONS = "1024"  # small/fast -- just enough for one real step's update-ratio
FIXED_SEED = "20260704"  # same seed for every candidate (narrows but doesn't
# eliminate data-window drift, since sample_epoch_keys depends on live
# training_visits state, not just the seed -- a true independent-DB clone
# per candidate would mean copying a 24GB labels.db five times, not
# practical for a one-off experiment).

TOTAL_GAMES = 300
TIME_SEC = 1.0
GAME_MAX_PLY = 128
CONCURRENCY = 4

MUON_TRIAL_LRS = [0.005, 0.01, 0.02, 0.05, 0.1]
ADAMW_CONTROL_LR = 0.001


def _run_trainer(*, out_dir: Path, weights: Path, lr: float, optimizer: str,
                  stream_max_positions: str, aux_lr: float | None = None,
                  commit_usage: bool = False, timeout: float = 1800) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(TRAINER),
        "--labels-db", str(LABELS_DB_PATH),
        "--out-dir", str(out_dir),
        "--weights", str(weights),
        "--epochs", "1",
        "--batch", "512",
        "--lr", str(lr),
        "--weight-decay", WEIGHT_DECAY,
        "--optimizer", optimizer,
        "--stream-max-positions", stream_max_positions,
        "--stream-featurize-chunk", "4096",
        "--stream-retired-replay-fraction", "0.05",
        "--stream-old-refresh-fraction", "0.05",
        "--val-split", "0.05",
        "--checkpoint-steps", "999999",
        "--patience", "0",
        "--cpu",
        "--defer-usage-commit",
        "--seed", FIXED_SEED,
        "--log-every", "200",
        "--log-interval-sec", "30",
    ]
    if not commit_usage:
        # Nothing changes training_visits/rowid-order selection state between
        # runs, so sample_epoch_keys() returns the IDENTICAL 16384 keys every
        # time -- all candidates train on the same data, only the method
        # differs. Only the one designated "real" run (commit_usage=True)
        # actually advances position-usage state for the real epoch.
        cmd += ["--no-usage-commit"]
    if aux_lr is not None:
        cmd += ["--aux-lr", str(aux_lr)]
    started = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(_REPO), capture_output=True, text=True, timeout=timeout)
    elapsed = time.perf_counter() - started
    diag_path = out_dir / "epoch_diagnostics_0001.json"
    diag = json.loads(diag_path.read_text(encoding="utf-8")) if diag_path.is_file() else {}
    weights_bin = out_dir / "net_weights_best.bin"
    ok = proc.returncode == 0 and weights_bin.is_file()
    return {
        "ok": ok,
        "returncode": proc.returncode,
        "elapsed_sec": round(elapsed, 1),
        "weights": str(weights_bin) if ok else None,
        "diag": diag,
        "stderr_tail": proc.stderr[-1500:] if not ok else "",
    }


def calibrate_muon_lr(init_weights: Path) -> list[float]:
    """Real (not guessed) Muon LR selection: run fast mini-trainings (small
    --stream-max-positions, same code path as the real candidates) at an
    AdamW control LR and a grid of Muon trial LRs, read back each run's own
    update_over_param_norm (already computed by trainer.py's diagnostics),
    and pick the two Muon LRs whose ratios are (a) closest to the AdamW
    control's ratio -- a comparable step size, not "same number, different
    geometry" -- and (b) roughly 2x that, a deliberately bolder step."""
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    print("[calibrate] AdamW control...", flush=True)
    control = _run_trainer(
        out_dir=CALIBRATION_DIR / "control_adamw",
        weights=init_weights,
        lr=ADAMW_CONTROL_LR,
        optimizer="adamw",
        stream_max_positions=CALIBRATION_POSITIONS,
        timeout=300,
    )
    control_ratio = control["diag"].get("update_over_param_norm")
    print(f"  control update/param ratio = {control_ratio}", flush=True)
    if control_ratio is None:
        print("  WARNING: control calibration failed, falling back to fixed Muon LRs [0.01, 0.02]", flush=True)
        return [0.01, 0.02]

    trial_results = []
    for lr in MUON_TRIAL_LRS:
        print(f"[calibrate] muon trial lr={lr}...", flush=True)
        r = _run_trainer(
            out_dir=CALIBRATION_DIR / f"trial_muon_{lr}",
            weights=init_weights,
            lr=lr,
            optimizer="muon",
            stream_max_positions=CALIBRATION_POSITIONS,
            aux_lr=ADAMW_CONTROL_LR,
            timeout=300,
        )
        ratio = r["diag"].get("update_over_param_norm")
        print(f"  lr={lr} update/param ratio = {ratio} ok={r['ok']}", flush=True)
        if r["ok"] and ratio is not None:
            trial_results.append((lr, ratio))

    if not trial_results:
        print("  WARNING: all muon trials failed, falling back to fixed LRs [0.01, 0.02]", flush=True)
        return [0.01, 0.02]

    closest = min(trial_results, key=lambda t: abs(t[1] - control_ratio))
    bolder_target = control_ratio * 2.0
    bolder = min(trial_results, key=lambda t: abs(t[1] - bolder_target))
    if bolder[0] == closest[0] and len(trial_results) > 1:
        remaining = [t for t in trial_results if t[0] != closest[0]]
        bolder = max(remaining, key=lambda t: t[1])
    print(f"[calibrate] chosen: comparable-step muon_lr={closest[0]} (ratio={closest[1]:.4g}), "
          f"bolder muon_lr={bolder[0]} (ratio={bolder[1]:.4g}), control ratio={control_ratio:.4g}", flush=True)
    return [closest[0], bolder[0]]


def train_candidates(init_weights: Path, muon_lrs: list[float]) -> list[dict[str, Any]]:
    specs = [
        {"name": "adamw_0.001_control", "optimizer": "adamw", "lr": ADAMW_CONTROL_LR, "aux_lr": None},
        {"name": "adamw_0.002", "optimizer": "adamw", "lr": 0.002, "aux_lr": None},
        {"name": "adamw_0.005", "optimizer": "adamw", "lr": 0.005, "aux_lr": None},
        {"name": f"muon_{muon_lrs[0]}_comparable", "optimizer": "muon", "lr": muon_lrs[0], "aux_lr": ADAMW_CONTROL_LR},
        {"name": f"muon_{muon_lrs[1]}_bolder", "optimizer": "muon", "lr": muon_lrs[1], "aux_lr": ADAMW_CONTROL_LR},
    ]
    candidates = []
    for i, spec in enumerate(specs):
        is_last = i == len(specs) - 1
        out_dir = EXPERIMENT_DIR / spec["name"]
        print(f"[train] {spec['name']} (optimizer={spec['optimizer']} lr={spec['lr']} aux_lr={spec['aux_lr']}) "
              f"commit_usage={is_last} -> {out_dir}", flush=True)
        r = _run_trainer(
            out_dir=out_dir,
            weights=init_weights,
            lr=spec["lr"],
            optimizer=spec["optimizer"],
            stream_max_positions=STREAM_MAX_POSITIONS,
            aux_lr=spec["aux_lr"],
            # Nothing before the last candidate touches training_visits, so
            # every candidate's sample_epoch_keys() call sees identical DB
            # state and returns the identical 16384 keys -- same data, only
            # the training method differs. The real epoch's position usage
            # only advances once, on this last run.
            commit_usage=is_last,
        )
        print(f"  rc={r['returncode']} elapsed={r['elapsed_sec']}s ok={r['ok']}", flush=True)
        if not r["ok"]:
            print(f"  STDERR TAIL: {r['stderr_tail']}", flush=True)
        else:
            d = r["diag"]
            print(
                f"  loss {d.get('train_loss_start')}->{d.get('train_loss_end')}  "
                f"grad_norm_mean={d.get('grad_norm_mean')}  "
                f"update/param={d.get('update_over_param_norm')}",
                flush=True,
            )
        candidates.append({
            "name": spec["name"],
            "optimizer": spec["optimizer"],
            "lr": spec["lr"],
            "aux_lr": spec["aux_lr"],
            "weights": r["weights"],
            "ok": r["ok"],
            "elapsed_sec": r["elapsed_sec"],
            "diagnostics": r["diag"],
        })
    return candidates


def build_entrants(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entrants = [
        {"name": c["name"], "weights": Path(c["weights"]), "kind": "candidate",
         "optimizer": c["optimizer"], "lr": c["lr"]}
        for c in candidates if c["ok"]
    ]
    chain = load_chain()
    epochs = chain.get("epochs") or []
    for e in epochs[-3:]:
        try:
            snap = accepted_snapshot_path(int(e["epoch"]))
        except Exception:
            continue
        if snap.is_file():
            entrants.append({
                "name": f"epoch_{e['epoch']}",
                "weights": snap,
                "kind": "historical",
                "optimizer": None,
                "lr": None,
            })
    return entrants


class SwissTournament:
    def __init__(self, entrants: list[dict[str, Any]]):
        self.entrants = entrants
        self.names = [e["name"] for e in entrants]
        self.score: dict[str, float] = {n: 0.0 for n in self.names}
        self.played: dict[str, set[str]] = {n: set() for n in self.names}
        self.games_played: dict[str, int] = {n: 0 for n in self.names}
        # head-to-head[a][b] = {"wins": n, "losses": n, "draws": n} from a's perspective
        self.head_to_head: dict[str, dict[str, dict[str, int]]] = {
            n: {m: {"wins": 0, "losses": 0, "draws": 0} for m in self.names if m != n}
            for n in self.names
        }
        self._lock = threading.Lock()
        self._sessions: dict[str, EngineSession] = {}
        for e in entrants:
            self._sessions[e["name"]] = EngineSession("titanium-v16", e["weights"])

    def close(self) -> None:
        for s in self._sessions.values():
            s.close()

    def _pairings_for_round(self, rng: random.Random) -> list[tuple[str, str]]:
        # Whoever has played the FEWEST games so far goes first -- guarantees
        # they get paired (partners are still available at that point in the
        # loop) instead of leaving them as the odd-one-out "bye". With an odd
        # entrant count someone always sits out a round; sorting by score
        # first (as this used to) meant the bye always fell to whoever had
        # the lowest score, so one bad early loss could permanently exclude
        # an entrant from ever playing again -- confirmed live: adamw_0.005
        # lost its first 6 games and then got zero games for the rest of a
        # 300-game run. Score is now only a tiebreaker for standard
        # score-based Swiss pairing among equally-caught-up entrants.
        order = sorted(self.names, key=lambda n: (self.games_played[n], -self.score[n], rng.random()))
        used: set[str] = set()
        pairs: list[tuple[str, str]] = []
        for a in order:
            if a in used:
                continue
            partner = None
            for b in order:
                if b == a or b in used:
                    continue
                if b in self.played[a]:
                    continue
                partner = b
                break
            if partner is None:
                for b in order:
                    if b != a and b not in used:
                        partner = b
                        break
            if partner is not None:
                pairs.append((a, partner))
                used.add(a)
                used.add(partner)
        return pairs

    def _play_one(self, name_a: str, name_b: str, rng: random.Random) -> dict[str, Any]:
        a_is_p0 = rng.random() < 0.5
        p0_name, p1_name = (name_a, name_b) if a_is_p0 else (name_b, name_a)
        sess_p0 = self._sessions[p0_name]
        sess_p1 = self._sessions[p1_name]
        moves: list[str] = []
        for ply in range(GAME_MAX_PLY):
            active = sess_p0 if ply % 2 == 0 else sess_p1
            if not active.sync(moves) or not active.alive():
                break
            mv = active.go(TIME_SEC)
            if not mv:
                break
            moves.append(mv)

        from self_play_overnight import check_winner
        winner = check_winner(moves)

        with self._lock:
            self.games_played[name_a] += 1
            self.games_played[name_b] += 1
            self.played[name_a].add(name_b)
            self.played[name_b].add(name_a)
            if winner is None:
                self.score[name_a] += 0.5
                self.score[name_b] += 0.5
                self.head_to_head[name_a][name_b]["draws"] += 1
                self.head_to_head[name_b][name_a]["draws"] += 1
                result = "draw"
            elif (winner == 0) == a_is_p0:
                self.score[name_a] += 1.0
                self.head_to_head[name_a][name_b]["wins"] += 1
                self.head_to_head[name_b][name_a]["losses"] += 1
                result = "a_wins"
            else:
                self.score[name_b] += 1.0
                self.head_to_head[name_b][name_a]["wins"] += 1
                self.head_to_head[name_a][name_b]["losses"] += 1
                result = "b_wins"
        return {"a": name_a, "b": name_b, "result": result, "plies": len(moves)}

    def run(self, total_games: int, concurrency: int) -> list[dict[str, Any]]:
        results = []
        rng = random.Random(42)
        games_done = 0
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            while games_done < total_games:
                pairs = self._pairings_for_round(rng)
                if not pairs:
                    break
                remaining = total_games - games_done
                pairs = pairs[: max(1, min(len(pairs), concurrency, remaining))]
                futures = [pool.submit(self._play_one, a, b, random.Random(rng.random())) for a, b in pairs]
                for f in as_completed(futures):
                    r = f.result()
                    results.append(r)
                    games_done += 1
                    print(f"  [{games_done}/{total_games}] {r['a']} vs {r['b']} -> {r['result']} ({r['plies']} plies)", flush=True)
        return results


def main() -> int:
    import os
    parent_epoch = os.environ.get("LR_SWEEP_PARENT_EPOCH")
    if parent_epoch:
        chain = load_chain()
        matches = [e for e in chain.get("epochs") or [] if int(e["epoch"]) == int(parent_epoch)]
        if not matches:
            raise SystemExit(f"no accepted epoch {parent_epoch} in chain")
        init_weights = accepted_snapshot_path(int(parent_epoch))
        if not init_weights.is_file():
            raise SystemExit(f"epoch {parent_epoch} snapshot missing on disk: {init_weights}")
    else:
        init_weights = resolve_latest_accepted_weights()
    print(f"parent weights: {init_weights}", flush=True)

    print("=== Phase 1: calibrating Muon LR via update-norm/weight-norm ratio ===", flush=True)
    muon_lrs = calibrate_muon_lr(init_weights)

    print("=== Phase 2: training 5 candidates (3 AdamW ladder + 2 calibrated Muon) ===", flush=True)
    candidates = train_candidates(init_weights, muon_lrs)
    ok_count = sum(1 for c in candidates if c["ok"])
    print(f"trained {ok_count}/{len(candidates)} candidates ok", flush=True)

    entrants = build_entrants(candidates)
    print(f"=== Phase 3: Swiss tournament, {len(entrants)} entrants, {TOTAL_GAMES} games ===", flush=True)
    for e in entrants:
        print(f"  entrant: {e['name']} ({e['kind']}) weights={e['weights']}", flush=True)

    tourney = SwissTournament(entrants)
    try:
        results = tourney.run(TOTAL_GAMES, CONCURRENCY)
    finally:
        tourney.close()

    standings = sorted(
        (
            {
                "name": n,
                "score": tourney.score[n],
                "games": tourney.games_played[n],
                "win_rate": round(tourney.score[n] / max(1, tourney.games_played[n]), 4),
                "vs_each_entrant": tourney.head_to_head[n],
            }
            for n in tourney.names
        ),
        key=lambda r: -r["win_rate"],
    )
    print("=== STANDINGS ===", flush=True)
    for s in standings:
        print(f"  {s['name']:28s} score={s['score']:.1f} games={s['games']} win_rate={s['win_rate']}", flush=True)
        for opp, rec in s["vs_each_entrant"].items():
            if rec["wins"] + rec["losses"] + rec["draws"] > 0:
                print(f"      vs {opp:26s} {rec['wins']}W-{rec['draws']}D-{rec['losses']}L", flush=True)

    report = {
        "parent_weights": str(init_weights),
        "calibrated_muon_lrs": muon_lrs,
        "candidates": candidates,
        "entrants": [{"name": e["name"], "kind": e["kind"], "optimizer": e["optimizer"], "lr": e["lr"]} for e in entrants],
        "standings": standings,
        "games": results,
        "total_games": len(results),
        "note": "immediate fitness only (this one epoch's tournament) -- "
                "lineage fitness (does the winning method keep winning over "
                "several generations) requires re-running this script from "
                "the new accepted winner, respawning all variants fresh.",
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report written: {REPORT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
