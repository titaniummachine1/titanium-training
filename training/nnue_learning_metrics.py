"""NNUE learning health - metrics for supervisor + pool UI.

Elo is lagging; these track whether micro-train is moving the eval surface.
Training targets are completed-game WDL outcomes only. Ka pool games are
ordinary completed games in the DB, not single-position teacher labels.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "training" / "data"
STATUS_PATH = DATA / "nnue_learning_status.json"

# Rough guide: need this many micro-trains before plateau verdict is meaningful.
MIN_TRAINED_FOR_VERDICT = 40
# After deploy, expect drift on probes within this many trains or flag stale.
DEPLOY_STALE_TRAIN_GAMES = 48


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ka_pool_stats() -> dict:
    from manifest import CURRENT_ENGINE, load_manifest

    m = load_manifest()
    ka_w, ka_l = 0, 0
    by_tc: dict[str, tuple[int, int]] = {}
    for row in m.get("matchups", {}).values():
        if row.get("a_engine") != CURRENT_ENGINE or row.get("b_engine") != "ka":
            continue
        aw, bw = int(row.get("a_wins", 0)), int(row.get("b_wins", 0))
        ka_w += aw
        ka_l += bw
        tc = row.get("tc_b", "?")
        pw, pl = by_tc.get(tc, (0, 0))
        by_tc[tc] = (pw + aw, pl + bw)
    n = ka_w + ka_l

    return {
        "wins": ka_w,
        "losses": ka_l,
        "games": n,
        "win_rate": round(ka_w / n, 3) if n else None,
        "by_tc": {k: {"w": v[0], "l": v[1]} for k, v in sorted(by_tc.items())},
        "note": "Ka full games train from final winner only; no single-position teacher",
    }


def collect_learning_report(*, write_json: bool = True) -> dict:
    from datagen import DB_PATH, max_game_id, untrained_game_ids
    from nnue_guards import DEPLOY_EVERY_GAMES, load_guard_state
    from plateau_probe import (
        PROBE_PATH,
        TRAINER_DRIFT_STALE_CP,
        TRAINER_STALE_RUNS,
        _load_state,
        candidate_vs_deployed_drift,
    )

    g = load_guard_state()
    state = _load_state()
    th = state.get("trainer_history", [])
    dh = state.get("deploy_history", [])

    trained = int(g.get("games_trained", 0))
    deploys = int(g.get("deploy_runs", 0))
    since_deploy = int(g.get("games_since_deploy", 0))
    last_trained_id = int(g.get("last_trained_game_id", 0))
    mx = max_game_id(DB_PATH)
    pending = untrained_game_ids(DB_PATH, last_trained_id)

    last_drift = float(th[-1].get("eval_drift_cp", 0.0)) if th else 0.0
    recent_drifts = [float(h.get("eval_drift_cp", 0.0)) for h in th[-5:]]
    avg_drift_5 = sum(recent_drifts) / len(recent_drifts) if recent_drifts else 0.0
    cand_drift = candidate_vs_deployed_drift()

    last_move_rate = float(dh[-1].get("move_change_rate", 0.0)) if dh else 0.0
    deploy_eval_drift = float(dh[-1].get("eval_drift_cp", 0.0)) if dh else 0.0

    plateau_runs = (
        len(recent_drifts) >= TRAINER_STALE_RUNS
        and all(d < TRAINER_DRIFT_STALE_CP for d in recent_drifts[-TRAINER_STALE_RUNS:])
    )

    if trained < MIN_TRAINED_FOR_VERDICT:
        phase = "WARMUP"
        verdict = f"need {MIN_TRAINED_FOR_VERDICT - trained} more train games for plateau call"
    elif plateau_runs and cand_drift < 2.0 and last_drift < TRAINER_DRIFT_STALE_CP:
        phase = "PLATEAU"
        verdict = (
            f"eval flat {last_drift:.1f}cp x {TRAINER_STALE_RUNS} probes; "
            f"candidate vs deployed {cand_drift:.1f}cp - net not shifting on probes"
        )
    elif last_drift >= TRAINER_DRIFT_STALE_CP or cand_drift >= 2.0:
        phase = "LEARNING"
        verdict = f"trainer drift {last_drift:.1f}cp, candidate gap {cand_drift:.1f}cp"
    else:
        phase = "QUIET"
        verdict = f"small drift {last_drift:.1f}cp - search may absorb before Elo moves"

    deploy_hint = (
        f"{since_deploy}/{DEPLOY_EVERY_GAMES} since deploy"
        + (
            f" - stale if still flat after {DEPLOY_STALE_TRAIN_GAMES}"
            if since_deploy >= DEPLOY_STALE_TRAIN_GAMES and phase == "PLATEAU"
            else ""
        )
    )

    ka = _ka_pool_stats()

    report = {
        "ts": _ts(),
        "phase": phase,
        "verdict": verdict,
        "games_db": mx,
        "games_trained": trained,
        "pending_train": len(pending),
        "deploy_runs": deploys,
        "games_since_deploy": since_deploy,
        "deploy_every": DEPLOY_EVERY_GAMES,
        "last_trainer_drift_cp": round(last_drift, 2),
        "avg_trainer_drift_5_cp": round(avg_drift_5, 2),
        "candidate_vs_deployed_cp": round(cand_drift, 2),
        "last_deploy_eval_drift_cp": round(deploy_eval_drift, 2),
        "last_deploy_move_rate": round(last_move_rate, 3),
        "recent_trainer_drifts": [round(x, 2) for x in recent_drifts],
        "deploy_hint": deploy_hint,
        "ka_pool": ka,
        "probe_file": str(PROBE_PATH),
    }

    if write_json:
        DATA.mkdir(parents=True, exist_ok=True)
        STATUS_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    return report


def format_supervisor_lines(report: dict) -> list[str]:
    ka = report.get("ka_pool", {})
    ka_wr = ka.get("win_rate")
    ka_s = f"Ka {ka.get('wins',0)}-{ka.get('losses',0)}"
    if ka_wr is not None:
        ka_s += f" ({ka_wr * 100:.0f}% us)"

    line1 = (
        f"NNUE [{report['phase']}] trained={report['games_trained']} "
        f"pending={report['pending_train']} {report['deploy_hint']} "
        f"drift={report['last_trainer_drift_cp']:.1f}cp "
        f"cand={report['candidate_vs_deployed_cp']:.1f}cp "
        f"move_chg={report['last_deploy_move_rate']:.0%}"
    )
    line2 = f"  {report['verdict']}"
    line3 = f"  Ka pool: {ka_s} | {ka.get('note', '')}"
    return [line1, line2, line3]


def format_scoreboard_train_block() -> list[str]:
    """Two lines for compact pool scoreboard."""
    try:
        r = collect_learning_report(write_json=False)
    except Exception:
        return ["| TRAIN status unavailable".ljust(71) + "|"]

    w = 71
    phase = r["phase"]
    line1 = (
        f"| TRAIN {phase:<8} drift {r['last_trainer_drift_cp']:>4.1f}cp "
        f"cand {r['candidate_vs_deployed_cp']:>4.1f}cp "
        f"| {r['games_trained']} trained pending {r['pending_train']}"
    ).ljust(w) + "|"
    line2 = (
        f"| deploy {r['deploy_runs']} (+{r['games_since_deploy']}) "
        f"move_chg {r['last_deploy_move_rate']:.0%} | "
        f"Ka {r['ka_pool'].get('wins',0)}-{r['ka_pool'].get('losses',0)} us"
    ).ljust(w) + "|"
    return [line1, line2]


if __name__ == "__main__":
    import sys

    r = collect_learning_report()
    for ln in format_supervisor_lines(r):
        print(ln)
    print(f"  wrote {STATUS_PATH}")
    sys.exit(0 if r["phase"] in ("LEARNING", "WARMUP", "QUIET") else 1)
