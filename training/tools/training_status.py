#!/usr/bin/env python3
"""One-shot consolidated training status report.

Replaces manually checking pid files, coordinator state, pool log tails,
oracle importer state, and the remote Oracle /status endpoint across several
separate commands. Run directly:

    python training/tools/training_status.py

Or via the /training-status slash command (see .claude/commands/).
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

_TRAINING = Path(__file__).resolve().parents[1]
_REPO = _TRAINING.parent
LOG_DIR = _TRAINING / "data" / "overnight_logs"

PID_FILES = {
    "local_game_pool": LOG_DIR / "local_game_pool.pid",
    "training_coordinator": LOG_DIR / "training_coordinator.pid",
    "oracle_importer": LOG_DIR / "oracle_importer.pid",
}


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _pid_alive(pid: int) -> bool:
    try:
        import psutil

        return psutil.pid_exists(pid) and psutil.Process(pid).is_running()
    except Exception:
        try:
            os.kill(pid, 0)
            return True
        except Exception:
            return False


def _tail_lines(path: Path, n: int) -> list[str]:
    if not path.is_file():
        return []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return [ln.rstrip("\n") for ln in lines[-n:]]
    except Exception:
        return []


def _last_matching(path: Path, needle: str) -> str | None:
    for line in reversed(_tail_lines(path, 4000)):
        if needle in line:
            return line
    return None


def _oracle_status(token_path: Path, url: str = "http://127.0.0.1:8765/status") -> dict | None:
    if not token_path.is_file():
        return None
    token = token_path.read_text(encoding="utf-8").strip()
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def _prior_epoch_strength() -> dict:
    sys.path.insert(0, str(_TRAINING))
    try:
        import streaming_epoch_validation as sev

        strength = sev._prior_epoch_selfplay_strength()
        strength["min_games"] = sev.PRIOR_EPOCH_MIN_GAMES
        strength["min_score"] = sev.PRIOR_EPOCH_MIN_SCORE
        return strength
    except Exception as exc:
        return {"error": str(exc)}


def main() -> int:
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("TRAINING STATUS")
    lines.append("=" * 60)

    # --- process aliveness ---
    lines.append("\n[processes]")
    for name, pid_file in PID_FILES.items():
        if not pid_file.is_file():
            lines.append(f"  {name:22s} NO PID FILE")
            continue
        try:
            pid = int(pid_file.read_text(encoding="utf-8").strip())
        except ValueError:
            lines.append(f"  {name:22s} BAD PID FILE")
            continue
        alive = _pid_alive(pid)
        status = f"alive (pid {pid})" if alive else f"DEAD (stale pid {pid})"
        lines.append(f"  {name:22s} {status}")

    # --- coordinator state ---
    lines.append("\n[coordinator]")
    coord = _read_json(LOG_DIR / "training_coordinator_state.json")
    if coord:
        lines.append(f"  state                  {coord.get('state')}")
        lines.append(f"  last_error             {coord.get('last_error')}")
        lines.append(f"  completed_cycles       {coord.get('completed_training_cycles')}")
        lines.append(f"  pending_new_eligible   {coord.get('pending_new_eligible')}")
        lines.append(f"  eligible_positions     {coord.get('eligible_positions')}")
    else:
        lines.append("  (no state file)")

    # --- accepted chain tail ---
    lines.append("\n[accepted checkpoint chain] (last 5)")
    chain = _read_json(LOG_DIR / "accepted_checkpoint_chain.json")
    epochs = chain.get("epochs") or []
    for e in epochs[-5:]:
        sha = (e.get("sha256") or "")[:12]
        lines.append(f"  epoch {e.get('epoch'):>3}  {e.get('accepted_at')}  {sha}")
    if not epochs:
        lines.append("  (none)")

    # --- local pool progress ---
    lines.append("\n[local pool]")
    pool_line = _last_matching(LOG_DIR / "local_game_pool.log", "Continuous pool:")
    lines.append(f"  {pool_line or '(no startup line found)'}")
    last_game = _last_matching(LOG_DIR / "local_game_pool.log", "] game ")
    lines.append(f"  last game: {last_game or '(none)'}")

    # --- oracle importer ---
    lines.append("\n[oracle importer]")
    oi = _read_json(LOG_DIR / "oracle_importer_state.json")
    if oi:
        lines.append(f"  imports_total          {oi.get('imports_total')}")
        lines.append(f"  last_import_at         {oi.get('last_import_at')}")
        lines.append(f"  last_game_id           {oi.get('last_game_id')}")
    else:
        lines.append("  (no state file)")

    # --- oracle remote worker ---
    lines.append("\n[oracle remote worker]")
    token_path = Path(os.environ.get("LOCALAPPDATA", "")) / "titanium-oracle-api-token"
    ostat = _oracle_status(token_path)
    if ostat is None:
        lines.append("  unreachable (tunnel down, or /status timed out)")
    else:
        lines.append(f"  ok                     {ostat.get('ok')} paused={ostat.get('paused')}")
        lines.append(f"  workers                {ostat.get('workers_configured')}/{ostat.get('workers_expected')} (nproc={ostat.get('nproc')})")
        lines.append(f"  completed (this gen)   {ostat.get('completed')}")
        lines.append(f"  games_per_hour         {ostat.get('games_per_hour')}")
        lines.append(f"  matchup_counts         {ostat.get('matchup_counts')}")

    # --- real strength gate ---
    lines.append("\n[strength gate: current vs immediately-previous accepted]")
    strength = _prior_epoch_strength()
    if "error" in strength:
        lines.append(f"  error: {strength['error']}")
    else:
        games = strength.get("games", 0)
        min_games = strength.get("min_games")
        score = strength.get("score")
        lines.append(f"  games since last accept   {games} / {min_games}")
        lines.append(f"  score so far              {score}")
        lines.append(f"  wins/draws/losses         {strength.get('wins')}/{strength.get('draws')}/{strength.get('losses')}")
        lines.append(f"  since                     {strength.get('since')}")

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
