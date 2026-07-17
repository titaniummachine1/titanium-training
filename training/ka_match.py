"""One-off games against the remote Ka engine (quoridor-ai.com), reusing the
existing tested harness (site/ishtar_match.js) instead of reimplementing the
WebSocket protocol in Python.

Ka is a stateless remote search API, not a local process -- see
.cursor/rules/ka-remote-engine.mdc and site/extracted/ENGINE_PROTOCOL.md.
Calls here are deliberately rare (a small fraction of self-play games) and
serialized (KA_CALL_LOCK) so the pool never hits the remote service with more
than one game at a time, regardless of thread count.
"""
from __future__ import annotations

import re
import subprocess
import threading
from pathlib import Path
from typing import Any

_TRAINING = Path(__file__).resolve().parent
_REPO = _TRAINING.parent
_SITE = _REPO / "site"
ISHTAR_MATCH_JS = _SITE / "ishtar_match.js"

# Ka is a remote, rate-limited-by-etiquette third party service (see the ToS
# note in ENGINE_PROTOCOL.md) -- only one game against it in flight at a time,
# regardless of how many local pool worker threads exist.
KA_CALL_LOCK = threading.Lock()

_GAME_RE = re.compile(r"^GAME (.+)$", re.MULTILINE)
_RESULT_RE = re.compile(r"^RESULT (\w)$", re.MULTILINE)


def play_ka_game(
    *,
    engine: str,
    weights: Path | None,
    time_sec: float,
    engine_bin: Path,
    our_is_p0: bool,
    opp_time: str = "intuition",
    max_ply: int = 128,
    timeout_sec: float = 900.0,
) -> dict[str, Any] | None:
    """Play one game vs Ka, our side chosen by the caller (our_is_p0) so the
    result can be scored correctly -- the harness's own --our-side random
    doesn't report back which side it picked via --dump-games output.

    Returns {"moves": [...], "outcome_p0": 1|-1|0, "current_won": bool} or
    None if the game didn't complete (timeout, network issue, incomplete)."""
    if not ISHTAR_MATCH_JS.is_file():
        return None

    import os

    env = os.environ.copy()
    if weights is not None and Path(weights).is_file():
        env["TITANIUM_NET_WEIGHTS_PATH"] = str(Path(weights).resolve())
    else:
        env.pop("TITANIUM_NET_WEIGHTS_PATH", None)
    env["NO_PROGRESS"] = "1"

    cmd = [
        "node",
        str(ISHTAR_MATCH_JS),
        "--engine", engine,
        "--opp", "ka",
        "--opp-time", opp_time,
        "--games", "1",
        "--our-side", "p1" if our_is_p0 else "p2",
        "--dump-games",
        "--source-tag", "",  # skip the (unavailable) old coordinator's persistGame
        "--no-fair-time",
        "--our-time", str(time_sec),
        "--bin", str(engine_bin),
        "--max-ply", str(max_ply),
    ]
    with KA_CALL_LOCK:
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(_SITE),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired:
            return None

    stdout = proc.stdout or ""
    game_match = _GAME_RE.search(stdout)
    result_match = _RESULT_RE.search(stdout)
    if not game_match or not result_match:
        return None

    moves = game_match.group(1).split()
    if not moves:
        return None
    winner_letter = result_match.group(1)  # W = P1 (p0) won, B = P2 (p1) won
    outcome_p0 = 1 if winner_letter == "W" else -1 if winner_letter == "B" else 0
    current_won = (outcome_p0 == 1) == our_is_p0
    return {"moves": moves, "outcome_p0": outcome_p0, "current_won": current_won}
