#!/usr/bin/env python3
"""git bisect run helper — is this engine commit weaker than partial-iter golden?

Builds the checked-out engine commit, plays 8 games @ 2s vs abe9ba5 reference.
Exit 0 = good (no clear regression), 1 = bad (golden wins >= 5/8), 125 = skip.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENGINE = ROOT / "engine"
GOLDEN = ENGINE / "worktrees" / "partial-golden" / "target" / "release" / "titanium.exe"
if os.name == "nt":
    GOLDEN = GOLDEN.with_suffix(".exe")
CAND_BIN = ENGINE / "target" / "release" / ("titanium.exe" if os.name == "nt" else "titanium")
SELF_MATCH = ROOT / "site" / "self_match.js"
GAMES_FILE = ROOT / "training" / "data" / "bisect_games.games"
LOG = ROOT / "training" / "data" / "bisect_step.log"

GAMES = int(os.environ.get("BISECT_GAMES", "8"))
TIME = float(os.environ.get("BISECT_TIME", "2"))
CONC = int(os.environ.get("BISECT_CONCURRENCY", "4"))
ENGINE_FLAG = "ace-v13-grafted"


def run(cmd: list[str], *, cwd: Path | None = None, timeout: int = 3600) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=cwd or ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def log(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    rev = run(["git", "rev-parse", "--short", "HEAD"], cwd=ENGINE).stdout.strip()
    line = f"[{rev}] {msg}\n"
    LOG.open("a", encoding="utf-8").write(line)
    print(line, end="")


def main() -> int:
    if not GOLDEN.exists():
        log("SKIP: golden binary missing — build worktree/partial-golden first")
        return 125

    rev = run(["git", "rev-parse", "HEAD"], cwd=ENGINE).stdout.strip()
    log(f"bisect step @ {rev}")

    b = run(["cargo", "build", "--release", "-p", "titanium"], cwd=ENGINE, timeout=600)
    if b.returncode != 0:
        log(f"SKIP: build failed\n{b.stderr[-500:]}")
        return 125
    if not CAND_BIN.exists():
        log("SKIP: candidate binary missing after build")
        return 125

    tag = f"bisect-{rev[:8]}-vs-golden"
    cmd = [
        "node",
        str(SELF_MATCH),
        "--engine-a", ENGINE_FLAG,
        "--engine-b", ENGINE_FLAG,
        "--bin-a", str(GOLDEN),
        "--bin-b", str(CAND_BIN),
        "--games", str(GAMES),
        "--time", str(TIME),
        "--concurrency", str(CONC),
        "--no-ponder",
        "--standalone",
        "--source-tag", tag,
        "--save-games", str(GAMES_FILE),
    ]
    m = run(cmd, timeout=3600)
    out = (m.stdout or "") + "\n" + (m.stderr or "")
    if m.returncode != 0:
        log(f"SKIP: match failed rc={m.returncode}\n{out[-800:]}")
        return 125

    summary = re.search(r"MATCH_SUMMARY A=(\d+) B=(\d+)", out)
    if not summary:
        log(f"SKIP: no MATCH_SUMMARY\n{out[-800:]}")
        return 125

    a_w, b_w = int(summary.group(1)), int(summary.group(2))
    # A = golden reference, B = candidate at this commit.
    golden_wins = a_w
    log(f"score golden {a_w} - {b_w} candidate ({GAMES}g @ {TIME}s)")

    if golden_wins >= (GAMES // 2 + 1):
        log("verdict: BAD (golden stronger — regression at or before this commit)")
        return 1
    log("verdict: GOOD")
    return 0


if __name__ == "__main__":
    sys.exit(main())
