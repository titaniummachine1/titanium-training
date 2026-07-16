"""Verify the Python HalfPW forward pass matches the Rust engine closely enough.

For each test position, runs `titanium eval <moves> --json`, then compares
`halfpw.forward(net, record)` against the engine's own `eval` field. Any
mismatch means the trainer's forward pass would mis-evaluate in-engine — fix it
before training. Run from repo root:

    python training/titanium_training/validation/parity_check.py
"""

import json
import os
import subprocess
import sys
from pathlib import Path

TRAINING_ROOT = Path(__file__).resolve().parents[2]
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from titanium_training.models.halfpw import Net, forward

from titanium_training.paths import ENGINE_BIN, REPO_ROOT, WEIGHTS_BIN

ROOT = REPO_ROOT
BIN = ENGINE_BIN
WEIGHTS = Path(os.environ.get("TITANIUM_NET_WEIGHTS_PATH", str(WEIGHTS_BIN)))
MAX_ALLOWED_DIFF_CP = 1

# Mid-game positions (both sides hold walls, not near mate) so `eval` is the
# pure net path, not the race/cert override.
POSITIONS = [
    ["e2", "e8", "e3", "e7", "d3h", "f5v"],
    ["e2", "e8", "e3", "e7", "e4", "e6", "a3h", "d4v"],
    ["e2", "e8", "d2", "f8", "c4h", "g5h"],
    ["e2", "e8", "e3", "e7", "d3h", "f5v", "c2h"],
    ["e2", "e8", "e3", "e7", "e4", "e6", "c6h", "f3v", "b5h"],
    ["e2", "e8", "e3", "e7", "e4", "e6", "e5", "d6", "f4h"],
]


def engine_dump(moves):
    out = subprocess.run(
        [str(BIN), "eval", *moves, "--json"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return json.loads(out)


def main():
    net = Net.load(WEIGHTS)
    ok = 0
    max_diff = 0
    for moves in POSITIONS:
        rec = engine_dump(moves)
        got = forward(net, rec)
        exp = rec["eval"]
        diff = abs(int(got) - int(exp))
        max_diff = max(max_diff, diff)
        tag = "OK  " if diff <= MAX_ALLOWED_DIFF_CP else "DIFF"
        if diff <= MAX_ALLOWED_DIFF_CP:
            ok += 1
        print(f"{tag} py={got:6d} engine={exp:6d} diff={diff:2d}  [{' '.join(moves)}]")
    print(f"\n{ok}/{len(POSITIONS)} within {MAX_ALLOWED_DIFF_CP}cp (max diff {max_diff})")
    sys.exit(0 if ok == len(POSITIONS) else 1)


if __name__ == "__main__":
    main()
