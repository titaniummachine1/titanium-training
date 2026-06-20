"""Compare training vs frozen HalfPW on fixed probe positions.

    python training/compare_halfpw.py
    python training/compare_halfpw.py --match   # run 8-game self-match v15 vs v15-frozen
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "engine" / "target" / "release" / "titanium.exe"
TRAIN = ROOT / "engine" / "src" / "acev13" / "net_weights.bin"
FROZEN = ROOT / "engine" / "src" / "acev13" / "net_weights_frozen.bin"

PROBE = [
    ["e2", "e8", "e3", "e7", "d3h", "f5v"],
    ["e2", "e8", "e3", "e7", "e4", "e6", "a3h", "d4v"],
    ["e2", "e8", "d2", "f8", "c4h", "g5h"],
    ["e2", "e8", "e3", "e7", "d3h", "f5v", "c2h"],
    ["e2", "e8", "e3", "e7", "e4", "e6", "c6h", "f3v", "b5h"],
    ["e2", "e8", "e3", "e7", "e4", "e6", "e5", "d6", "f4h"],
]


def _eval_engine(engine: str, moves: list[str]) -> int:
    cmd = [str(BIN), "eval", *moves, "--json"]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return int(json.loads(out.stdout.strip())["eval"])


def _eval_halfpw(weights: Path, moves: list[str]) -> int:
        from titanium_training.models.halfpw import Net, forward

    stdin = " ".join(moves) + "\n"
    proc = subprocess.run(
        [str(BIN), "eval-batch"],
        input=stdin.encode(),
        capture_output=True,
        check=True,
    )
    rec = json.loads(proc.stdout.decode().splitlines()[0])
    net = Net.load(str(weights))
    return int(round(forward(net, rec)))


def compare_evals() -> None:
    if not FROZEN.exists():
        print("Run: python training/freeze_baseline_weights.py")
        sys.exit(1)
    if not BIN.exists():
        print(f"Missing {BIN} — cargo build --release -p titanium")
        sys.exit(1)

    print("HalfPW probe: training (net_weights.bin) vs frozen (v13 baseline)")
    print(f"  train  {TRAIN.stat().st_size} bytes")
    print(f"  frozen {FROZEN.stat().st_size} bytes")
    if TRAIN.read_bytes() == FROZEN.read_bytes():
        print("  WARNING: blobs are identical — no training delta yet")
    print()

    drifts = []
    for i, moves in enumerate(PROBE):
        e_train = _eval_engine("titanium-v15", moves)
        e_frozen = _eval_halfpw(FROZEN, moves)
        # frozen engine flag uses embedded frozen blob
        cmd = [str(BIN), "genmove", "--engine", "titanium-v15-frozen", "--time", "0.01", *moves]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        root_frozen = "(none)"
        for line in (proc.stdout + proc.stderr).splitlines():
            if line.strip().startswith("bestmove "):
                root_frozen = line.strip().split(" ", 1)[1]
                break
        cmd[3] = "titanium-v15"
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        root_train = "(none)"
        for line in (proc.stdout + proc.stderr).splitlines():
            if line.strip().startswith("bestmove "):
                root_train = line.strip().split(" ", 1)[1]
                break
        drift = abs(e_train - e_frozen)
        drifts.append(drift)
        mv = "same" if root_train == root_frozen else f"{root_frozen} -> {root_train}"
        print(f"  [{i+1}] train={e_train:+4d}  frozen={e_frozen:+4d}  drift={drift}cp  root {mv}")

    mean = sum(drifts) / len(drifts)
    print(f"\nmean |eval| drift: {mean:.1f} cp")


def run_match(games: int, time_sec: float) -> None:
    script = ROOT / "site" / "self_match.js"
    cmd = [
        "node", str(script),
        "--engine-a", "titanium-v15",
        "--engine-b", "titanium-v15-frozen",
        "--time", str(time_sec),
        "--games", str(games),
        "--no-ponder",
        "--source-tag", "v15-vs-v15-frozen",
        "--save-games", str(ROOT / "training" / "data" / "v15_vs_frozen.games"),
    ]
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--match", action="store_true", help="run self-match after eval compare")
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--time", type=float, default=5.0)
    args = ap.parse_args()
    compare_evals()
    if args.match:
        print(f"\n--- self-match {args.games}g @ {args.time}s ---")
        run_match(args.games, args.time)


if __name__ == "__main__":
    main()
