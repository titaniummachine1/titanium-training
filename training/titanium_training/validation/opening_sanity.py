"""Opening-collapse sanity check for promoted Titanium weights.

The first two pawn moves for each side must move toward the center:
    White e1 -> e2 -> e3
    Black e9 -> e8 -> e7

Any wall-first or sideways-first candidate is collapsed enough to block deploy.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
from pathlib import Path

from titanium_training.paths import ENGINE_BIN

EXPECTED_OPENING = ("e2", "e8", "e3", "e7")
_BESTMOVE_RE = re.compile(r"\bbestmove\s+([a-i][1-9](?:[hv])?)\b")


class OpeningSanityError(RuntimeError):
    pass


def _bestmove(
    history: tuple[str, ...],
    *,
    weights_path: Path,
    engine_bin: Path = ENGINE_BIN,
    time_sec: float = 0.15,
) -> str:
    env = os.environ.copy()
    env["TITANIUM_NET_WEIGHTS_PATH"] = str(Path(weights_path).resolve())
    proc = subprocess.run(
        [
            str(engine_bin),
            "genmove",
            "--engine",
            "titanium-v16",
            "--time",
            str(time_sec),
            # The book is PROHIBITED here: collapse detection only means
            # anything if the NET produces e2 e8 e3 e7 by itself. The book
            # force-plays this exact trunk and would mask a collapsed model.
            "--book",
            "off",
            *history,
        ],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    match = _BESTMOVE_RE.search(text)
    if proc.returncode != 0 or not match:
        tail = "\n".join(text.strip().splitlines()[-12:])
        raise OpeningSanityError(
            f"opening sanity could not read bestmove at ply {len(history) + 1}: {tail}"
        )
    return match.group(1)


def opening_sequence(
    weights_path: Path,
    *,
    engine_bin: Path = ENGINE_BIN,
    time_sec: float = 0.15,
    plies: int = 4,
) -> tuple[str, ...]:
    history: list[str] = []
    for _ in range(plies):
        history.append(
            _bestmove(
                tuple(history),
                weights_path=weights_path,
                engine_bin=engine_bin,
                time_sec=time_sec,
            )
        )
    return tuple(history)


def assert_opening_sanity(
    weights_path: Path,
    *,
    engine_bin: Path = ENGINE_BIN,
    time_sec: float = 0.15,
) -> tuple[str, ...]:
    weights_path = Path(weights_path)
    if not weights_path.is_file():
        raise OpeningSanityError(f"weights file missing: {weights_path}")
    seq = opening_sequence(weights_path, engine_bin=engine_bin, time_sec=time_sec)
    if seq != EXPECTED_OPENING:
        raise OpeningSanityError(
            f"collapsed opening for {weights_path}: {' '.join(seq)} "
            f"(expected {' '.join(EXPECTED_OPENING)})"
        )
    return seq


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("weights", type=Path)
    ap.add_argument("--engine-bin", type=Path, default=ENGINE_BIN)
    ap.add_argument("--time", type=float, default=0.15)
    args = ap.parse_args()
    try:
        seq = assert_opening_sanity(args.weights, engine_bin=args.engine_bin, time_sec=args.time)
    except OpeningSanityError as exc:
        print(f"FAIL {exc}")
        return 1
    print(f"PASS {' '.join(seq)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
