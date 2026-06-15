"""Training data generation for HalfPW retrain.

Generates (features, target) records from self-play games using the
current engine.  Each record contains:
  - All fields from `titanium eval <moves> --json` (turn, pawns, walls, eval,
    d0/d1 scalars, d0_field/d1_field arrays, hw/vw)
  - Geometry inputs computed from the BFS distance fields:
      delta0[cell] = d0_field[cell] - d0  (signed distance above pawn's path rank for P0)
      delta1[cell] = d1_field[cell] - d1  (same for P1)
      corridor_width0 = count(d0_field[cell] == d0)
      corridor_width1 = count(d1_field[cell] == d1)
  - Target: game outcome (+1 = P0 wins, -1 = P1 wins).
    Quoridor has no draws; ply-cap adjudications are discarded.

Usage:
    python training/datagen.py --games 500 --time 0.2 --out data/games.jsonl

Options:
    --games N       Number of self-play games (default: 200)
    --time S        Seconds per move (default: 0.1)
    --engine E      Engine variant to self-play (default: ace-v13-grafted)
    --out PATH      Output JSONL file (default: training/data/games.jsonl)
    --min-ply N     Skip positions before this ply (default: 4)
    --max-ply N     Skip positions after this ply (default: 150)
    --sample-rate R Sample each position with probability R (default: 1.0)
    --openings book Use book-weighted openings (default: random)
"""

import argparse
import json
import subprocess
import sys
import random
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent
BIN     = ROOT / "engine" / "target" / "release" / "titanium.exe"
WEIGHTS = ROOT / "engine" / "src" / "acev13" / "net_weights.bin"


def run_match(engine, games, time_s, openings):
    """Run a self-play match and return raw stdout lines."""
    cmd = [
        str(BIN), "match",
        "--a", engine, "--b", engine,
        "--games", str(games),
        "--time", str(time_s),
        "--dump-games",     # prints game move lists to stdout
    ]
    if openings == "book":
        cmd += ["--openings", "book"]
    result = subprocess.run(cmd, capture_output=True, check=True)
    return result.stdout.decode("utf-8", errors="replace").splitlines()


def eval_batch(all_move_lists):
    """Feed all move sequences to titanium eval-batch in one subprocess call.

    Returns a list of JSON records in the same order as all_move_lists.
    Single startup cost regardless of how many positions — orders of magnitude
    faster than launching `titanium eval --json` per position.
    """
    stdin_text = "\n".join(" ".join(m) if m else "" for m in all_move_lists) + "\n"
    result = subprocess.run(
        [str(BIN), "eval-batch"],
        input=stdin_text.encode("utf-8"),
        capture_output=True, check=True,
    )
    lines = result.stdout.decode("utf-8", errors="replace").splitlines()
    return [json.loads(l) for l in lines if l.strip()]


def compute_geometry(rec):
    """Compute geometry features from d0_field/d1_field in the record."""
    d0f  = rec.get("d0_field", [])
    d1f  = rec.get("d1_field", [])
    d0   = rec["d0"]
    d1   = rec["d1"]

    # delta fields: how much longer than the shortest path is each cell?
    # Clamped to [0, 127] and capped at 255 for unreachable cells.
    def delta_field(bfs_field, shortest):
        return [min(255, max(0, int(v) - shortest)) for v in bfs_field]

    delta0 = delta_field(d0f, d0)
    delta1 = delta_field(d1f, d1)

    # Corridor width: number of cells at the pawn's own distance-to-goal rank
    width0 = sum(1 for d in d0f if int(d) == d0)
    width1 = sum(1 for d in d1f if int(d) == d1)

    return {
        "delta0": delta0,
        "delta1": delta1,
        "corridor_width0": width0,
        "corridor_width1": width1,
    }


def games_to_records(games, min_ply, max_ply, sample_rate):
    """Convert a list of (move_list, outcome) games into training records using
    a single eval-batch call — one subprocess for all positions.
    """
    # Collect all (ply_index, move_list_slice, outcome) entries first
    entries = []
    for move_list, outcome in games:
        for ply in range(min_ply, min(max_ply + 1, len(move_list) + 1)):
            if sample_rate < 1.0 and random.random() > sample_rate:
                continue
            entries.append((ply, move_list[:ply], outcome))

    if not entries:
        return []

    # One batch eval call for all positions
    all_move_lists = [e[1] for e in entries]
    evals = eval_batch(all_move_lists)

    records = []
    for (ply, _, outcome), rec in zip(entries, evals):
        geom = compute_geometry(rec)
        rec.update(geom)
        rec["outcome"] = outcome   # +1 = P0 wins, -1 = P1 wins (no draws)
        rec["ply"] = ply
        records.append(rec)
    return records


def parse_dump_games(lines):
    """Parse --dump-games output into list of (move_list, outcome) tuples.

    Expected format (one game per two lines):
        GAME <moves...>
        RESULT <W|B|D>
    """
    games = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("GAME "):
            moves = line.split()[1:]
            result_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
            if not result_line.startswith("RESULT "):
                i += 2
                continue
            r = result_line.split()[1]
            if r not in ("W", "B"):
                i += 2
                continue  # skip adjudicated ply-cap games; Quoridor always has a winner
            outcome = 1 if r == "W" else -1
            games.append((moves, outcome))
            i += 2
        else:
            i += 1
    return games


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games",       type=int,   default=200)
    ap.add_argument("--time",        type=float, default=0.1)
    ap.add_argument("--engine",      default="ace-v13-grafted")
    ap.add_argument("--out",         default="training/data/games.jsonl")
    ap.add_argument("--min-ply",     type=int,   default=4)
    ap.add_argument("--max-ply",     type=int,   default=150)
    ap.add_argument("--sample-rate", type=float, default=1.0)
    ap.add_argument("--openings",    default="random",
                    choices=["random", "book"])
    ap.add_argument("--from-file",   default=None, metavar="PATH",
                    help="Read GAME/RESULT lines from this file instead of running a match. "
                         "Appends to --out. Use with self_match.js --save-games.")
    ap.add_argument("--append",      action="store_true",
                    help="Append to --out instead of overwriting (implied by --from-file).")
    args = ap.parse_args()

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.from_file:
        # Ingest pre-recorded GAME/RESULT lines (e.g. from self_match.js --save-games).
        src = Path(args.from_file)
        if not src.exists():
            print(f"ERROR: --from-file path not found: {src}")
            sys.exit(1)
        lines = src.read_text(encoding="utf-8").splitlines()
        games = parse_dump_games(lines)
        if not games:
            print("No games found in file.")
            sys.exit(1)
        print(f"Ingesting {len(games)} games from {src} ...")
        records = games_to_records(games, args.min_ply, args.max_ply, args.sample_rate)
        mode = "a"  # always append when reading from a file
    else:
        print(f"Generating {args.games} games @ {args.time}s/move with {args.engine}...")
        try:
            lines = run_match(args.engine, args.games, args.time, args.openings)
        except subprocess.CalledProcessError:
            print("ERROR: titanium match --dump-games not yet supported.")
            print("Add the --dump-games flag to 'titanium match' in main.rs, then re-run.")
            sys.exit(1)
        games = parse_dump_games(lines)
        if not games:
            print("No games parsed from output.  Is --dump-games implemented in the engine?")
            sys.exit(1)
        print(f"  {len(games)} games parsed; running eval-batch on all positions...")
        records = games_to_records(games, args.min_ply, args.max_ply, args.sample_rate)
        mode = "a" if args.append else "w"

    with open(out_path, mode) as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"Done: {len(records)} training records -> {out_path}")


if __name__ == "__main__":
    main()
