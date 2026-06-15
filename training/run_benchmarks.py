"""Run all missing A/B strength benchmarks AND collect training data simultaneously.

Every benchmark game is saved to the training database — no compute wasted.
Results are logged to training/data/benchmarks_log.jsonl for future reference.

Usage:
    python training/run_benchmarks.py                    # run all pending benchmarks
    python training/run_benchmarks.py --list             # show planned benchmarks
    python training/run_benchmarks.py --only <name>      # run a specific benchmark
    python training/run_benchmarks.py --remote-only      # only Ka/Ishtar remote benchmarks
    python training/run_benchmarks.py --local-only       # only local engine benchmarks
    python training/run_benchmarks.py --db PATH          # custom database path

The database is append-only; re-running the same benchmark adds more training
data without overwriting existing results.  Each result entry is tagged with
the engine pair and time control so training can weight by source if needed.

Remote benchmarks (Ka/Ishtar) require internet access and call
site/ishtar_match.js via Node.js.  Exact site strength settings are used:
  Ka:     intuition=1, short=1000, medium=5000, long=20000 visits
  Ishtar: short=3200/p32, medium=200000/p1024, long=1000000/p2048
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from datagen import parse_dump_games, games_to_records

ROOT      = Path(__file__).resolve().parent.parent
BIN       = ROOT / "engine" / "target" / "release" / "titanium.exe"
MATCH_JS  = ROOT / "site" / "ishtar_match.js"
DB_PATH   = ROOT / "training" / "data" / "all_games.jsonl"
LOG_PATH  = ROOT / "training" / "data" / "benchmarks_log.jsonl"

# ── Local engine benchmarks ───────────────────────────────────────────────────
# (name, engine_a, engine_b, games, time_s, openings, notes)
LOCAL_BENCHMARKS = [
    # Key: how does grafted compare to plain ace-v13?
    ("grafted-vs-ace-v13-2s",
     "ace-v13-grafted", "ace-v13",        112, 2.0, "book",
     "grafted (cheap-cert + adaptive-TT) vs plain ACE v13"),

    # Key: how does our best ACE engine compare to Titanium?
    ("grafted-vs-titanium-2s",
     "ace-v13-grafted", "titanium",       112, 2.0, "book",
     "our best ACE engine vs Titanium+cert"),

    # Isolate: value of cheap-cert alone
    ("cert-vs-ace-v13-2s",
     "ace-v13-cert",    "ace-v13",        112, 2.0, "book",
     "cheap-cert (no adaptive-TT) vs plain ACE v13"),

    # Isolate: value of adaptive TT alone
    ("att-vs-ace-v13-2s",
     "ace-v13-att",     "ace-v13",        112, 2.0, "book",
     "adaptive-TT (no cheap-cert) vs plain ACE v13"),

    # Titanium with cert vs without cert
    ("titanium-cert-vs-plain-2s",
     "titanium",        "titanium-plain", 112, 2.0, "book",
     "Titanium+cert vs plain Titanium (value of endgame certificate)"),

    # Dead-zone wall prune: does it help grafted?
    ("dz-vs-grafted-2s",
     "ace-v13-dz",      "ace-v13-grafted",112, 2.0, "book",
     "dead-zone prune vs grafted (NPS vs accuracy trade-off)"),

    # 5s versions of the two most important matchups (more decisive signal)
    ("grafted-vs-ace-v13-5s",
     "ace-v13-grafted", "ace-v13",        112, 5.0, "book",
     "grafted vs ace-v13 @ 5s (deeper signal than 2s)"),

    ("grafted-vs-titanium-5s",
     "ace-v13-grafted", "titanium",       112, 5.0, "book",
     "grafted vs Titanium @ 5s"),

    # Self-play for data diversity (different opening seed spread)
    ("grafted-selfplay-book-2s",
     "ace-v13-grafted", "ace-v13-grafted",224, 2.0, "book",
     "grafted self-play with book openings -- bulk training data"),
]

# ── Remote benchmarks (Ka and Ishtar, all time controls) ─────────────────────
# (name, our_engine, opp, opp_time, games, our_time_s, concurrency, notes)
# visits/parallelism are taken directly from ENGINES config in engine_client.js
# to match the site's exact Alpha strength settings for each preset.
#
# Game counts scaled to expected speed:
#   Ka intuition (1 visit)         -- nearly instant, 224 games
#   Ka short     (1000 visits)     -- fast, 112 games
#   Ka medium    (5000 visits)     -- moderate, 56 games
#   Ka long      (20000 visits)    -- slower, 32 games
#   Ishtar short (3200/p32)        -- moderate, 56 games
#   Ishtar medium(200k/p1024)      -- slow, 24 games
#   Ishtar long  (1M/p2048)        -- very slow, 12 games
#
# Concurrency: since Ishtar/Ka run on remote clusters our CPU is free during
# their think time, so we can safely run more games in parallel than local.
# Keep at 4 so each of OUR engines gets a clean core during ITS think.
REMOTE_BENCHMARKS = [
    # Ka -- full strength ladder
    ("grafted-vs-ka-intuition",
     "ace-v13-grafted", "ka", "intuition", 224, 2.0, 4,
     "vs Ka intuition (1 visit) -- sanity floor"),

    ("grafted-vs-ka-short",
     "ace-v13-grafted", "ka", "short",     112, 2.0, 4,
     "vs Ka short (1000 visits)"),

    ("grafted-vs-ka-medium",
     "ace-v13-grafted", "ka", "medium",     56, 2.0, 4,
     "vs Ka medium (5000 visits)"),

    ("grafted-vs-ka-long",
     "ace-v13-grafted", "ka", "long",       32, 2.0, 4,
     "vs Ka long (20000 visits) -- site Alpha strength"),

    # Ishtar -- skip intuition (2 visits, trivially weak)
    ("grafted-vs-ishtar-short",
     "ace-v13-grafted", "ishtar", "short",  56, 2.0, 4,
     "vs Ishtar short (3200 visits, p=32)"),

    ("grafted-vs-ishtar-medium",
     "ace-v13-grafted", "ishtar", "medium", 24, 2.0, 4,
     "vs Ishtar medium (200k visits, p=1024)"),

    ("grafted-vs-ishtar-long",
     "ace-v13-grafted", "ishtar", "long",   12, 2.0, 4,
     "vs Ishtar long (1M visits, p=2048) -- site Alpha strength"),
]


# ── Local match runner ────────────────────────────────────────────────────────

def run_match_dump(engine_a, engine_b, games, time_s, openings):
    """Run a local match with --dump-games; return (stdout_lines, stderr_text)."""
    cmd = [
        str(BIN), "match",
        "--a", engine_a, "--b", engine_b,
        "--games", str(games),
        "--time", str(time_s),
        "--dump-games",
        "--no-early-stop",   # always complete -- every game is training data
    ]
    if openings == "book":
        cmd += ["--openings", "book"]
    result = subprocess.run(cmd, capture_output=True)
    stdout = result.stdout.decode("utf-8", errors="replace")
    stderr = result.stderr.decode("utf-8", errors="replace")
    return stdout.splitlines(), stderr


def parse_local_match_result(stderr_text):
    """Extract win/loss counts from the STRENGTH MATCH RESULT block."""
    result = {}
    for line in stderr_text.splitlines():
        if "A wins" in line and "B wins" in line:
            parts = line.split("|")
            try:
                result["a_wins"] = int(parts[0].split()[-1])
                result["b_wins"] = int(parts[1].split()[-1])
                result["draws"]  = int(parts[2].split()[-1])
            except (IndexError, ValueError):
                pass
        if "A score" in line and "Elo" in line:
            result["summary"] = line.strip()
    return result


# ── Remote match runner ───────────────────────────────────────────────────────

def run_match_dump_remote(our_engine, opp, opp_time, games, our_time_s, concurrency):
    """Run a remote match via ishtar_match.js --dump-games.

    Returns (stdout_lines, stderr_text).  stdout carries GAME/RESULT pairs;
    stderr carries progress and the MATCH_SUMMARY line.
    """
    cmd = [
        "node", str(MATCH_JS),
        "--engine", our_engine,
        "--opp", opp,
        "--opp-time", opp_time,
        "--games", str(games),
        "--our-time", str(our_time_s),
        "--concurrency", str(concurrency),
        "--dump-games",
    ]
    result = subprocess.run(cmd, capture_output=True, cwd=str(MATCH_JS.parent))
    stdout = result.stdout.decode("utf-8", errors="replace")
    stderr = result.stderr.decode("utf-8", errors="replace")
    return stdout.splitlines(), stderr


def parse_remote_match_result(stderr_text):
    """Parse MATCH_SUMMARY line emitted by ishtar_match.js to stderr."""
    result = {}
    for line in stderr_text.splitlines():
        m = re.search(
            r"MATCH_SUMMARY OUR=(\d+) OPP=(\d+) DRAWS=(\d+) SCORE=([\d.]+)/(\d+) ELO=([+-]?\d+(?:\.\d+)?)",
            line,
        )
        if m:
            our, opp_w, draws, score, n, elo = m.groups()
            result["our_wins"] = int(our)
            result["opp_wins"] = int(opp_w)
            result["draws"]    = int(draws)
            result["n"]        = int(n)
            result["elo"]      = float(elo)
            result["summary"]  = (
                f"OUR {our} | OPP {opp_w} | draws {draws}  "
                f"score {score}/{n}  ~{'+' if float(elo)>=0 else ''}{float(elo):.0f} Elo"
            )
    return result


# ── Shared helpers ────────────────────────────────────────────────────────────

def append_to_db(db_path, records, tag):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with open(db_path, "a") as f:
        for rec in records:
            rec["_src"] = tag
            f.write(json.dumps(rec) + "\n")


def append_log(log_path, entry):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def process_game_records(stdout_lines, tag, db_path, min_ply, max_ply):
    """Parse GAME/RESULT lines, run eval-batch, append training records."""
    game_records = parse_dump_games(stdout_lines)
    n_games = len(game_records)
    print(f"  {n_games} complete games collected")
    if n_games == 0:
        return 0
    print(f"  Running eval-batch on all positions...")
    records = games_to_records(game_records, min_ply, max_ply, sample_rate=1.0)
    append_to_db(db_path, records, tag)
    print(f"  {len(records)} training records -> {db_path.name}")
    return len(records)


# ── Benchmark runners ─────────────────────────────────────────────────────────

def run_local_benchmark(bm, db_path, log_path, min_ply=4, max_ply=150):
    name, eng_a, eng_b, games, time_s, openings, notes = bm
    print(f"\n{'='*60}")
    print(f"LOCAL BENCHMARK: {name}")
    print(f"  A={eng_a}  B={eng_b}  games={games}  time={time_s}s  openings={openings}")
    print(f"  {notes}")
    print(f"{'='*60}")

    t0 = time.time()
    stdout_lines, stderr = run_match_dump(eng_a, eng_b, games, time_s, openings)
    elapsed = time.time() - t0

    for line in stderr.splitlines():
        print(" ", line)

    tag = f"{name}|{eng_a}|{eng_b}|{time_s}s"
    n_records = process_game_records(stdout_lines, tag, db_path, min_ply, max_ply)

    match_result = parse_local_match_result(stderr)
    log_entry = {
        "name": name, "type": "local",
        "engine_a": eng_a, "engine_b": eng_b,
        "games": games, "time_s": time_s, "openings": openings,
        "n_records": n_records, "elapsed_s": round(elapsed, 1), "notes": notes,
        **match_result,
    }
    append_log(log_path, log_entry)

    summary = match_result.get("summary", "no result parsed")
    print(f"\n  RESULT: {summary}")
    print(f"  Elapsed: {elapsed/60:.1f} min  |  {n_records} training records saved")
    return log_entry


def run_remote_benchmark(bm, db_path, log_path, min_ply=4, max_ply=150):
    name, our_engine, opp, opp_time, games, our_time_s, concurrency, notes = bm
    print(f"\n{'='*60}")
    print(f"REMOTE BENCHMARK: {name}")
    print(f"  OUR={our_engine} ({our_time_s}s)  OPP={opp} ({opp_time})")
    print(f"  games={games}  concurrency={concurrency}")
    print(f"  {notes}")
    print(f"{'='*60}")

    t0 = time.time()
    stdout_lines, stderr = run_match_dump_remote(
        our_engine, opp, opp_time, games, our_time_s, concurrency
    )
    elapsed = time.time() - t0

    for line in stderr.splitlines():
        print(" ", line)

    tag = f"{name}|{our_engine}|{opp}-{opp_time}|{our_time_s}s"
    n_records = process_game_records(stdout_lines, tag, db_path, min_ply, max_ply)

    match_result = parse_remote_match_result(stderr)
    log_entry = {
        "name": name, "type": "remote",
        "our_engine": our_engine, "opp": opp, "opp_time": opp_time,
        "games": games, "our_time_s": our_time_s, "concurrency": concurrency,
        "n_records": n_records, "elapsed_s": round(elapsed, 1), "notes": notes,
        **match_result,
    }
    append_log(log_path, log_entry)

    summary = match_result.get("summary", "no result parsed")
    print(f"\n  RESULT: {summary}")
    print(f"  Elapsed: {elapsed/60:.1f} min  |  {n_records} training records saved")
    return log_entry


def db_record_count(db_path):
    if not db_path.exists():
        return 0
    return sum(1 for _ in open(db_path))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list",        action="store_true", help="List all benchmarks and exit")
    ap.add_argument("--only",        default=None,        help="Run only this benchmark by name")
    ap.add_argument("--local-only",  action="store_true", help="Run only local engine benchmarks")
    ap.add_argument("--remote-only", action="store_true", help="Run only remote (Ka/Ishtar) benchmarks")
    ap.add_argument("--db",          default=str(DB_PATH), help="Training database path")
    ap.add_argument("--log",         default=str(LOG_PATH),help="Benchmark log path")
    ap.add_argument("--min-ply",     type=int, default=4)
    ap.add_argument("--max-ply",     type=int, default=150)
    args = ap.parse_args()

    db_path  = Path(args.db)
    log_path = Path(args.log)

    # Merge both benchmark lists, tagging each with its runner
    all_benchmarks = (
        [("local",  bm) for bm in LOCAL_BENCHMARKS] +
        [("remote", bm) for bm in REMOTE_BENCHMARKS]
    )

    if args.list:
        print(f"\n{'LOCAL BENCHMARKS':}")
        print(f"{'NAME':<38} {'A':<20} {'B':<20} {'G':>4} {'T':>4}  NOTES")
        for bm in LOCAL_BENCHMARKS:
            name, a, b, g, t, op, notes = bm
            print(f"{name:<38} {a:<20} {b:<20} {g:>4} {t:>4}s  {notes}")
        print(f"\n{'REMOTE BENCHMARKS (Ka / Ishtar)':}")
        print(f"{'NAME':<38} {'OUR':<20} {'OPP':<8} {'TIME':<10} {'G':>4} {'OUR_T':>5}  NOTES")
        for bm in REMOTE_BENCHMARKS:
            name, our, opp, opp_t, g, our_t, conc, notes = bm
            print(f"{name:<38} {our:<20} {opp:<8} {opp_t:<10} {g:>4} {our_t:>5}s  {notes}")
        return

    # Filter by --only / --local-only / --remote-only
    to_run = []
    for kind, bm in all_benchmarks:
        if args.only is not None and bm[0] != args.only:
            continue
        if args.local_only and kind != "local":
            continue
        if args.remote_only and kind != "remote":
            continue
        to_run.append((kind, bm))

    if not to_run:
        if args.only:
            print(f"No benchmark named '{args.only}'.  Use --list to see available names.")
        else:
            print("No benchmarks matched the given filters.")
        sys.exit(1)

    print(f"Running {len(to_run)} benchmark(s)")
    print(f"Training DB: {db_path}  ({db_record_count(db_path)} existing records)")
    print(f"Results log: {log_path}")

    results = []
    for kind, bm in to_run:
        if kind == "local":
            entry = run_local_benchmark(bm, db_path, log_path, args.min_ply, args.max_ply)
        else:
            entry = run_remote_benchmark(bm, db_path, log_path, args.min_ply, args.max_ply)
        results.append(entry)

    print(f"\n{'='*60}")
    print(f"ALL DONE -- {len(results)} benchmarks complete")
    print(f"Training DB: {db_record_count(db_path)} total records")
    print(f"\nSummary:")
    for e in results:
        summary = e.get("summary", "--")
        print(f"  {e['name']:<38}  {summary}")


if __name__ == "__main__":
    main()
