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
SELF_MATCH = ROOT / "site" / "self_match.js"
MATCH_JS  = ROOT / "site" / "ishtar_match.js"
DB_PATH   = ROOT / "training" / "data" / "all_games.db"
LOG_PATH  = ROOT / "training" / "data" / "benchmarks_log.jsonl"
BENCH_GAMES_DIR = ROOT / "training" / "data" / "benchmarks"

# Current production engine (Titanium v15).  ace-v13-ti-pure is the JS baseline.
CURRENT = "titanium-v15"

# ── Local engine benchmarks ───────────────────────────────────────────────────
# (name, engine_a, engine_b, games, time_s, openings, notes)
LOCAL_BENCHMARKS = [
    # Primary: how far is v15 beyond the JS v13 baseline?
    ("v15-vs-ti-pure-5s",
     CURRENT, "ace-v13-ti-pure", 112, 5.0, "book",
     "Titanium v15 vs JS v13 baseline (+ O1 movegen only) @ 5s"),

    # v15 vs plain ace-v13 (O1 movegen, no cert/TT extras)
    ("v15-vs-ace-v13-2s",
     CURRENT, "ace-v13",        112, 2.0, "book",
     "Titanium v15 vs plain ace-v13 @ 2s"),

    # v15 vs legacy Titanium+cert stack
    ("v15-vs-titanium-2s",
     CURRENT, "titanium",       112, 2.0, "book",
     "Titanium v15 vs Titanium+cert @ 2s"),

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

    # Dead-zone wall prune vs v15
    ("dz-vs-v15-2s",
     "ace-v13-dz",      CURRENT,          112, 2.0, "book",
     "dead-zone prune vs Titanium v15 (NPS vs accuracy trade-off)"),

    # 5s versions of key matchups
    ("v15-vs-ace-v13-5s",
     CURRENT, "ace-v13",        112, 5.0, "book",
     "Titanium v15 vs ace-v13 @ 5s"),

    ("v15-vs-titanium-5s",
     CURRENT, "titanium",       112, 5.0, "book",
     "Titanium v15 vs Titanium+cert @ 5s"),

    # Self-play for data diversity
    ("v15-selfplay-book-2s",
     CURRENT, CURRENT,          224, 2.0, "book",
     "Titanium v15 self-play with book openings — bulk training data"),
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
    # Ka -- full strength ladder (~12s/move server-side computation)
    # Persistent WS per game avoids ~25s cold-start per move.
    # Concurrency 2: enough parallelism without overloading the remote server.
    ("titanium-v15-vs-ka-intuition",
     CURRENT, "ka", "intuition",  32, 2.0, 2,
     "vs Ka intuition (1 visit) -- sanity floor"),

    ("titanium-v15-vs-ka-short",
     CURRENT, "ka", "short",      32, 2.0, 2,
     "vs Ka short (1000 visits)"),

    ("titanium-v15-vs-ka-medium",
     CURRENT, "ka", "medium",     20, 2.0, 2,
     "vs Ka medium (5000 visits)"),

    ("titanium-v15-vs-ka-long",
     CURRENT, "ka", "long",       16, 2.0, 2,
     "vs Ka long (20000 visits) -- site Alpha strength"),

    # Ishtar -- skip intuition (2 visits, trivially weak)
    # Ishtar uses MCTS with heavy parallelism; moves take 5-30s depending on preset.
    ("titanium-v15-vs-ishtar-short",
     CURRENT, "ishtar", "short",  32, 2.0, 2,
     "vs Ishtar short (3200 visits, p=32)"),

    ("titanium-v15-vs-ishtar-medium",
     CURRENT, "ishtar", "medium", 16, 2.0, 2,
     "vs Ishtar medium (200k visits, p=1024)"),

    ("titanium-v15-vs-ishtar-long",
     CURRENT, "ishtar", "long",    8, 2.0, 2,
     "vs Ishtar long (1M visits, p=2048) -- site Alpha strength"),
]


# ── Local match runner ────────────────────────────────────────────────────────

def run_self_match(engine_a, engine_b, games, time_s, save_path, tag, concurrency=4):
    """Run pondering self-match; games saved + ingested per-game by self_match.js."""
    BENCH_GAMES_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        "node", str(SELF_MATCH),
        "--engine-a", engine_a,
        "--engine-b", engine_b,
        "--games", str(games),
        "--time", str(time_s),
        "--ponder-time", str(time_s),
        "--concurrency", str(concurrency),
        "--save-games", str(save_path),
        "--source-tag", tag,
    ]
    result = subprocess.run(cmd, capture_output=True, cwd=str(ROOT))
    stdout = result.stdout.decode("utf-8", errors="replace")
    stderr = result.stderr.decode("utf-8", errors="replace")
    return stdout.splitlines(), stderr


def parse_self_match_result(stderr_text):
    """Parse MATCH_SUMMARY from self_match.js stderr."""
    result = {}
    for line in stderr_text.splitlines():
        m = re.search(
            r"MATCH_SUMMARY A=(\d+) B=(\d+) DRAWS=(\d+) SCORE=([\d.]+)/(\d+) ELO=([+-]?(?:Infinity|\d+(?:\.\d+)?))",
            line,
        )
        if m:
            a, b, d, score, n, elo = m.groups()
            result["a_wins"] = int(a)
            result["b_wins"] = int(b)
            result["draws"]  = int(d)
            result["n"]      = int(n)
            result["elo"]    = float(elo)
            result["summary"] = (
                f"A {a} | B {b} | draws {d}  score {score}/{n}  "
                f"~{'+' if float(elo) >= 0 else ''}{float(elo):.0f} Elo"
            )
    return result


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
            r"MATCH_SUMMARY OUR=(\d+) OPP=(\d+) DRAWS=(\d+) SCORE=([\d.]+)/(\d+) ELO=([+-]?(?:Infinity|\d+(?:\.\d+)?))",
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
    save_path = BENCH_GAMES_DIR / f"{name}.games"
    tag = f"{name}|{eng_a}|{eng_b}|{time_s}s"
    print(f"\n{'='*60}")
    print(f"LOCAL BENCHMARK: {name}")
    print(f"  A={eng_a}  B={eng_b}  games={games}  time={time_s}s  openings={openings}")
    print(f"  games file: {save_path}")
    print(f"  {notes}")
    print(f"{'='*60}")

    t0 = time.time()
    _, stderr = run_self_match(eng_a, eng_b, games, time_s, save_path, tag)
    elapsed = time.time() - t0

    for line in stderr.splitlines():
        print(" ", line)

    # self_match.js ingests each game incrementally — no duplicate batch ingest here.
    from manifest import update_source, count_games_in_file
    update_source(name, save_path, engine_a=eng_a, engine_b=eng_b)
    n_games = count_games_in_file(save_path)

    match_result = parse_self_match_result(stderr)
    log_entry = {
        "name": name, "type": "local",
        "engine_a": eng_a, "engine_b": eng_b,
        "games": games, "time_s": time_s, "openings": openings,
        "games_file": str(save_path),
        "n_games_saved": n_games,
        "n_records": "incremental",  # ingested per-game by self_match.js
        "elapsed_s": round(elapsed, 1), "notes": notes,
        **match_result,
    }
    append_log(log_path, log_entry)

    summary = match_result.get("summary", "no result parsed")
    print(f"\n  RESULT: {summary}")
    print(f"  Elapsed: {elapsed/60:.1f} min  |  {n_games} games -> {save_path.name} (ingested incrementally)")
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
