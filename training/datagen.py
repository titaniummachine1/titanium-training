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

Storage: SQLite (training/data/all_games.db).  Arrays are stored as packed
uint8 BLOBs — ~3x smaller than JSONL and random-access fast.

Usage:
    python training/datagen.py --games 500 --time 0.2

Options:
    --games N         Number of self-play games (default: 200)
    --time S          Seconds per move (default: 0.1)
    --engine E        Engine variant to self-play (default: titanium-v15)
    --out PATH        Output DB (default: training/data/all_games.db)
    --min-ply N       Skip positions before this ply (default: 4)
    --max-ply N       Skip positions after this ply (default: 150)
    --sample-rate R   Sample each position with probability R (default: 1.0)
    --openings book   Use book-weighted openings (default: random)
    --from-file PATH  Read GAME/RESULT lines from file instead of running a match
    --incremental PATH  Ingest only new bytes from PATH (tracks .ingested_offset)
    --tag NAME        Source tag stored as src on each record
    --migrate-jsonl PATH  One-time migration: JSONL → DB, then exit
"""

import argparse
import json
import sqlite3
import struct
import subprocess
import sys
import random
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent
BIN     = ROOT / "engine" / "target" / "release" / "titanium.exe"
WEIGHTS = ROOT / "engine" / "src" / "acev13" / "net_weights.bin"
DB_PATH = ROOT / "training" / "data" / "all_games.db"

# ── SQLite helpers ────────────────────────────────────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    id      INTEGER PRIMARY KEY,
    src     TEXT,
    ply     INTEGER,
    turn    INTEGER,
    outcome INTEGER,
    d0      INTEGER,
    d1      INTEGER,
    eval    INTEGER,
    pawn0   INTEGER,
    pawn1   INTEGER,
    wl0     INTEGER,
    wl1     INTEGER,
    cw0     INTEGER,
    cw1     INTEGER,
    d0_field BLOB,
    d1_field BLOB,
    delta0   BLOB,
    delta1   BLOB,
    hw       BLOB,
    vw       BLOB
);
CREATE INDEX IF NOT EXISTS idx_src ON records(src);
"""


def open_db(path: Path) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.executescript(SCHEMA)
    return conn


def pack_blob(lst) -> bytes:
    return bytes(int(v) & 0xFF for v in lst)


def unpack_blob(blob: bytes) -> list:
    return list(blob)


def insert_records(conn: sqlite3.Connection, records: list, tag: str | None = None):
    rows = []
    for r in records:
        rows.append((
            tag or r.get("_src"),
            r.get("ply"),
            r.get("turn"),
            r.get("outcome"),
            r.get("d0"),
            r.get("d1"),
            r.get("eval"),
            r.get("pawn0"),
            r.get("pawn1"),
            r.get("wl0"),
            r.get("wl1"),
            r.get("corridor_width0"),
            r.get("corridor_width1"),
            pack_blob(r.get("d0_field", [])),
            pack_blob(r.get("d1_field", [])),
            pack_blob(r.get("delta0", [])),
            pack_blob(r.get("delta1", [])),
            pack_blob(r.get("hw", [])),
            pack_blob(r.get("vw", [])),
        ))
    conn.executemany(
        "INSERT INTO records "
        "(src,ply,turn,outcome,d0,d1,eval,pawn0,pawn1,wl0,wl1,cw0,cw1,"
        "d0_field,d1_field,delta0,delta1,hw,vw) VALUES "
        "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()


def load_records_from_db(path: Path) -> list:
    """Load all records from DB as dicts (for train.py compatibility)."""
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    cur = conn.execute(
        "SELECT src,ply,turn,outcome,d0,d1,eval,pawn0,pawn1,wl0,wl1,"
        "cw0,cw1,d0_field,d1_field,delta0,delta1,hw,vw FROM records"
    )
    out = []
    for row in cur:
        out.append({
            "_src":            row["src"],
            "ply":             row["ply"],
            "turn":            row["turn"],
            "outcome":         row["outcome"],
            "d0":              row["d0"],
            "d1":              row["d1"],
            "eval":            row["eval"],
            "pawn0":           row["pawn0"],
            "pawn1":           row["pawn1"],
            "wl0":             row["wl0"],
            "wl1":             row["wl1"],
            "corridor_width0": row["cw0"],
            "corridor_width1": row["cw1"],
            "d0_field":        unpack_blob(row["d0_field"]),
            "d1_field":        unpack_blob(row["d1_field"]),
            "delta0":          unpack_blob(row["delta0"]),
            "delta1":          unpack_blob(row["delta1"]),
            "hw":              unpack_blob(row["hw"]),
            "vw":              unpack_blob(row["vw"]),
        })
    conn.close()
    return out


# ── Engine helpers ────────────────────────────────────────────────────────────

def run_match(engine, games, time_s, openings):
    """Run a self-play match and return raw stdout lines."""
    cmd = [
        str(BIN), "match",
        "--a", engine, "--b", engine,
        "--games", str(games),
        "--time", str(time_s),
        "--dump-games",
    ]
    if openings == "book":
        cmd += ["--openings", "book"]
    result = subprocess.run(cmd, capture_output=True, check=True)
    return result.stdout.decode("utf-8", errors="replace").splitlines()


def eval_batch(all_move_lists):
    """Feed all move sequences to titanium eval-batch in one subprocess call."""
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
    d0f = rec.get("d0_field", [])
    d1f = rec.get("d1_field", [])
    d0  = rec["d0"]
    d1  = rec["d1"]

    def delta_field(bfs_field, shortest):
        return [min(255, max(0, int(v) - shortest)) for v in bfs_field]

    return {
        "delta0":          delta_field(d0f, d0),
        "delta1":          delta_field(d1f, d1),
        "corridor_width0": sum(1 for d in d0f if int(d) == d0),
        "corridor_width1": sum(1 for d in d1f if int(d) == d1),
    }


def games_to_records(games, min_ply, max_ply, sample_rate):
    """Convert (move_list, outcome) games to training records via one eval-batch call."""
    entries = []
    for move_list, outcome in games:
        for ply in range(min_ply, min(max_ply + 1, len(move_list) + 1)):
            if sample_rate < 1.0 and random.random() > sample_rate:
                continue
            entries.append((ply, move_list[:ply], outcome))

    if not entries:
        return []

    evals = eval_batch([e[1] for e in entries])

    records = []
    for (ply, _, outcome), rec in zip(entries, evals):
        rec.update(compute_geometry(rec))
        rec["outcome"] = outcome
        rec["ply"] = ply
        records.append(rec)
    return records


# ── Incremental ingest ────────────────────────────────────────────────────────

def offset_path_for(src: Path) -> Path:
    return src.with_suffix(src.suffix + ".ingested_offset")


def ingest_incremental(
    src_path: Path,
    out_path: Path,
    min_ply: int = 4,
    max_ply: int = 150,
    sample_rate: float = 1.0,
    tag: str | None = None,
) -> int:
    """Append only new GAME/RESULT pairs from src_path into out_path (SQLite).

    Tracks byte offset in <src>.ingested_offset so calling after each game
    is safe: only new bytes are processed and no records are duplicated.
    """
    src_path = Path(src_path)
    out_path = Path(out_path)

    if not src_path.exists():
        return 0

    off_path = offset_path_for(src_path)
    if off_path.exists():
        offset = int(off_path.read_text(encoding="utf-8").strip() or "0")
    else:
        # First call: assume everything already ingested (avoids duplicates on
        # re-deploy when a prior end-of-match pass wrote the full file).
        offset = src_path.stat().st_size

    with open(src_path, encoding="utf-8", errors="replace") as f:
        f.seek(offset)
        chunk = f.read()
        new_offset = f.tell()

    if not chunk.strip():
        return 0

    games = parse_dump_games(chunk.splitlines())
    if not games:
        off_path.write_text(str(new_offset), encoding="utf-8")
        return 0

    records = games_to_records(games, min_ply, max_ply, sample_rate)
    conn = open_db(out_path)
    insert_records(conn, records, tag=tag)
    conn.close()

    off_path.write_text(str(new_offset), encoding="utf-8")
    return len(records)


# ── Migration ─────────────────────────────────────────────────────────────────

def migrate_jsonl_to_db(jsonl_path: Path, db_path: Path):
    """One-time migration: read a JSONL file and insert all records into a DB."""
    jsonl_path = Path(jsonl_path)
    db_path = Path(db_path)
    lines = [l for l in jsonl_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    records = [json.loads(l) for l in lines]
    conn = open_db(db_path)
    insert_records(conn, records)
    conn.close()
    print(f"Migrated {len(records)} records: {jsonl_path} -> {db_path}")
    size_before = jsonl_path.stat().st_size
    size_after  = db_path.stat().st_size
    print(f"  {size_before/1024:.0f} KB JSONL -> {size_after/1024:.0f} KB SQLite  ({size_before/max(size_after,1):.1f}x smaller)")


# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_dump_games(lines):
    """Parse GAME/RESULT lines into [(move_list, outcome)] tuples."""
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
                continue
            games.append((moves, 1 if r == "W" else -1))
            i += 2
        else:
            i += 1
    return games


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games",         type=int,   default=200)
    ap.add_argument("--time",          type=float, default=0.1)
    ap.add_argument("--engine",        default="titanium-v15")
    ap.add_argument("--out",           default=str(DB_PATH))
    ap.add_argument("--min-ply",       type=int,   default=4)
    ap.add_argument("--max-ply",       type=int,   default=150)
    ap.add_argument("--sample-rate",   type=float, default=1.0)
    ap.add_argument("--openings",      default="random", choices=["random", "book"])
    ap.add_argument("--from-file",     default=None, metavar="PATH",
                    help="Read GAME/RESULT lines from file instead of running a match.")
    ap.add_argument("--incremental",   default=None, metavar="PATH",
                    help="Ingest only new bytes from PATH (byte-offset sidecar).")
    ap.add_argument("--tag",           default=None,
                    help="Source tag stored in the src column.")
    ap.add_argument("--migrate-jsonl", default=None, metavar="PATH",
                    help="Migrate a JSONL file into the DB, then exit.")
    ap.add_argument("--append",        action="store_true",
                    help="Ignored (DB always appends); kept for backwards compat.")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── one-time migration ──
    if args.migrate_jsonl:
        migrate_jsonl_to_db(Path(args.migrate_jsonl), out_path)
        sys.exit(0)

    # ── per-game incremental ingest ──
    if args.incremental:
        n = ingest_incremental(
            Path(args.incremental), out_path,
            args.min_ply, args.max_ply, args.sample_rate, tag=args.tag,
        )
        if n:
            print(f"Incremental: +{n} records -> {out_path.name}")
        sys.exit(0)

    # ── bulk ingest from file ──
    if args.from_file:
        src = Path(args.from_file)
        if not src.exists():
            print(f"ERROR: --from-file path not found: {src}")
            sys.exit(1)
        games = parse_dump_games(src.read_text(encoding="utf-8").splitlines())
        if not games:
            print("No games found in file.")
            sys.exit(1)
        print(f"Ingesting {len(games)} games from {src} ...")
        records = games_to_records(games, args.min_ply, args.max_ply, args.sample_rate)
    else:
        # ── run self-play match ──
        print(f"Generating {args.games} games @ {args.time}s/move with {args.engine}...")
        try:
            lines = run_match(args.engine, args.games, args.time, args.openings)
        except subprocess.CalledProcessError:
            print("ERROR: titanium match --dump-games not yet supported.")
            sys.exit(1)
        games = parse_dump_games(lines)
        if not games:
            print("No games parsed. Is --dump-games implemented in the engine?")
            sys.exit(1)
        print(f"  {len(games)} games parsed; running eval-batch on all positions...")
        records = games_to_records(games, args.min_ply, args.max_ply, args.sample_rate)

    conn = open_db(out_path)
    insert_records(conn, records, tag=args.tag)
    conn.close()
    print(f"Done: {len(records)} records -> {out_path}")


if __name__ == "__main__":
    main()
