# Canonical Position Store Runbook

This is the durable training-data store for Titanium v15.

It replaces repeated notation-string storage with:

- canonical unique positions
- stable 1-byte move codes
- parent/move/child graph edges
- compact per-game move paths
- versioned labels
- append-only binary self-play shards

Important truth first: the global structure is a **canonical position graph**, not a strict DAG. Full Quoridor states can repeat through pawn backtracking even when no walls change.

## What Is Stored

- `positions`: one packed canonical state per unique reachable position
- `edges`: `parent_position_id + move_code_u8 -> child_position_id`
- `games`: compact replayable game records
- `game_paths`: one byte per move, stored as a BLOB
- `labels`: versioned value / teacher / pressure / counterfactual labels
- `observations`: repeated occurrence counts and result totals
- `imports`: idempotent migration/import bookkeeping
- `relabel_queue`: future labeling work

## Move Encoding

Stable `u8` alphabet:

- `0..63`: horizontal wall slots
- `64..127`: vertical wall slots
- `128..135`: pawn destination directions
- `136..255`: reserved and rejected

Coordinate orientation matches the engine and website board logic:

- origin: `a1`
- `x` increases from `a -> i`
- `y` increases from `1 -> 9`
- wall slot `0` is `a1h`
- wall slot `64` is `a1v`

Pawn codes are destination-direction classes, not “step kinds”:

- `128`: north
- `129`: south
- `130`: east
- `131`: west
- `132`: northeast
- `133`: northwest
- `134`: southeast
- `135`: southwest

The current board state disambiguates single-step, straight jump, and diagonal jump.

## Legality Authority

Do **not** trust detached website movegen for migration validation.

Importers validate by replaying moves against the canonical state rules in `training/position_store_state.py`. If a source row is illegal under those rules, it is rejected and logged. We currently see a small family of legacy rows with impossible black move `d8` after `e2`; those are preserved as rejects rather than silently reinterpreted.

## Files

- CLI: `training/position_store.py`
- DB/schema/import logic: `training/position_store_lib.py`
- Canonical state + move codec: `training/position_store_state.py`
- Tests: `training/test_position_store.py`

## First-Time Setup

Run from repository root:

```powershell
Set-Location "C:\gitProjects\Quoridor best AI"
python training\position_store.py init
python training\position_store.py inventory
```

The inventory command writes timestamped JSON + Markdown reports under:

```text
training/data/position_store_reports/
```

## Safe Migration Order

Do not import everything blindly on the first pass.

1. Create the destination database.
2. Run dry-runs on each source family.
3. Inspect rejects.
4. Perform the real import into a fresh destination database.
5. Run `audit`.
6. Run `storage-report`.

Recommended commands:

```powershell
python training\position_store.py init

python training\position_store.py import-games training\data\all_games.db --dry-run
python training\position_store.py import-positions training\data\search_pressure.jsonl --dry-run
python training\position_store.py import-positions training\data\zero_teacher\labels\search_budget.jsonl --dry-run
python training\position_store.py import-positions "KaAiData\ANOTHER TRAINING DAT ASTUFF SUPER USEFULL\selfplay_iters_000001_000020\iter_000001\shard_000.jsonl" --dry-run
```

Real smoke migration into a separate database:

```powershell
python training\position_store.py --db training\data\position_graph_smoke.db init
python training\position_store.py --db training\data\position_graph_smoke.db import-games training\data\all_games.db
python training\position_store.py --db training\data\position_graph_smoke.db import-positions training\data\search_pressure.jsonl
python training\position_store.py --db training\data\position_graph_smoke.db import-positions training\data\zero_teacher\labels\search_budget.jsonl
python training\position_store.py --db training\data\position_graph_smoke.db import-positions "KaAiData\ANOTHER TRAINING DAT ASTUFF SUPER USEFULL\selfplay_iters_000001_000020\iter_000001\shard_000.jsonl"
python training\position_store.py --db training\data\position_graph_smoke.db audit
python training\position_store.py --db training\data\position_graph_smoke.db storage-report training\data\all_games.db training\data\search_pressure.jsonl training\data\zero_teacher\labels\search_budget.jsonl "KaAiData\ANOTHER TRAINING DAT ASTUFF SUPER USEFULL\selfplay_iters_000001_000020\iter_000001\shard_000.jsonl"
```

## Binary Self-Play Shards

Workers should not write directly into SQLite per move.

Write completed shard files as:

- `*.partial` while writing
- `*.ready` after atomic completion

Import them with:

```powershell
python training\position_store.py ingest-shards training\data\selfplay_shards
```

The importer ignores partial files and marks completed imports by renaming shard files to `.imported`.

## Important SQLite Rule

Do **not** run multiple imports concurrently into the same SQLite database file.

SQLite will lock, and that is expected. Run imports sequentially per database file.

## Current Smoke Results

Validated locally on `2026-06-19`:

- `training/test_position_store.py`: `17 passed`
- legacy `all_games.db` dry-run:
  - `1499` rows seen
  - `1484` accepted
  - `15` rejected
- `search_pressure.jsonl` dry-run:
  - `4999` rows seen
  - `4964` accepted
  - `35` rejected
- `zero_teacher/labels/search_budget.jsonl` dry-run:
  - `203` rows seen
  - `202` accepted
  - `1` rejected
- friend shard `iter_000001/shard_000.jsonl` dry-run:
  - `112780` rows seen
  - `112780` accepted
  - `0` rejected

Repeated reject signature so far:

- illegal black move `d8` from a standard-start position after `e2`

Those rows are treated as source-data errors, not remapped into something “close enough.”

## Storage Notes

Measured smoke import:

- source bytes:
  - `all_games.db`: `286,720`
  - `search_pressure.jsonl`: `7,440,172`
  - `zero search budget`: `859,486`
  - friend shard: `66,564,982`
  - total: `75,151,360`
- resulting smoke DB bytes: `140,423,168`

Why the DB is still larger than the raw sources in this mixed smoke:

- labels carry versioned metadata and indexes
- imported teacher payloads are stored in compact JSON, but still not free
- SQLite page/index overhead is real

What *is* compact already:

- canonical packed state is fixed-width (`24` bytes)
- game paths are exactly one byte per move
- repeated positions are merged

Measured legacy game dedup:

- replayed positions from accepted games: `90,684`
- unique canonical positions from those games: `42,095`
- dedup ratio: about `2.15x`

So the graph model is buying real position reuse already, even before future payload compaction work.

## Export For Training

To export compatible labeled rows:

```powershell
python training\position_store.py export-training training\data\position_training_export.jsonl --label-type teacher_value
```

This writes packed state hex plus compatible label metadata without replaying every game from root at export time.

## Recommended Next Steps

1. Keep using this store for canonical game + position migration.
2. If friend/teacher datasets become a major long-term storage driver, move policy payloads from JSON into compact binary side tables.
3. Add an engine-vs-website legality parity suite before trusting website-produced move traces as authoritative.
