// migrate_benchmark.rs — SQLite ➜ local native Turso migration & benchmark tool.
//
// Uses the `turso` crate (Rust-native Turso DB engine) exclusively.
// `libsql` is NOT used here.  `rusqlite` is used only on the source/read side
// because it is the correct tool for reading existing SQLite WAL stores.

use anyhow::{Context, Result};
use clap::Parser;
use rusqlite::Connection as SqliteConnection;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;
use turso::{Builder, Connection as TursoConnection, Error as TursoError};
use turso::params::Params as TursoParams;
use turso::value::Value as TursoValue;

// ---------------------------------------------------------------------------
// Retry helper for MVCC concurrent write conflicts.
// ---------------------------------------------------------------------------

fn is_retryable_turso_error(e: &TursoError) -> bool {
    matches!(e, TursoError::Busy(_) | TursoError::BusySnapshot(_))
        || matches!(e, TursoError::Error(msg) if msg.contains("conflict"))
}

// ---------------------------------------------------------------------------
// Query helpers
// ---------------------------------------------------------------------------

async fn query_row_count(conn: &TursoConnection, sql: &str) -> Result<i64> {
    let mut rows = conn.query(sql, ()).await.context("count query failed")?;
    match rows.next().await? {
        Some(row) => {
            let val: i64 = row.get(0)?;
            Ok(val)
        }
        None => Ok(0),
    }
}

async fn query_max_position_id(conn: &TursoConnection) -> Result<i64> {
    let mut rows = conn
        .query("SELECT MAX(position_id) FROM positions", ())
        .await
        .context("max position_id query failed")?;
    match rows.next().await? {
        Some(row) => {
            let val: Option<i64> = row.get(0).ok();
            Ok(val.unwrap_or(0))
        }
        None => Ok(0),
    }
}

async fn query_dest_row(conn: &TursoConnection, id: i64) -> Option<(Vec<u8>, Vec<u8>)> {
    let params = TursoParams::Positional(vec![TursoValue::Integer(id)]);
    let mut rows = conn
        .query(
            "SELECT canonical_hash, packed_state FROM positions WHERE position_id = ?",
            params,
        )
        .await
        .ok()?;
    let row = rows.next().await.ok()??;
    let h: Vec<u8> = row.get(0).ok()?;
    let p: Vec<u8> = row.get(1).ok()?;
    Some((h, p))
}

// ---------------------------------------------------------------------------
// CLI args
// ---------------------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(
    name = "migrate_benchmark",
    about = "Migration & Benchmarking tool — SQLite canonical store ➜ local native Turso DB"
)]
struct Args {
    #[arg(
        long,
        default_value = "c:\\gitProjects\\Quoridor best AI\\training\\data\\canonical\\game_store.db"
    )]
    source: PathBuf,

    #[arg(long, default_value = "local_turso_dest.db")]
    dest: PathBuf,

    #[arg(long)]
    dry_run: bool,

    #[arg(long)]
    resume: bool,

    #[arg(long)]
    verify_only: bool,

    #[arg(long, default_value_t = 1000)]
    batch_size: usize,

    #[arg(long, default_value_t = 1)]
    workers: usize,

    #[arg(long)]
    limit_positions: Option<usize>,

    /// Run concurrent-writer benchmark (1/2/4/8 writers × 50 k positions each)
    #[arg(long)]
    run_benchmark: bool,

    /// Run idempotent recovery smoke-test
    #[arg(long)]
    run_recovery: bool,
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    println!("migrate_benchmark config: {:?}", args);

    if args.verify_only {
        println!("--- Phase 5: Verification Only ---");
        verify_equality(&args.source, &args.dest, args.limit_positions).await?;
        return Ok(());
    }

    if args.run_benchmark {
        println!("--- Phase 6 & 7: Concurrent-Writer Benchmarking ---");
        run_benchmarks(&args.source, &args.dest, args.batch_size, args.workers).await?;
        return Ok(());
    }

    if args.run_recovery {
        println!("--- Phase 8: Recovery & Idempotence ---");
        run_recovery_test(&args.source, &args.dest, args.batch_size).await?;
        return Ok(());
    }

    println!("--- Phase 3: Initializing target Turso Schema ---");
    init_turso_schema(&args.dest).await?;

    println!("--- Phase 4: Migration Starting ---");
    let start_time = Instant::now();
    migrate_database(&args.source, &args.dest, &args).await?;
    println!("Migration completed in {:?}", start_time.elapsed());

    println!("--- Phase 5: Verifying Migration Equality ---");
    verify_equality(&args.source, &args.dest, args.limit_positions).await?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Schema initialisation (native Turso)
// ---------------------------------------------------------------------------

async fn init_turso_schema(dest: &Path) -> Result<()> {
    let db = Builder::new_local(dest.to_str().unwrap())
        .build()
        .await
        .context("Failed to open/create Turso database")?;
    let conn = db.connect().context("Failed to open connection")?;

    let schema_statements = vec![
        "CREATE TABLE IF NOT EXISTS positions (
            position_id INTEGER PRIMARY KEY,
            canonical_hash BLOB NOT NULL,
            fast_hash INTEGER NOT NULL,
            packed_state BLOB NOT NULL,
            side_to_move INTEGER NOT NULL,
            ply_min_seen INTEGER,
            ply_max_seen INTEGER,
            first_seen_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL,
            total_visits INTEGER NOT NULL DEFAULT 0,
            source_flags INTEGER NOT NULL DEFAULT 0,
            schema_version INTEGER NOT NULL,
            UNIQUE(canonical_hash, packed_state)
        )",
        "CREATE TABLE IF NOT EXISTS edges (
            parent_position_id INTEGER NOT NULL REFERENCES positions(position_id),
            move_code_u8 INTEGER NOT NULL,
            child_position_id INTEGER NOT NULL REFERENCES positions(position_id),
            visit_count INTEGER NOT NULL DEFAULT 0,
            p0_win_count INTEGER NOT NULL DEFAULT 0,
            p1_win_count INTEGER NOT NULL DEFAULT 0,
            draw_count INTEGER NOT NULL DEFAULT 0,
            first_seen_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL,
            UNIQUE(parent_position_id, move_code_u8, child_position_id)
        )",
        "CREATE TABLE IF NOT EXISTS games (
            game_id INTEGER PRIMARY KEY,
            start_position_id INTEGER NOT NULL REFERENCES positions(position_id),
            result INTEGER,
            move_count INTEGER NOT NULL,
            generator_engine_hash TEXT,
            generator_trunk_hash TEXT,
            search_config_hash TEXT,
            random_seed TEXT,
            worker_id TEXT,
            source TEXT NOT NULL,
            created_at TEXT NOT NULL,
            game_metadata TEXT
        )",
        "CREATE TABLE IF NOT EXISTS game_paths (
            game_id INTEGER PRIMARY KEY REFERENCES games(game_id),
            packed_u8_move_sequence BLOB NOT NULL
        )",
        "CREATE TABLE IF NOT EXISTS labels (
            label_id INTEGER PRIMARY KEY,
            position_id INTEGER NOT NULL REFERENCES positions(position_id),
            label_type TEXT NOT NULL,
            value REAL,
            score REAL,
            bound TEXT,
            best_move_u8 INTEGER,
            nodes INTEGER,
            completed_depth INTEGER,
            selective_depth INTEGER,
            is_proven INTEGER NOT NULL DEFAULT 0,
            engine_hash TEXT,
            trunk_hash TEXT,
            search_config_hash TEXT,
            label_schema_version INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            quality_rank INTEGER NOT NULL DEFAULT 0,
            source TEXT NOT NULL,
            payload_json TEXT
        )",
        "CREATE TABLE IF NOT EXISTS observations (
            position_id INTEGER NOT NULL REFERENCES positions(position_id),
            source_cohort TEXT NOT NULL,
            visit_count INTEGER NOT NULL DEFAULT 0,
            p0_wins INTEGER NOT NULL DEFAULT 0,
            p1_wins INTEGER NOT NULL DEFAULT 0,
            draws INTEGER NOT NULL DEFAULT 0,
            evaluation_summary TEXT,
            first_seen TEXT NOT NULL,
            last_seen TEXT NOT NULL,
            UNIQUE(position_id, source_cohort)
        )",
        "CREATE TABLE IF NOT EXISTS relabel_queue (
            queue_id INTEGER PRIMARY KEY,
            position_id INTEGER NOT NULL REFERENCES positions(position_id),
            requested_label_type TEXT NOT NULL,
            requested_node_budget INTEGER,
            priority INTEGER NOT NULL DEFAULT 0,
            reason TEXT NOT NULL,
            required_engine_hash TEXT,
            required_trunk_hash TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            attempt_count INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )",
        "CREATE TABLE IF NOT EXISTS imports (
            import_id INTEGER PRIMARY KEY,
            source_path TEXT NOT NULL,
            source_hash TEXT NOT NULL,
            format TEXT NOT NULL,
            record_count INTEGER NOT NULL DEFAULT 0,
            accepted_count INTEGER NOT NULL DEFAULT 0,
            rejected_count INTEGER NOT NULL DEFAULT 0,
            duplicate_count INTEGER NOT NULL DEFAULT 0,
            started_at TEXT NOT NULL,
            completed_at TEXT,
            importer_version TEXT NOT NULL,
            status TEXT NOT NULL,
            error_report_path TEXT,
            UNIQUE(source_hash, format)
        )",
        "CREATE TABLE IF NOT EXISTS store_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )",
        "CREATE INDEX IF NOT EXISTS idx_positions_fast_hash ON positions(fast_hash)",
        "CREATE INDEX IF NOT EXISTS idx_labels_position_type ON labels(position_id, label_type, trunk_hash, engine_hash)",
        "CREATE INDEX IF NOT EXISTS idx_observations_source ON observations(source_cohort, visit_count DESC)",
    ];

    for stmt in schema_statements {
        conn.execute(stmt, ())
            .await
            .with_context(|| format!("Failed to execute schema stmt: {}", &stmt[..64.min(stmt.len())]))?;
    }

    println!("Schema initialised in {:?}", dest);
    Ok(())
}

// ---------------------------------------------------------------------------
// Migration (rusqlite source → Turso dest)
// ---------------------------------------------------------------------------

async fn migrate_database(source: &Path, dest: &Path, args: &Args) -> Result<()> {
    if args.dry_run {
        println!("Dry run mode enabled. No data will be written.");
        return Ok(());
    }

    let src_conn = SqliteConnection::open(source).context("Cannot open source SQLite")?;
    let db = Builder::new_local(dest.to_str().unwrap())
        .build()
        .await
        .context("Cannot open destination Turso DB")?;
    let dest_conn = db.connect().context("Cannot connect to Turso DB")?;

    // -- store_metadata -------------------------------------------------------
    {
        println!("Migrating store_metadata…");
        let mut stmt = src_conn.prepare("SELECT key, value FROM store_metadata")?;
        let rows: Vec<(String, String)> = stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
            .collect::<rusqlite::Result<_>>()?;

        let tx = dest_conn.transaction().await?;
        for (k, v) in rows {
            let params = TursoParams::Positional(vec![
                TursoValue::Text(k),
                TursoValue::Text(v),
            ]);
            tx.execute("INSERT OR REPLACE INTO store_metadata (key, value) VALUES (?, ?)", params)
                .await?;
        }
        tx.commit().await?;
    }

    // -- imports -------------------------------------------------------
    {
        println!("Migrating imports…");
        let mut stmt = src_conn.prepare(
            "SELECT import_id, source_path, source_hash, format, record_count, \
             accepted_count, rejected_count, duplicate_count, started_at, \
             completed_at, importer_version, status, error_report_path FROM imports",
        )?;
        type ImportRow = (
            Option<i64>, String, String, String, i64, i64, i64, i64,
            String, Option<String>, String, String, Option<String>,
        );
        let rows: Vec<ImportRow> = stmt
            .query_map([], |row| {
                Ok((
                    row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?,
                    row.get(4)?, row.get(5)?, row.get(6)?, row.get(7)?,
                    row.get(8)?, row.get(9)?, row.get(10)?, row.get(11)?,
                    row.get(12)?,
                ))
            })?
            .collect::<rusqlite::Result<_>>()?;

        let tx = dest_conn.transaction().await?;
        for r in rows {
            let params = TursoParams::Positional(vec![
                r.0.map(TursoValue::Integer).unwrap_or(TursoValue::Null),
                TursoValue::Text(r.1),
                TursoValue::Text(r.2),
                TursoValue::Text(r.3),
                TursoValue::Integer(r.4),
                TursoValue::Integer(r.5),
                TursoValue::Integer(r.6),
                TursoValue::Integer(r.7),
                TursoValue::Text(r.8),
                r.9.map(TursoValue::Text).unwrap_or(TursoValue::Null),
                TursoValue::Text(r.10),
                TursoValue::Text(r.11),
                r.12.map(TursoValue::Text).unwrap_or(TursoValue::Null),
            ]);
            tx.execute(
                "INSERT OR REPLACE INTO imports \
                 (import_id, source_path, source_hash, format, record_count, \
                  accepted_count, rejected_count, duplicate_count, started_at, \
                  completed_at, importer_version, status, error_report_path) \
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                params,
            )
            .await?;
        }
        tx.commit().await?;
    }

    // -- positions -------------------------------------------------------
    {
        println!("Migrating positions…");
        let start_pos_id: i64 = if args.resume {
            query_max_position_id(&dest_conn).await.unwrap_or(0)
        } else {
            0
        };
        let limit_clause = match args.limit_positions {
            Some(n) => format!("LIMIT {}", n),
            None => String::new(),
        };
        let query = format!(
            "SELECT position_id, canonical_hash, fast_hash, packed_state, side_to_move, \
             ply_min_seen, ply_max_seen, first_seen_at, last_seen_at, total_visits, \
             source_flags, schema_version \
             FROM positions WHERE position_id > {} ORDER BY position_id {}",
            start_pos_id, limit_clause
        );

        type PosRow = (i64, Vec<u8>, i64, Vec<u8>, i64, Option<i64>, Option<i64>, String, String, i64, i64, i64);
        let mut stmt = src_conn.prepare(&query)?;
        let rows: Vec<PosRow> = stmt
            .query_map([], |row| {
                Ok((
                    row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?,
                    row.get(4)?, row.get(5)?, row.get(6)?, row.get(7)?,
                    row.get(8)?, row.get(9)?, row.get(10)?, row.get(11)?,
                ))
            })?
            .collect::<rusqlite::Result<_>>()?;

        let mut total_migrated: usize = 0;
        for chunk in rows.chunks(args.batch_size) {
            let tx = dest_conn.transaction().await?;
            for r in chunk {
                let params = TursoParams::Positional(vec![
                    TursoValue::Integer(r.0),
                    TursoValue::Blob(r.1.clone()),
                    TursoValue::Integer(r.2),
                    TursoValue::Blob(r.3.clone()),
                    TursoValue::Integer(r.4),
                    r.5.map(TursoValue::Integer).unwrap_or(TursoValue::Null),
                    r.6.map(TursoValue::Integer).unwrap_or(TursoValue::Null),
                    TursoValue::Text(r.7.clone()),
                    TursoValue::Text(r.8.clone()),
                    TursoValue::Integer(r.9),
                    TursoValue::Integer(r.10),
                    TursoValue::Integer(r.11),
                ]);
                tx.execute(
                    "INSERT OR REPLACE INTO positions \
                     (position_id, canonical_hash, fast_hash, packed_state, side_to_move, \
                      ply_min_seen, ply_max_seen, first_seen_at, last_seen_at, \
                      total_visits, source_flags, schema_version) \
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    params,
                )
                .await?;
            }
            tx.commit().await?;
            total_migrated += chunk.len();
            println!("  … {} positions migrated", total_migrated);
        }
        println!("Positions: {} rows migrated total.", total_migrated);
    }

    // -- edges -------------------------------------------------------
    {
        println!("Migrating edges…");
        type EdgeRow = (i64, i64, i64, i64, i64, i64, i64, String, String);
        let mut stmt = src_conn.prepare(
            "SELECT parent_position_id, move_code_u8, child_position_id, \
             visit_count, p0_win_count, p1_win_count, draw_count, \
             first_seen_at, last_seen_at FROM edges",
        )?;
        let rows: Vec<EdgeRow> = stmt
            .query_map([], |row| {
                Ok((
                    row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?,
                    row.get(4)?, row.get(5)?, row.get(6)?, row.get(7)?,
                    row.get(8)?,
                ))
            })?
            .collect::<rusqlite::Result<_>>()?;

        for chunk in rows.chunks(args.batch_size) {
            let tx = dest_conn.transaction().await?;
            for r in chunk {
                let params = TursoParams::Positional(vec![
                    TursoValue::Integer(r.0),
                    TursoValue::Integer(r.1),
                    TursoValue::Integer(r.2),
                    TursoValue::Integer(r.3),
                    TursoValue::Integer(r.4),
                    TursoValue::Integer(r.5),
                    TursoValue::Integer(r.6),
                    TursoValue::Text(r.7.clone()),
                    TursoValue::Text(r.8.clone()),
                ]);
                tx.execute(
                    "INSERT OR REPLACE INTO edges \
                     (parent_position_id, move_code_u8, child_position_id, \
                      visit_count, p0_win_count, p1_win_count, draw_count, \
                      first_seen_at, last_seen_at) \
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    params,
                )
                .await?;
            }
            tx.commit().await?;
        }
        println!("Edges migrated.");
    }

    // -- games + game_paths -------------------------------------------------------
    {
        println!("Migrating games & game_paths…");
        type GameRow = (i64, i64, Option<i64>, i64, Option<String>, Option<String>, Option<String>, Option<String>, Option<String>, String, String, Option<String>);
        let mut stmt = src_conn.prepare(
            "SELECT game_id, start_position_id, result, move_count, \
             generator_engine_hash, generator_trunk_hash, search_config_hash, \
             random_seed, worker_id, source, created_at, game_metadata FROM games",
        )?;
        let rows: Vec<GameRow> = stmt
            .query_map([], |row| {
                Ok((
                    row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?,
                    row.get(4)?, row.get(5)?, row.get(6)?, row.get(7)?,
                    row.get(8)?, row.get(9)?, row.get(10)?, row.get(11)?,
                ))
            })?
            .collect::<rusqlite::Result<_>>()?;

        let tx = dest_conn.transaction().await?;
        for r in &rows {
            let params = TursoParams::Positional(vec![
                TursoValue::Integer(r.0),
                TursoValue::Integer(r.1),
                r.2.map(TursoValue::Integer).unwrap_or(TursoValue::Null),
                TursoValue::Integer(r.3),
                r.4.clone().map(TursoValue::Text).unwrap_or(TursoValue::Null),
                r.5.clone().map(TursoValue::Text).unwrap_or(TursoValue::Null),
                r.6.clone().map(TursoValue::Text).unwrap_or(TursoValue::Null),
                r.7.clone().map(TursoValue::Text).unwrap_or(TursoValue::Null),
                r.8.clone().map(TursoValue::Text).unwrap_or(TursoValue::Null),
                TursoValue::Text(r.9.clone()),
                TursoValue::Text(r.10.clone()),
                r.11.clone().map(TursoValue::Text).unwrap_or(TursoValue::Null),
            ]);
            tx.execute(
                "INSERT OR REPLACE INTO games \
                 (game_id, start_position_id, result, move_count, \
                  generator_engine_hash, generator_trunk_hash, search_config_hash, \
                  random_seed, worker_id, source, created_at, game_metadata) \
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                params,
            )
            .await?;
        }
        tx.commit().await?;

        let mut stmt = src_conn.prepare("SELECT game_id, packed_u8_move_sequence FROM game_paths")?;
        let path_rows: Vec<(i64, Vec<u8>)> = stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
            .collect::<rusqlite::Result<_>>()?;

        let tx = dest_conn.transaction().await?;
        for (id, seq) in path_rows {
            let params = TursoParams::Positional(vec![
                TursoValue::Integer(id),
                TursoValue::Blob(seq),
            ]);
            tx.execute(
                "INSERT OR REPLACE INTO game_paths (game_id, packed_u8_move_sequence) VALUES (?, ?)",
                params,
            )
            .await?;
        }
        tx.commit().await?;
        println!("Games migrated.");
    }

    // -- observations -------------------------------------------------------
    {
        println!("Migrating observations…");
        type ObsRow = (i64, String, i64, i64, i64, i64, Option<String>, String, String);
        let mut stmt = src_conn.prepare(
            "SELECT position_id, source_cohort, visit_count, p0_wins, p1_wins, \
             draws, evaluation_summary, first_seen, last_seen FROM observations",
        )?;
        let rows: Vec<ObsRow> = stmt
            .query_map([], |row| {
                Ok((
                    row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?,
                    row.get(4)?, row.get(5)?, row.get(6)?, row.get(7)?,
                    row.get(8)?,
                ))
            })?
            .collect::<rusqlite::Result<_>>()?;

        for chunk in rows.chunks(args.batch_size) {
            let tx = dest_conn.transaction().await?;
            for r in chunk {
                let params = TursoParams::Positional(vec![
                    TursoValue::Integer(r.0),
                    TursoValue::Text(r.1.clone()),
                    TursoValue::Integer(r.2),
                    TursoValue::Integer(r.3),
                    TursoValue::Integer(r.4),
                    TursoValue::Integer(r.5),
                    r.6.clone().map(TursoValue::Text).unwrap_or(TursoValue::Null),
                    TursoValue::Text(r.7.clone()),
                    TursoValue::Text(r.8.clone()),
                ]);
                tx.execute(
                    "INSERT OR REPLACE INTO observations \
                     (position_id, source_cohort, visit_count, p0_wins, p1_wins, \
                      draws, evaluation_summary, first_seen, last_seen) \
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    params,
                )
                .await?;
            }
            tx.commit().await?;
        }
        println!("Observations migrated.");
    }

    // -- labels -------------------------------------------------------
    {
        println!("Migrating labels…");
        #[allow(clippy::type_complexity)]
        type LabelRow = (
            i64, i64, String, Option<f64>, Option<f64>, Option<String>,
            Option<i64>, Option<i64>, Option<i64>, Option<i64>, i64,
            Option<String>, Option<String>, Option<String>, i64, String, i64, String, Option<String>,
        );
        let mut stmt = src_conn.prepare(
            "SELECT label_id, position_id, label_type, value, score, bound, \
             best_move_u8, nodes, completed_depth, selective_depth, is_proven, \
             engine_hash, trunk_hash, search_config_hash, label_schema_version, \
             created_at, quality_rank, source, payload_json FROM labels",
        )?;
        let rows: Vec<LabelRow> = stmt
            .query_map([], |row| {
                Ok((
                    row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?,
                    row.get(4)?, row.get(5)?, row.get(6)?, row.get(7)?,
                    row.get(8)?, row.get(9)?, row.get(10)?, row.get(11)?,
                    row.get(12)?, row.get(13)?, row.get(14)?, row.get(15)?,
                    row.get(16)?, row.get(17)?, row.get(18)?,
                ))
            })?
            .collect::<rusqlite::Result<_>>()?;

        for chunk in rows.chunks(args.batch_size) {
            let tx = dest_conn.transaction().await?;
            for r in chunk {
                let params = TursoParams::Positional(vec![
                    TursoValue::Integer(r.0),
                    TursoValue::Integer(r.1),
                    TursoValue::Text(r.2.clone()),
                    r.3.map(TursoValue::Real).unwrap_or(TursoValue::Null),
                    r.4.map(TursoValue::Real).unwrap_or(TursoValue::Null),
                    r.5.clone().map(TursoValue::Text).unwrap_or(TursoValue::Null),
                    r.6.map(TursoValue::Integer).unwrap_or(TursoValue::Null),
                    r.7.map(TursoValue::Integer).unwrap_or(TursoValue::Null),
                    r.8.map(TursoValue::Integer).unwrap_or(TursoValue::Null),
                    r.9.map(TursoValue::Integer).unwrap_or(TursoValue::Null),
                    TursoValue::Integer(r.10),
                    r.11.clone().map(TursoValue::Text).unwrap_or(TursoValue::Null),
                    r.12.clone().map(TursoValue::Text).unwrap_or(TursoValue::Null),
                    r.13.clone().map(TursoValue::Text).unwrap_or(TursoValue::Null),
                    TursoValue::Integer(r.14),
                    TursoValue::Text(r.15.clone()),
                    TursoValue::Integer(r.16),
                    TursoValue::Text(r.17.clone()),
                    r.18.clone().map(TursoValue::Text).unwrap_or(TursoValue::Null),
                ]);
                tx.execute(
                    "INSERT OR REPLACE INTO labels \
                     (label_id, position_id, label_type, value, score, bound, \
                      best_move_u8, nodes, completed_depth, selective_depth, is_proven, \
                      engine_hash, trunk_hash, search_config_hash, label_schema_version, \
                      created_at, quality_rank, source, payload_json) \
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    params,
                )
                .await?;
            }
            tx.commit().await?;
        }
        println!("Labels migrated.");
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Equality verification
// ---------------------------------------------------------------------------

async fn verify_equality(source: &Path, dest: &Path, limit_positions: Option<usize>) -> Result<()> {
    let src_conn = SqliteConnection::open(source).context("Cannot open source SQLite")?;
    let db = Builder::new_local(dest.to_str().unwrap())
        .build()
        .await
        .context("Cannot open destination Turso DB")?;
    let dest_conn = db.connect().context("Cannot connect to Turso DB")?;

    let tables = [
        "store_metadata",
        "imports",
        "positions",
        "edges",
        "games",
        "game_paths",
        "observations",
        "labels",
    ];

    let mut all_ok = true;
    for table in &tables {
        let src_count: i64 = src_conn
            .query_row(&format!("SELECT COUNT(*) FROM {}", table), [], |r| r.get(0))
            .unwrap_or(0);
        let dest_count: i64 =
            query_row_count(&dest_conn, &format!("SELECT COUNT(*) FROM {}", table))
                .await
                .unwrap_or(0);

        let ok = limit_positions.is_some() || src_count == dest_count;
        println!(
            "  Table {:20}: SQLite={:8}  Turso={:8}  {}",
            table,
            src_count,
            dest_count,
            if ok { "OK" } else { "MISMATCH ❌" }
        );
        if !ok {
            all_ok = false;
        }
    }

    // Payload-level spot-check: first 1 000 positions
    println!("Payload spot-check: comparing first 1 000 positions by hash & blob…");
    let mut stmt = src_conn.prepare(
        "SELECT position_id, canonical_hash, packed_state FROM positions ORDER BY position_id LIMIT 1000",
    )?;
    let sample: Vec<(i64, Vec<u8>, Vec<u8>)> = stmt
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))?
        .collect::<rusqlite::Result<_>>()?;

    let mut mismatches = 0usize;
    for (id, canon_hash, packed) in &sample {
        match query_dest_row(&dest_conn, *id).await {
            Some((dest_hash, dest_packed)) => {
                if *canon_hash != dest_hash || *packed != dest_packed {
                    eprintln!("PAYLOAD MISMATCH at position_id={}", id);
                    mismatches += 1;
                }
            }
            None => {
                if limit_positions.is_none() {
                    eprintln!("MISSING position_id={} in destination!", id);
                    mismatches += 1;
                }
            }
        }
    }

    if mismatches > 0 {
        anyhow::bail!("{} payload mismatches detected", mismatches);
    }

    if !all_ok {
        anyhow::bail!("Row-count mismatches detected in one or more tables.");
    }

    println!("SUCCESS: All equality checks passed.");
    Ok(())
}

// ---------------------------------------------------------------------------
// Concurrent-writer benchmark  (native Turso MVCC, BEGIN CONCURRENT)
// ---------------------------------------------------------------------------

async fn run_benchmarks(source: &Path, dest: &Path, batch_size: usize, workers: usize) -> Result<()> {
    println!("=== Concurrent-writer benchmark ===");
    println!("Workers = {}, Batch size = {}", workers, batch_size);

    // Load 50 000 positions from SQLite source into memory
    let src_conn = SqliteConnection::open(source).context("Cannot open source SQLite")?;
    let mut stmt = src_conn.prepare(
        "SELECT position_id, canonical_hash, fast_hash, packed_state, side_to_move, \
         first_seen_at, last_seen_at, total_visits, schema_version \
         FROM positions LIMIT 50000",
    )?;
    type PosRow9 = (i64, Vec<u8>, i64, Vec<u8>, i64, String, String, i64, i64);
    let all_positions: Arc<Vec<PosRow9>> = Arc::new(
        stmt.query_map([], |row| {
            Ok((
                row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?,
                row.get(4)?, row.get(5)?, row.get(6)?, row.get(7)?,
                row.get(8)?,
            ))
        })?
        .collect::<rusqlite::Result<_>>()?,
    );
    println!("Loaded {} positions for benchmark.", all_positions.len());

    // Open Turso with MVCC
    let db = Arc::new(
        Builder::new_local(dest.to_str().unwrap())
            .build()
            .await
            .context("Cannot open destination Turso DB")?,
    );

    // Enable MVCC journal mode (needed for BEGIN CONCURRENT)
    let setup_conn = db.connect().context("Cannot connect to Turso DB")?;
    setup_conn
        .pragma_update("journal_mode", "'mvcc'")
        .await
        .context("Failed to set journal_mode=mvcc")?;
    drop(setup_conn);

    let (tx_chan, mut rx_chan) = mpsc::channel::<(usize, std::time::Duration)>(64);
    let start_time = Instant::now();
    let positions_per_worker = (all_positions.len() / workers).max(1);

    let mut handles = Vec::new();
    for w in 0..workers {
        let db_clone = db.clone();
        let pos_clone = all_positions.clone();
        let tx_clone = tx_chan.clone();
        let start_idx = w * positions_per_worker;
        let end_idx = if w == workers - 1 {
            pos_clone.len()
        } else {
            ((w + 1) * positions_per_worker).min(pos_clone.len())
        };

        handles.push(tokio::spawn(async move {
            let conn = db_clone.connect().expect("worker connect failed");
            let w_start = Instant::now();
            let mut count = 0usize;
            let slice = &pos_clone[start_idx..end_idx];

            for chunk in slice.chunks(batch_size) {
                loop {
                    // BEGIN CONCURRENT — MVCC optimistic write
                    conn.execute("BEGIN CONCURRENT", ()).await.expect("BEGIN CONCURRENT");
                    let mut failed = false;
                    for r in chunk {
                        let params = TursoParams::Positional(vec![
                            TursoValue::Integer(r.0),
                            TursoValue::Blob(r.1.clone()),
                            TursoValue::Integer(r.2),
                            TursoValue::Blob(r.3.clone()),
                            TursoValue::Integer(r.4),
                            TursoValue::Text(r.5.clone()),
                            TursoValue::Text(r.6.clone()),
                            TursoValue::Integer(r.7),
                            TursoValue::Integer(r.8),
                        ]);
                        let res = conn.execute(
                            "INSERT OR REPLACE INTO positions \
                             (position_id, canonical_hash, fast_hash, packed_state, side_to_move, \
                              first_seen_at, last_seen_at, total_visits, schema_version) \
                             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            params,
                        ).await;
                        if let Err(ref e) = res {
                            if is_retryable_turso_error(e) {
                                let _ = conn.execute("ROLLBACK", ()).await;
                                failed = true;
                                break;
                            } else {
                                let _ = conn.execute("ROLLBACK", ()).await;
                                panic!("non-retryable error in worker {}: {}", w, e);
                            }
                        }
                    }
                    if failed {
                        tokio::task::yield_now().await;
                        continue;
                    }
                    let commit_result = conn.execute("COMMIT", ()).await;
                    match commit_result {
                        Ok(_) => {
                            count += chunk.len();
                            break;
                        }
                        Err(ref e) if is_retryable_turso_error(e) => {
                            let _ = conn.execute("ROLLBACK", ()).await;
                            tokio::task::yield_now().await;
                        }
                        Err(e) => {
                            let _ = conn.execute("ROLLBACK", ()).await;
                            panic!("commit error in worker {}: {}", w, e);
                        }
                    }
                }
            }

            tx_clone
                .send((count, w_start.elapsed()))
                .await
                .expect("channel send failed");
        }));
    }
    drop(tx_chan);

    let mut total_rows = 0usize;
    while let Some((count, duration)) = rx_chan.recv().await {
        total_rows += count;
        println!("  Worker ingested {:6} rows in {:?}", count, duration);
    }
    for h in handles {
        h.await.expect("task panicked");
    }

    let elapsed = start_time.elapsed();
    let rate = total_rows as f64 / elapsed.as_secs_f64();
    println!(
        "=== Ingestion result: {} rows, {:?} elapsed, {:.0} pos/sec ===",
        total_rows, elapsed, rate
    );

    // Reader sanity check
    let conn = db.connect().context("reader connect")?;
    let r_start = Instant::now();
    let mut rows = conn
        .query("SELECT COUNT(*), SUM(total_visits) FROM positions", ())
        .await?;
    if let Some(r) = rows.next().await? {
        let cnt: i64 = r.get(0)?;
        let sum: i64 = r.get(1).unwrap_or(0);
        println!(
            "Read check: count={}, sum_visits={} in {:?}",
            cnt, sum, r_start.elapsed()
        );
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Idempotence / recovery smoke-test
// ---------------------------------------------------------------------------

async fn run_recovery_test(source: &Path, dest: &Path, batch_size: usize) -> Result<()> {
    println!("=== Recovery & idempotence smoke-test ===");

    let src_conn = SqliteConnection::open(source).context("Cannot open source SQLite")?;
    let mut stmt = src_conn.prepare(
        "SELECT position_id, canonical_hash, fast_hash, packed_state, side_to_move, \
         first_seen_at, last_seen_at, total_visits, schema_version \
         FROM positions LIMIT 2000",
    )?;
    type PosRow9 = (i64, Vec<u8>, i64, Vec<u8>, i64, String, String, i64, i64);
    let pos_list: Vec<PosRow9> = stmt
        .query_map([], |row| {
            Ok((
                row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?,
                row.get(4)?, row.get(5)?, row.get(6)?, row.get(7)?,
                row.get(8)?,
            ))
        })?
        .collect::<rusqlite::Result<_>>()?;

    let db = Builder::new_local(dest.to_str().unwrap())
        .build()
        .await
        .context("Cannot open destination Turso DB")?;
    let conn = db.connect().context("Cannot connect to Turso DB")?;

    let insert_batch = |slice: &[PosRow9]| {
        let mut p = Vec::new();
        for r in slice {
            p.push(TursoParams::Positional(vec![
                TursoValue::Integer(r.0),
                TursoValue::Blob(r.1.clone()),
                TursoValue::Integer(r.2),
                TursoValue::Blob(r.3.clone()),
                TursoValue::Integer(r.4),
                TursoValue::Text(r.5.clone()),
                TursoValue::Text(r.6.clone()),
                TursoValue::Integer(r.7),
                TursoValue::Integer(r.8),
            ]));
        }
        p
    };

    let first_batch = &pos_list[..batch_size.min(pos_list.len())];

    // First insert
    {
        let tx = conn.transaction().await?;
        for params in insert_batch(first_batch) {
            tx.execute(
                "INSERT OR REPLACE INTO positions \
                 (position_id, canonical_hash, fast_hash, packed_state, side_to_move, \
                  first_seen_at, last_seen_at, total_visits, schema_version) \
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                params,
            )
            .await?;
        }
        tx.commit().await?;
        println!("Initial batch committed ({} rows).", first_batch.len());
    }

    // Idempotent re-insert (INSERT OR REPLACE must not duplicate)
    {
        let tx = conn.transaction().await?;
        for params in insert_batch(first_batch) {
            tx.execute(
                "INSERT OR REPLACE INTO positions \
                 (position_id, canonical_hash, fast_hash, packed_state, side_to_move, \
                  first_seen_at, last_seen_at, total_visits, schema_version) \
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                params,
            )
            .await?;
        }
        tx.commit().await?;
        println!("Idempotent re-insert committed.");
    }

    let count = query_row_count(&conn, "SELECT COUNT(*) FROM positions").await?;
    let expected = first_batch.len() as i64;
    println!("Row count after idempotent re-insert: {} (expected {})", count, expected);

    if count != expected {
        anyhow::bail!(
            "Idempotence failed: expected {} rows, got {}",
            expected,
            count
        );
    }

    println!("SUCCESS: Idempotence verified.");
    Ok(())
}
