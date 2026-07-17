use libsql::Builder;
use std::path::Path;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let db_path = "local_turso_poc.db";
    
    // Clean up any old POC db files to ensure a clean slate
    if Path::new(db_path).exists() {
        let _ = std::fs::remove_file(db_path);
    }
    
    println!("--- Phase 2: Open a local Turso database ---");
    let db = Builder::new_local(db_path).build().await?;
    let conn = db.connect()?;
    
    println!("--- Create tables ---");
    conn.execute(
        "CREATE TABLE IF NOT EXISTS test_positions (
            position_id INTEGER PRIMARY KEY,
            canonical_hash BLOB NOT NULL UNIQUE,
            visit_count INTEGER NOT NULL DEFAULT 0
        )",
        (),
    ).await?;
    
    println!("--- Create indexes ---");
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_test_pos_hash ON test_positions(canonical_hash)",
        (),
    ).await?;
    
    println!("--- Insert rows in transaction ---");
    // In libsql, we can get a transaction
    let tx = conn.transaction().await?;
    tx.execute(
        "INSERT INTO test_positions (position_id, canonical_hash, visit_count) VALUES (1, ?, 42)",
        libsql::params![vec![0x12, 0x34, 0x56]],
    ).await?;
    tx.execute(
        "INSERT INTO test_positions (position_id, canonical_hash, visit_count) VALUES (2, ?, 100)",
        libsql::params![vec![0xab, 0xcd, 0xef]],
    ).await?;
    tx.commit().await?;
    
    println!("--- Query rows ---");
    let mut rows = conn.query("SELECT position_id, canonical_hash, visit_count FROM test_positions ORDER BY position_id", ()).await?;
    while let Some(row) = rows.next().await? {
        let id: i64 = row.get(0)?;
        let hash: Vec<u8> = row.get(1)?;
        let visits: i64 = row.get(2)?;
        println!("Row: id={}, hash={:?}, visits={}", id, hash, visits);
    }
    
    // Drop connection and DB to close
    drop(conn);
    drop(db);
    
    println!("--- Reopen and verify persisted data ---");
    let db2 = Builder::new_local(db_path).build().await?;
    let conn2 = db2.connect()?;
    let mut rows2 = conn2.query("SELECT position_id, canonical_hash, visit_count FROM test_positions ORDER BY position_id", ()).await?;
    
    let mut count = 0;
    while let Some(row) = rows2.next().await? {
        let id: i64 = row.get(0)?;
        let hash: Vec<u8> = row.get(1)?;
        let visits: i64 = row.get(2)?;
        println!("Reopened Row: id={}, hash={:?}, visits={}", id, hash, visits);
        count += 1;
    }
    
    if count == 2 {
        println!("SUCCESS: Reopened and verified persisted data matches!");
    } else {
        println!("FAILURE: Expected 2 rows, found {}", count);
    }
    
    Ok(())
}
