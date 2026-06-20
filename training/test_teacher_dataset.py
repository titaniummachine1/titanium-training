"""Tests for teacher dataset sidecar repair and policy read."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from teacher_dataset.sidecar_paths import classify_sidecar_path, resolve_sidecar_path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "training" / "data" / "canonical" / "position_teacher_store.db"


@pytest.mark.skipif(not DB.exists(), reason="teacher store not present")
def test_wrong_base_sidecar_path_remaps_to_teacher_sidecars() -> None:
    conn = sqlite3.connect(DB)
    row = conn.execute(
        "SELECT l.payload_json, hex(p.canonical_hash) "
        "FROM labels l JOIN positions p ON p.position_id = l.position_id "
        "WHERE l.payload_json LIKE ? LIMIT 1",
        ("%iter_000002.policy.bin%",),
    ).fetchone()
    conn.close()
    assert row is not None
    payload = json.loads(row[0])
    ref = payload["sidecar_ref"]
    stored = ref["sidecar"]
    assert "friend_selfplay" in stored
    cls = classify_sidecar_path(stored, root=ROOT.parent)
    assert cls in {
        "repaired_wrong_base_friend_selfplay_at_root",
        "path_ok_teacher_sidecars_relative",
        "path_ok_other",
    }
    path = resolve_sidecar_path(stored, root=ROOT.parent)
    assert path.is_file(), path


def test_sidecar_record_roundtrip_unit() -> None:
    from teacher_dataset.sidecar_reader import decode_record
    import struct

    n = 2
    raw = bytes([n]) + bytes(32) + struct.pack("<BH", 128, 32768) + struct.pack("<BH", 129, 16384)
    rec = decode_record(raw)
    assert rec.move_codes == (128, 129)


def test_friend_state_uses_current_player_field() -> None:
    from teacher_dataset.friend_state import parse_friend_state

    state = parse_friend_state(
        {
            "state": {
                "player0Cell": 4,
                "player1Cell": 76,
                "player0Walls": 10,
                "player1Walls": 10,
                "horizontalWalls": 0,
                "verticalWalls": 0,
                "currentPlayer": 1,
            }
        }
    )
    assert state.side_to_move == 1
    assert state.packed_state()[5] == 1


def test_policy_chunk_writer_readback() -> None:
    from teacher_dataset.policy_binary import EncodedPolicy, PolicyChunkWriter, read_policy_chunk
    import tempfile

    writer = PolicyChunkWriter(chunk_id=0)
    enc = EncodedPolicy.from_sparse([128, 129], [0.5, 0.5])
    rid = writer.add(enc)
    bin_bytes, idx_bytes = writer.finalize()
    with tempfile.TemporaryDirectory() as tmp:
        bin_path = Path(tmp) / "policy.bin"
        idx_path = Path(tmp) / "policy.idx"
        bin_path.write_bytes(bin_bytes)
        idx_path.write_bytes(idx_bytes)
        back = read_policy_chunk(bin_path, idx_path, rid)
    assert back.move_codes == enc.move_codes


def test_golden_vector_packed_hash_roundtrip() -> None:
    import json
    from teacher_dataset.canonical_identity import canonical_hash_from_packed, verify_stored_canonical
    from teacher_dataset.friend_state import parse_friend_state

    vectors = json.loads((Path(__file__).resolve().parent / "fixtures" / "position_golden_vectors.json").read_text())
    for vec in vectors:
        if "state" not in vec:
            continue
        packed = parse_friend_state({"state": vec["state"]}).packed_state()
        assert len(packed) == 24
        stored = canonical_hash_from_packed(packed)
        assert verify_stored_canonical(packed, stored)


def test_friend_state_rejects_missing_current_player() -> None:
    from teacher_dataset.friend_state import parse_friend_state

    with pytest.raises(KeyError, match="state.currentPlayer"):
        parse_friend_state(
            {
                "state": {
                    "player0Cell": 4,
                    "player1Cell": 76,
                    "player0Walls": 10,
                    "player1Walls": 10,
                    "horizontalWalls": 0,
                    "verticalWalls": 0,
                }
            }
        )


def test_friend_state_rejects_side_to_move_only() -> None:
    """sideToMove is a legacy alias; parse_friend_state must not silently accept it."""
    from teacher_dataset.friend_state import parse_friend_state

    with pytest.raises(KeyError, match="state.currentPlayer"):
        parse_friend_state(
            {
                "state": {
                    "player0Cell": 4,
                    "player1Cell": 76,
                    "player0Walls": 10,
                    "player1Walls": 10,
                    "horizontalWalls": 0,
                    "verticalWalls": 0,
                    "sideToMove": 0,
                }
            }
        )


def test_sidecar_iter_rejects_bad_magic(tmp_path: Path) -> None:
    import gzip

    from teacher_dataset.sidecar_reader import iter_sidecar_records

    bad = tmp_path / "bad.policy.bin.gz"
    with gzip.open(bad, "wb") as f:
        f.write(b"BADMAGIC" + b"\x00" * 10)
    with pytest.raises(ValueError, match="TIQSIDE1"):
        iter_sidecar_records(bad)


def test_sidecar_decode_rejects_truncated_record() -> None:
    from teacher_dataset.sidecar_reader import decode_record

    with pytest.raises(ValueError, match="record too short"):
        decode_record(b"\x01" + b"\x00" * 10)


def test_sidecar_decode_rejects_length_mismatch() -> None:
    from teacher_dataset.sidecar_reader import decode_record

    n = 2
    raw = bytes([n]) + bytes(32) + bytes(3)
    with pytest.raises(ValueError, match="record length mismatch"):
        decode_record(raw)


def test_sidecar_decode_rejects_move_code_out_of_range() -> None:
    import struct

    from teacher_dataset.sidecar_reader import decode_record

    n = 1
    raw = bytes([n]) + bytes(32) + bytes([136]) + struct.pack("<H", 0)
    with pytest.raises(ValueError, match="move code 136 out of range"):
        decode_record(raw)


def test_sidecar_decode_accepts_all_valid_move_codes() -> None:
    """All move codes 0..135 must be accepted without ValueError."""
    import struct

    from teacher_dataset.sidecar_reader import decode_record

    for code in range(136):
        n = 1
        raw = bytes([n]) + bytes(32) + bytes([code]) + struct.pack("<H", 32768)
        rec = decode_record(raw)
        assert rec.move_codes == (code,)


def test_build_teacher_dataset_manifest_structure(tmp_path: Path) -> None:
    """build_teacher_dataset writes manifest.json + schema.json, no .partial files, promotion_allowed=False."""
    import sqlite3

    from teacher_dataset.build import build_teacher_dataset
    from teacher_dataset.schema import TEACHER_DATASET_SCHEMA_VERSION

    db = tmp_path / "teacher.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(
        """
        CREATE TABLE positions (
            position_id INTEGER PRIMARY KEY,
            canonical_hash BLOB NOT NULL,
            packed_state BLOB NOT NULL,
            side_to_move INTEGER NOT NULL DEFAULT 0,
            total_visits INTEGER NOT NULL DEFAULT 0,
            source_flags INTEGER
        );
        CREATE TABLE labels (
            label_id INTEGER PRIMARY KEY,
            position_id INTEGER NOT NULL,
            label_type TEXT NOT NULL,
            value REAL,
            best_move_u8 INTEGER,
            source TEXT,
            payload_json TEXT
        );
        CREATE TABLE observations (
            observation_id INTEGER PRIMARY KEY,
            position_id INTEGER NOT NULL,
            source_cohort TEXT,
            visit_count INTEGER,
            p0_wins INTEGER,
            p1_wins INTEGER,
            draws INTEGER
        );
        """
    )
    conn.commit()
    conn.close()

    out_dir = tmp_path / "candidate"
    manifest = build_teacher_dataset(output_dir=out_dir, sqlite_db=db)

    assert (out_dir / "manifest.json").exists(), "manifest.json must exist after build"
    assert (out_dir / "schema.json").exists(), "schema.json must exist after build"
    assert manifest["schema_version"] == TEACHER_DATASET_SCHEMA_VERSION
    assert manifest["promotion_allowed"] is False

    partial_files = list(out_dir.rglob("*.partial"))
    assert partial_files == [], f"partial files must not remain after build: {partial_files}"


def test_golden_tiqside1_fixture_exact_bytes() -> None:
    """Python reader must decode the Rust-format golden fixture to exact expected values."""
    from teacher_dataset.sidecar_reader import iter_sidecar_records

    fixture = Path(__file__).resolve().parent / "fixtures" / "golden_tiqside1.policy.bin.gz"
    assert fixture.exists(), f"golden fixture missing: {fixture}"
    recs = iter_sidecar_records(fixture)
    assert len(recs) == 2, f"expected 2 records, got {len(recs)}"

    off0, rec0 = recs[0]
    assert off0 == 10, f"record 0 offset should be 10 (after 8+2 header), got {off0}"
    assert rec0.canonical_hash == b"\x00" * 32
    assert rec0.move_codes == (0,)
    assert rec0.policy_values_u16 == (65535,)  # 1.0 exactly

    off1, rec1 = recs[1]
    assert off1 == 46, f"record 1 offset should be 46 (10 + 36), got {off1}"
    assert rec1.canonical_hash == b"\x01" * 32
    assert rec1.move_codes == (0, 135)  # boundary move codes 0 and 135
    assert rec1.policy_values_u16 == (32768, 32767)


def test_iter_sidecar_records_parses_multiple_records() -> None:
    """iter_sidecar_records must parse all records, not fail on the first due to length-mismatch bug."""
    import gzip
    import struct

    from teacher_dataset.sidecar_reader import TIQSIDE1_MAGIC, TIQSIDE1_VERSION, iter_sidecar_records

    def make_record(n: int, hash_byte: int, code: int, value: int) -> bytes:
        return bytes([n]) + bytes([hash_byte] * 32) + bytes([code]) + struct.pack("<H", value)

    # Build a minimal valid TIQSIDE1 sidecar with 3 records
    header = TIQSIDE1_MAGIC + struct.pack("<H", TIQSIDE1_VERSION)
    body = make_record(1, 0xAA, 10, 1000) + make_record(1, 0xBB, 20, 2000) + make_record(1, 0xCC, 30, 3000)
    content = header + body

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".policy.bin.gz", delete=False) as f:
        fpath = Path(f.name)
    try:
        with gzip.open(fpath, "wb") as gz:
            gz.write(content)
        recs = iter_sidecar_records(fpath)
    finally:
        fpath.unlink(missing_ok=True)

    assert len(recs) == 3, f"expected 3 records, got {len(recs)}"
    assert recs[0][1].move_codes == (10,)
    assert recs[1][1].move_codes == (20,)
    assert recs[2][1].move_codes == (30,)


def test_read_policy_chunk_handles_large_record_ids() -> None:
    """read_policy_chunk must not truncate count to u16 (bug: read version field as count)."""
    from teacher_dataset.policy_binary import EncodedPolicy, PolicyChunkWriter, read_policy_chunk
    import tempfile

    writer = PolicyChunkWriter(chunk_id=0)
    # Add 70,000 records to exceed u16 max (65535)
    target_rid = 65536
    for _ in range(target_rid + 1):
        enc = EncodedPolicy.from_sparse([0], [0.5])
        writer.add(enc)
    bin_bytes, idx_bytes = writer.finalize()
    with tempfile.TemporaryDirectory() as tmp:
        bin_path = Path(tmp) / "policy.bin"
        idx_path = Path(tmp) / "policy.idx"
        bin_path.write_bytes(bin_bytes)
        idx_path.write_bytes(idx_bytes)
        enc = read_policy_chunk(bin_path, idx_path, target_rid)
    assert enc.move_codes == (0,)


def test_policy_lookup_requires_packed_identity_not_hash_only() -> None:
    from teacher_dataset.jsonl_policy_index import build_jsonl_policy_index
    from teacher_dataset.policy_lookup import PolicyLookupStats, lookup_teacher_policy
    from teacher_dataset.sidecar_policy_index import build_sidecar_policy_index

    sidecar_index, _ = build_sidecar_policy_index()
    _jc, jsonl_by_packed = build_jsonl_policy_index()
    stats = PolicyLookupStats()
    fake_canonical = b"\x01" * 32
    fake_packed = b"\x02" * 24
    result = lookup_teacher_policy(
        canonical_hash=fake_canonical,
        packed_state=fake_packed,
        policy_hash="deadbeef",
        sidecar_ref=None,
        source="friend_selfplay:iter_000001",
        label_id=1,
        sidecar_index=sidecar_index,
        jsonl_by_packed=jsonl_by_packed,
        stats=stats,
    )
    assert result is None
    assert stats.unresolved == 1
