#!/usr/bin/env python3
"""Build an isolated corpus of exact labels for canonical hands-empty states.

This tool is intentionally *not* an importer.  It opens the canonical SQLite
store read-only, solves only positions whose two wall-count bytes are zero,
and writes a new JSONL file.  The resulting file is an immutable staging
artifact; a separate, audited import step is required before it can affect
training data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator, Sequence

_TRAINING = Path(__file__).resolve().parent
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

from titanium_training.data.hands_empty_oracle import (  # noqa: E402
    HandsEmptyOracleResult,
    PACKED_STATE_LEN,
    PROTOCOL,
    oracle_packed_batch,
)
from titanium_training.paths import DATA_DIR, TEACHER_STORE_DB  # noqa: E402
from titanium_training.store.state import PositionState  # noqa: E402

OUTPUT_DIR = DATA_DIR / "oracle_endgame_labels"
RECORD_FORMAT = "titanium-exact-hands-empty-label-v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _readonly_connection(db_path: Path) -> sqlite3.Connection:
    if not db_path.is_file():
        raise FileNotFoundError(db_path)
    # ``mode=ro`` prevents accidental schema/journal writes even when a caller
    # points this command at a production store.
    con = sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)
    con.execute("PRAGMA query_only=ON")
    columns = {str(row[1]) for row in con.execute("PRAGMA table_info(positions)")}
    required = {"position_id", "canonical_hash", "packed_state", "side_to_move"}
    missing = required - columns
    if missing:
        con.close()
        raise ValueError(f"positions table lacks canonical packed-state columns: {sorted(missing)}")
    return con


def _eligible_rows(con: sqlite3.Connection, *, max_positions: int | None) -> Iterator[tuple[int, bytes, bytes, int]]:
    # SQLite byte offsets are one-based: packed[3] and packed[4] are the two
    # remaining-wall counters in PositionState.packed_state().
    sql = """
        SELECT position_id, canonical_hash, packed_state, side_to_move
        FROM positions
        WHERE length(packed_state) = ?
          AND substr(packed_state, 4, 1) = x'00'
          AND substr(packed_state, 5, 1) = x'00'
        ORDER BY canonical_hash, packed_state, position_id
    """
    params: list[object] = [PACKED_STATE_LEN]
    if max_positions is not None:
        sql += " LIMIT ?"
        params.append(max_positions)
    for row in con.execute(sql, params):
        position_id, canonical_hash, packed_state, side_to_move = row
        packed = bytes(packed_state)
        canonical = bytes(canonical_hash)
        if len(packed) != PACKED_STATE_LEN or packed[3] != 0 or packed[4] != 0:
            raise RuntimeError(f"eligible SQL row {position_id} violated hands-empty filter")
        state = PositionState.unpack_state(packed)
        state.validate()
        if int(side_to_move) != state.side_to_move:
            raise RuntimeError(f"position {position_id} side_to_move disagrees with packed state")
        if canonical != state.canonical_hash():
            raise RuntimeError(f"position {position_id} canonical_hash disagrees with packed state")
        yield int(position_id), canonical, packed, int(side_to_move)


def _create_only(temp_path: Path, output_path: Path) -> None:
    """Publish a complete file without ever replacing a prior corpus."""
    try:
        # Hard links are atomic create-only on the same filesystem.  The temp
        # file is deliberately created in output_path.parent for that reason.
        os.link(temp_path, output_path)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite immutable corpus: {output_path}") from exc
    finally:
        temp_path.unlink(missing_ok=True)


def generate_labels(
    db_path: Path,
    output_path: Path,
    *,
    batch_size: int = 128,
    max_positions: int | None = None,
    timeout_sec: float | None = None,
    oracle: Callable[..., list[HandsEmptyOracleResult]] = oracle_packed_batch,
) -> dict[str, object]:
    """Generate a create-only JSONL corpus; source SQLite is never written."""
    db_path, output_path = Path(db_path), Path(output_path)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_positions is not None and max_positions <= 0:
        raise ValueError("max_positions must be positive when supplied")
    if output_path.suffix != ".jsonl":
        raise ValueError("output_path must end in .jsonl")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite immutable corpus: {output_path}")
    if output_path.resolve() == db_path.resolve():
        raise ValueError("output_path must not be the source database")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_db_sha256 = _sha256_file(db_path)
    generated_at = datetime.now(timezone.utc).isoformat()
    con = _readonly_connection(db_path)
    fd, raw_tmp = tempfile.mkstemp(prefix=f".{output_path.name}.", suffix=".partial", dir=output_path.parent)
    temp_path = Path(raw_tmp)
    count = 0
    content_sha256 = hashlib.sha256()
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            batch: list[tuple[int, bytes, bytes, int]] = []
            for item in _eligible_rows(con, max_positions=max_positions):
                batch.append(item)
                if len(batch) >= batch_size:
                    count += _write_batch(handle, batch, source_db_sha256, generated_at, content_sha256, timeout_sec, oracle)
                    batch.clear()
            if batch:
                count += _write_batch(handle, batch, source_db_sha256, generated_at, content_sha256, timeout_sec, oracle)
            handle.flush()
            os.fsync(handle.fileno())
        _create_only(temp_path, output_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    finally:
        con.close()
    return {
        "format": RECORD_FORMAT,
        "output": str(output_path),
        "records": count,
        "content_sha256": content_sha256.hexdigest(),
        "source_db_sha256": source_db_sha256,
    }


def _write_batch(
    handle,
    batch: Sequence[tuple[int, bytes, bytes, int]],
    source_db_sha256: str,
    generated_at: str,
    content_sha256: "hashlib._Hash",
    timeout_sec: float | None,
    oracle: Callable[..., list[HandsEmptyOracleResult]],
) -> int:
    answers = oracle([(index, row[2]) for index, row in enumerate(batch)], timeout_sec=timeout_sec)
    if len(answers) != len(batch):
        raise RuntimeError(f"oracle answer count mismatch: {len(answers)} answers vs {len(batch)} inputs")
    for index, ((position_id, canonical_hash, packed, side_to_move), answer) in enumerate(zip(batch, answers, strict=True)):
        if answer.row != index or not answer.eligible or answer.exact_value_plies is None or answer.value_stm is None or answer.distance_plies is None:
            raise RuntimeError(f"oracle did not return an exact answer for position {position_id}")
        record = {
            "format": RECORD_FORMAT,
            "position_id": position_id,
            "canonical_hash_hex": canonical_hash.hex(),
            "packed_state_hex": packed.hex(),
            "side_to_move": side_to_move,
            "protocol": PROTOCOL,
            "exact_value_plies": answer.exact_value_plies,
            "value_stm": answer.value_stm,
            "distance_plies": answer.distance_plies,
            "source_db_sha256": source_db_sha256,
            "generated_at": generated_at,
        }
        encoded = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        handle.write(encoded.decode("utf-8"))
        content_sha256.update(encoded)
    return len(batch)


def main() -> int:
    from prep_guard import guard_real_work

    guard_real_work("labeling", detail="oracle_endgame_labels.py")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, default=TEACHER_STORE_DB, help="read-only canonical SQLite store")
    ap.add_argument("--out", type=Path, default=None, help="new .jsonl path (default is a unique file under oracle_endgame_labels)")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-positions", type=int, default=None, help="bounded pilot; omit to label all eligible rows")
    ap.add_argument("--timeout-sec", type=float, default=None)
    args = ap.parse_args()
    output = args.out
    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = OUTPUT_DIR / f"hands_empty_exact_{stamp}_{os.getpid()}.jsonl"
    print(json.dumps(generate_labels(args.db, output, batch_size=args.batch_size, max_positions=args.max_positions, timeout_sec=args.timeout_sec), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
