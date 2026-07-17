#!/usr/bin/env python3
"""Collect bounded, exact alpha-beta labels for canonical packed positions."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from extend_teacher_dataset import float_stm_to_value_i16
from label_resolution import stm_from_eval_cp
from titanium_training.paths import ENGINE_BIN
from titanium_training.store.state import PositionState

RECORD_FORMAT = "titanium-bounded-ab-value-label-v1"
PROTOCOL = "score-out-v1"
PACKED_STATE_LEN = 24


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str | None:
    root = Path(__file__).resolve().parents[1]
    try:
        head = (root / ".git" / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref: "):
            return (root / ".git" / head[5:]).read_text(encoding="utf-8").strip() or None
        return head or None
    except (OSError, UnicodeError):
        return None


def _read_rows(db_path: Path, max_positions: int) -> list[tuple[int, bytes, bytes, int]]:
    if not db_path.is_file():
        raise FileNotFoundError(db_path)
    con = sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        con.execute("PRAGMA query_only=ON")
        rows = con.execute(
            """
            SELECT position_id, canonical_hash, packed_state, side_to_move
            FROM positions
            WHERE length(packed_state) = ?
            ORDER BY canonical_hash, packed_state, position_id
            LIMIT ?
            """,
            (PACKED_STATE_LEN, max_positions),
        )
        result: list[tuple[int, bytes, bytes, int]] = []
        for position_id, canonical_hash, packed_state, side_to_move in rows:
            packed = bytes(packed_state)
            canonical = bytes(canonical_hash)
            try:
                state = PositionState.unpack_state(packed)
                state.validate()
            except (TypeError, ValueError) as exc:
                raise ValueError(f"position {position_id} has invalid packed_state: {exc}") from exc
            if type(side_to_move) is not int or side_to_move != state.side_to_move:
                raise ValueError(f"position {position_id} side_to_move disagrees with packed state")
            if canonical != hashlib.sha256(packed).digest():
                raise ValueError(f"position {position_id} canonical_hash disagrees with packed state")
            result.append((int(position_id), canonical, packed, side_to_move))
        return result
    finally:
        con.close()


def _value_i16(result: dict[str, Any]) -> int:
    return int(float_stm_to_value_i16(stm_from_eval_cp(result["score"])))


def _is_int(value: object, *, minimum: int = 0, maximum: int | None = None) -> bool:
    return (
        type(value) is int
        and value >= minimum
        and (maximum is None or value <= maximum)
    )


def _valid_score_out(result: dict[str, Any] | None, *, side_to_move: int, node_budget: int) -> bool:
    if result is None:
        return False
    return (
        result.get("schema") == PROTOCOL
        and result.get("ok") is True
        and result.get("input") == "packed"
        and type(result.get("side_to_move")) is int
        and result["side_to_move"] == side_to_move
        and _is_int(result.get("score"), minimum=-(2**31), maximum=2**31 - 1)
        and result.get("bound") == "exact"
        and type(result.get("proven")) is bool
        and _is_int(result.get("nodes"))
        and _is_int(result.get("node_budget"), minimum=1)
        and result["node_budget"] == node_budget
        and _is_int(result.get("depth"))
        and isinstance(result.get("selected_move"), str)
    )


def _score_out(engine_bin: Path, packed: str, node_budget: int) -> dict[str, Any] | None:
    try:
        completed = subprocess.run(
            [str(engine_bin), "score-out", "--nodes", str(node_budget), "--packed", packed],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise RuntimeError(f"failed to invoke Titanium: {exc}") from exc
    if completed.returncode != 0:
        raise RuntimeError(f"Titanium score-out failed with exit code {completed.returncode}")
    try:
        decoder = json.JSONDecoder()
        start = len(completed.stdout) - len(completed.stdout.lstrip())
        parsed, end = decoder.raw_decode(completed.stdout, start)
    except (AttributeError, json.JSONDecodeError, TypeError):
        return None
    if completed.stdout[end:].strip():
        return None
    return parsed if isinstance(parsed, dict) else None


def collect_labels(
    db_path: Path,
    output_path: Path,
    *,
    max_positions: int = 500,
    node_budget: int = 200_000,
    engine_bin: Path = ENGINE_BIN,
) -> dict[str, Any]:
    """Write a complete create-only JSONL corpus, leaving the source DB untouched."""
    if max_positions <= 0:
        raise ValueError("max_positions must be positive")
    if node_budget <= 0:
        raise ValueError("node_budget must be positive")
    db_path, output_path, engine_bin = Path(db_path), Path(output_path), Path(engine_bin)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite output: {output_path}")
    if output_path.resolve() == db_path.resolve():
        raise ValueError("output must not be the source database")
    if not engine_bin.is_file():
        raise FileNotFoundError(engine_bin)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_db_sha256 = _sha256_file(db_path)
    engine_sha256 = _sha256_file(engine_bin)
    generated_at = datetime.now(timezone.utc).isoformat()
    rows = _read_rows(db_path, max_positions)
    skipped = 0
    fd, raw_tmp = tempfile.mkstemp(prefix=f".{output_path.name}.", suffix=".partial", dir=output_path.parent)
    temp_path = Path(raw_tmp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            for position_id, canonical, packed_bytes, side_to_move in rows:
                packed = packed_bytes.hex()
                result = _score_out(engine_bin, packed, node_budget)
                if not _valid_score_out(result, side_to_move=side_to_move, node_budget=node_budget):
                    skipped += 1
                    continue
                assert result is not None
                value_i16 = _value_i16(result)
                record = {
                    "format": RECORD_FORMAT,
                    "position_id": position_id,
                    "canonical_hash_hex": canonical.hex(),
                    "packed_state_hex": packed,
                    "side_to_move": side_to_move,
                    "source_db_sha256": source_db_sha256,
                    "engine_executable_sha256": engine_sha256,
                    "git_commit": _git_commit(),
                    "generated_at": generated_at,
                    "protocol": {"schema": PROTOCOL, "node_budget": node_budget},
                    "value_i16": value_i16,
                    "score_out": result,
                }
                record.update({key: value for key, value in result.items() if key not in record})
                handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temp_path, output_path)
        except FileExistsError as exc:
            raise FileExistsError(f"refusing to overwrite output: {output_path}") from exc
        return {"format": RECORD_FORMAT, "output": str(output_path), "selected": len(rows),
                "records": len(rows) - skipped, "skipped": skipped}
    finally:
        temp_path.unlink(missing_ok=True)


def main() -> int:
    from prep_guard import guard_real_work

    guard_real_work("labeling", detail="score_out_labels.py")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-positions", type=int, default=500)
    parser.add_argument("--node-budget", type=int, default=200_000)
    parser.add_argument("--engine-bin", type=Path, default=ENGINE_BIN)
    args = parser.parse_args()
    summary = collect_labels(args.db, args.out, max_positions=args.max_positions,
                             node_budget=args.node_budget, engine_bin=args.engine_bin)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
