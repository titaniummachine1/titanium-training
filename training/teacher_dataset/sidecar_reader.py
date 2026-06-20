"""Read TIQSIDE1 gzip sidecar records (Rust importer format)."""
from __future__ import annotations

import gzip
import struct
from dataclasses import dataclass
from pathlib import Path

TIQSIDE1_MAGIC = b"TIQSIDE1"
TIQSIDE1_VERSION = 1


@dataclass(frozen=True)
class SidecarRecord:
    canonical_hash: bytes
    move_codes: tuple[int, ...]
    policy_values_u16: tuple[int, ...]

    @property
    def policy_values(self) -> tuple[float, ...]:
        return tuple(v / 65535.0 for v in self.policy_values_u16)


def read_record_at_offset(path: Path, offset: int, record_bytes: int) -> SidecarRecord:
    with gzip.open(path, "rb") as handle:
        handle.seek(offset)
        raw = handle.read(record_bytes)
    return decode_record(raw)


def decode_record(raw: bytes) -> SidecarRecord:
    if len(raw) < 1 + 32:
        raise ValueError(f"record too short: {len(raw)}")
    n = raw[0]
    canonical = raw[1:33]
    expected = 1 + 32 + n * 3
    if len(raw) != expected:
        raise ValueError(f"record length mismatch: got {len(raw)} expected {expected}")
    moves: list[int] = []
    values: list[int] = []
    pos = 33
    for _ in range(n):
        code = raw[pos]
        if code > 135:
            raise ValueError(f"move code {code} out of range 0..135 at offset {pos}")
        moves.append(code)
        values.append(struct.unpack_from("<H", raw, pos + 1)[0])
        pos += 3
    return SidecarRecord(canonical, tuple(moves), tuple(values))


def iter_sidecar_records(path: Path) -> list[tuple[int, SidecarRecord]]:
    """Scan entire decompressed sidecar; returns (decompressed_offset, record) pairs."""
    with gzip.open(path, "rb") as handle:
        data = handle.read()
    if len(data) < 10 or data[:8] != TIQSIDE1_MAGIC:
        raise ValueError(f"bad TIQSIDE1 header in {path}")
    version = struct.unpack_from("<H", data, 8)[0]
    if version != TIQSIDE1_VERSION:
        raise ValueError(f"unsupported TIQSIDE1 version {version} in {path.name}")
    pos = 10  # skip magic (8) + version u16 (2)
    out: list[tuple[int, SidecarRecord]] = []
    while pos < len(data):
        if pos + 33 > len(data):
            raise EOFError(f"truncated record header at decompressed offset {pos}")
        n = data[pos]
        rec_len = 1 + 32 + n * 3
        if pos + rec_len > len(data):
            raise EOFError(f"truncated record body at offset {pos}: need {rec_len}, have {len(data) - pos}")
        rec = decode_record(data[pos : pos + rec_len])
        out.append((pos, rec))
        pos += rec_len
    return out
