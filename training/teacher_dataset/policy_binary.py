"""Compact sparse policy encoding for immutable teacher_dataset sidecars."""
from __future__ import annotations

import hashlib
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path

from .schema import POLICY_CHUNK_MAGIC, POLICY_INDEX_MAGIC, POLICY_SIDECAR_SCHEMA_VERSION


@dataclass(frozen=True)
class EncodedPolicy:
    move_codes: tuple[int, ...]
    values_u16: tuple[int, ...]
    content_hash: bytes

    @classmethod
    def from_sparse(cls, move_codes: list[int] | tuple[int, ...], values: list[float] | tuple[float, ...]) -> EncodedPolicy:
        if len(move_codes) != len(values):
            raise ValueError("move_codes/values length mismatch")
        u16 = tuple(min(65535, max(0, int(round(float(v) * 65535)))) for v in values)
        payload = struct.pack("<BB", len(move_codes) & 0xFF, 1)  # count, u16 encoding
        for mv, q in zip(move_codes, u16):
            payload += struct.pack("<BH", mv & 0xFF, q)
        content_hash = hashlib.blake2b(payload, digest_size=32).digest()
        return cls(tuple(int(m) & 0xFF for m in move_codes), u16, content_hash)

    def to_bytes(self) -> bytes:
        header = struct.pack("<BB", len(self.move_codes) & 0xFF, 1)
        body = b"".join(struct.pack("<BH", mv, q) for mv, q in zip(self.move_codes, self.values_u16))
        return header + body


@dataclass
class PolicyChunkReaderStats:
    file_opens: int = 0
    bytes_read: int = 0
    record_reads: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "file_opens": self.file_opens,
            "bytes_read": self.bytes_read,
            "record_reads": self.record_reads,
        }


class PolicyChunkReader:
    """Load policy bin/idx once; read individual records by slice only."""

    def __init__(self, bin_path: Path, idx_path: Path) -> None:
        self.bin_path = Path(bin_path)
        self.idx_path = Path(idx_path)
        self.stats = PolicyChunkReaderStats()
        self._bin_blob: bytes | None = None
        self._idx_data: bytes | None = None
        self._count: int = 0
        self._header_size = len(POLICY_INDEX_MAGIC) + struct.calcsize("<HI")
        self._entry_size = struct.calcsize("<IQII32s")

    def __enter__(self) -> PolicyChunkReader:
        self.open()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def open(self) -> None:
        if self._bin_blob is not None:
            return
        self.stats.file_opens += 2
        self._bin_blob = self.bin_path.read_bytes()
        self.stats.bytes_read += len(self._bin_blob)
        self._idx_data = self.idx_path.read_bytes()
        self.stats.bytes_read += len(self._idx_data)
        if not self._idx_data.startswith(POLICY_INDEX_MAGIC):
            raise ValueError("bad policy index magic")
        header = self._bin_blob[:8]
        if not header.startswith(POLICY_CHUNK_MAGIC):
            raise ValueError("bad policy chunk magic")
        _version, self._count = struct.unpack_from("<HI", self._idx_data, 8)

    def close(self) -> None:
        self._bin_blob = None
        self._idx_data = None

    @property
    def record_count(self) -> int:
        self.open()
        return self._count

    def read(self, record_id: int) -> EncodedPolicy:
        self.open()
        assert self._bin_blob is not None and self._idx_data is not None
        if record_id < 0 or record_id >= self._count:
            raise IndexError(f"policy record_id out of range: {record_id}")
        entry_off = self._header_size + record_id * self._entry_size
        rid, payload_off, payload_len, crc, content_hash = struct.unpack_from(
            "<IQII32s", self._idx_data, entry_off
        )
        if rid != record_id:
            raise ValueError(f"index rid mismatch: {rid} != {record_id}")
        payload = self._bin_blob[payload_off : payload_off + payload_len]
        self.stats.record_reads += 1
        if (zlib.crc32(payload) & 0xFFFFFFFF) != crc:
            raise ValueError("policy payload crc mismatch")
        n_moves, enc = struct.unpack_from("<BB", payload, 0)
        if enc != 1:
            raise ValueError(f"unsupported policy encoding: {enc}")
        move_codes: list[int] = []
        values_u16: list[int] = []
        pos = 2
        for _ in range(n_moves):
            mv, q = struct.unpack_from("<BH", payload, pos)
            move_codes.append(mv)
            values_u16.append(q)
            pos += 3
        return EncodedPolicy(tuple(move_codes), tuple(values_u16), content_hash)


def read_policy_chunk(bin_path: Path, idx_path: Path, record_id: int) -> EncodedPolicy:
    """Read one encoded policy record from finalized chunk files."""
    with PolicyChunkReader(bin_path, idx_path) as reader:
        return reader.read(record_id)


@dataclass
class PolicyChunkWriter:
    chunk_id: int = 0
    records: list[tuple[int, bytes, bytes]] = field(default_factory=list)

    def add(self, encoded: EncodedPolicy) -> int:
        rid = len(self.records)
        self.records.append((rid, encoded.to_bytes(), encoded.content_hash))
        return rid

    def finalize(self) -> tuple[bytes, bytes]:
        """Return (bin_bytes, idx_bytes) ready for atomic rename."""
        bin_parts = [
            POLICY_CHUNK_MAGIC,
            struct.pack("<HII", POLICY_SIDECAR_SCHEMA_VERSION, len(self.records), 0),
        ]
        idx_parts = [POLICY_INDEX_MAGIC, struct.pack("<HI", POLICY_SIDECAR_SCHEMA_VERSION, len(self.records))]
        offset = len(b"".join(bin_parts)) + 4  # checksum placeholder
        for rid, payload, content_hash in self.records:
            crc = zlib.crc32(payload) & 0xFFFFFFFF
            bin_parts.append(struct.pack("<I", len(payload)))
            bin_parts.append(payload)
            idx_parts.append(
                struct.pack("<IQII32s", rid, offset, len(payload), crc, content_hash)
            )
            offset += 4 + len(payload)
        bin_body = b"".join(bin_parts)
        chunk_crc = zlib.crc32(bin_body) & 0xFFFFFFFF
        bin_bytes = bin_body[:16] + struct.pack("<I", chunk_crc) + bin_body[20:]
        idx_bytes = b"".join(idx_parts)
        return bin_bytes, idx_bytes
