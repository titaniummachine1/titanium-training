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


def read_policy_chunk(bin_path: Path, idx_path: Path, record_id: int) -> EncodedPolicy:
    """Read one encoded policy record from finalized chunk files."""
    idx_data = idx_path.read_bytes()
    if not idx_data.startswith(POLICY_INDEX_MAGIC):
        raise ValueError("bad policy index magic")
    _version, count = struct.unpack_from("<HI", idx_data, 8)  # version u16 then count u32
    if record_id < 0 or record_id >= count:
        raise IndexError(f"policy record_id out of range: {record_id}")
    header_size = len(POLICY_INDEX_MAGIC) + struct.calcsize("<HI")
    entry_size = struct.calcsize("<IQII32s")
    entry_off = header_size + record_id * entry_size
    rid, payload_off, payload_len, crc, content_hash = struct.unpack_from("<IQII32s", idx_data, entry_off)
    if rid != record_id:
        raise ValueError(f"index rid mismatch: {rid} != {record_id}")
    blob = bin_path.read_bytes()
    payload = blob[payload_off : payload_off + payload_len]
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
