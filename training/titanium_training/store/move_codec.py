"""Compact move storage — ACE i16 encoding, 2 bytes per ply (matches engine)."""

from __future__ import annotations

import struct

# Move encoding (engine/src/titanium/mod.rs algebraic_to_move_id / move_id_to_algebraic)


def algebraic_to_ace(text: str) -> int:
    b = text.encode("ascii")
    col = b[0] - ord("a")
    row = b[1] - ord("1")
    if len(b) > 2:
        slot = (7 - row) * 8 + col
        if b[2] == ord("h"):
            return 100 + slot
        if b[2] == ord("v"):
            return 200 + slot
        raise ValueError(f"bad wall suffix in {text!r}")
    return (8 - row) * 9 + col


def ace_to_algebraic(ace: int) -> str:
    if ace < 100:
        r, c = divmod(ace, 9)
        return f"{chr(ord('a') + c)}{9 - r}"
    if ace < 200:
        slot = ace - 100
        r, c = divmod(slot, 8)
        return f"{chr(ord('a') + c)}{8 - r}h"
    slot = ace - 200
    r, c = divmod(slot, 8)
    return f"{chr(ord('a') + c)}{8 - r}v"


def pack_moves(moves: list[str]) -> bytes:
    """2 bytes little-endian ACE code per move."""
    return b"".join(struct.pack("<H", algebraic_to_ace(m)) for m in moves)


def unpack_moves(data: bytes) -> list[str]:
    if len(data) % 2:
        raise ValueError("moves_bin length must be even")
    out: list[str] = []
    for i in range(0, len(data), 2):
        (ace,) = struct.unpack_from("<H", data, i)
        out.append(ace_to_algebraic(ace))
    return out


def moves_from_row(moves_text: str | None, moves_bin: bytes | None) -> list[str]:
    if moves_bin:
        return unpack_moves(moves_bin)
    if moves_text:
        return moves_text.split()
    return []
