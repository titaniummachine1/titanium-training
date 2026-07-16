from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

POSITION_SCHEMA_VERSION = 1
MOVE_SCHEMA_VERSION = 1

BOARD_SIZE = 9
WALL_GRID_SIZE = 8
WALLS_PER_PLAYER = 10
CELL_COUNT = BOARD_SIZE * BOARD_SIZE
WALL_SLOT_COUNT = WALL_GRID_SIZE * WALL_GRID_SIZE

HORIZONTAL_WALL_BASE = 0
VERTICAL_WALL_BASE = 64
PAWN_MOVE_BASE = 128

PAWN_NORTH = 128
PAWN_SOUTH = 129
PAWN_EAST = 130
PAWN_WEST = 131
PAWN_NORTHEAST = 132
PAWN_NORTHWEST = 133
PAWN_SOUTHEAST = 134
PAWN_SOUTHWEST = 135

VALID_MOVE_CODES = frozenset(range(136))

START_P0_CELL = 4
START_P1_CELL = 76


def col_to_file(col: int) -> str:
    if not 0 <= col < BOARD_SIZE:
        raise ValueError(f"column out of range: {col}")
    return chr(ord("a") + col)


def file_to_col(file_char: str) -> int:
    if len(file_char) != 1 or not ("a" <= file_char <= "i"):
        raise ValueError(f"bad file: {file_char!r}")
    return ord(file_char) - ord("a")


def cell_to_coords(cell: int) -> tuple[int, int]:
    if not 0 <= cell < CELL_COUNT:
        raise ValueError(f"cell out of range: {cell}")
    return divmod(cell, BOARD_SIZE)


def coords_to_cell(row: int, col: int) -> int:
    if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
        raise ValueError(f"bad coordinates: {(row, col)}")
    return row * BOARD_SIZE + col


def cell_to_notation(cell: int) -> str:
    row, col = cell_to_coords(cell)
    return f"{col_to_file(col)}{row + 1}"


def notation_to_cell(text: str) -> int:
    if len(text) != 2:
        raise ValueError(f"bad square: {text!r}")
    col = file_to_col(text[0])
    if text[1] not in "123456789":
        raise ValueError(f"bad rank: {text!r}")
    row = int(text[1]) - 1
    return coords_to_cell(row, col)


def wall_slot_to_notation(slot: int, horizontal: bool) -> str:
    if not 0 <= slot < WALL_SLOT_COUNT:
        raise ValueError(f"wall slot out of range: {slot}")
    row, col = divmod(slot, WALL_GRID_SIZE)
    suffix = "h" if horizontal else "v"
    return f"{col_to_file(col)}{row + 1}{suffix}"


def notation_to_wall_slot(text: str) -> tuple[int, bool]:
    if len(text) != 3 or text[2] not in ("h", "v"):
        raise ValueError(f"bad wall move: {text!r}")
    col = file_to_col(text[0])
    if text[1] not in "12345678":
        raise ValueError(f"bad wall rank: {text!r}")
    row = int(text[1]) - 1
    slot = row * WALL_GRID_SIZE + col
    return slot, text[2] == "h"


def wall_code_to_notation(code: int) -> str:
    if 0 <= code < 64:
        return wall_slot_to_notation(code, True)
    if 64 <= code < 128:
        return wall_slot_to_notation(code - 64, False)
    raise ValueError(f"wall code out of range: {code}")


def wall_notation_to_code(text: str) -> int:
    slot, horizontal = notation_to_wall_slot(text)
    return slot if horizontal else 64 + slot


def direction_code_from_delta(dr: int, dc: int) -> int:
    if dc == 0 and dr > 0:
        return PAWN_NORTH
    if dc == 0 and dr < 0:
        return PAWN_SOUTH
    if dr == 0 and dc > 0:
        return PAWN_EAST
    if dr == 0 and dc < 0:
        return PAWN_WEST
    if dr > 0 and dc > 0:
        return PAWN_NORTHEAST
    if dr > 0 and dc < 0:
        return PAWN_NORTHWEST
    if dr < 0 and dc > 0:
        return PAWN_SOUTHEAST
    if dr < 0 and dc < 0:
        return PAWN_SOUTHWEST
    raise ValueError(f"delta does not map to pawn direction: {(dr, dc)}")


def code_to_direction_name(code: int) -> str:
    names = {
        PAWN_NORTH: "north",
        PAWN_SOUTH: "south",
        PAWN_EAST: "east",
        PAWN_WEST: "west",
        PAWN_NORTHEAST: "northeast",
        PAWN_NORTHWEST: "northwest",
        PAWN_SOUTHEAST: "southeast",
        PAWN_SOUTHWEST: "southwest",
    }
    if code not in names:
        raise ValueError(f"not a pawn code: {code}")
    return names[code]


@dataclass(frozen=True)
class PositionState:
    player0_cell: int = START_P0_CELL
    player1_cell: int = START_P1_CELL
    player0_walls: int = WALLS_PER_PLAYER
    player1_walls: int = WALLS_PER_PLAYER
    horizontal_walls: int = 0
    vertical_walls: int = 0
    side_to_move: int = 0

    @staticmethod
    def initial() -> "PositionState":
        return PositionState()

    def validate(self, *, require_paths: bool = True) -> None:
        if not 0 <= self.player0_cell < CELL_COUNT:
            raise ValueError("player0 cell out of range")
        if not 0 <= self.player1_cell < CELL_COUNT:
            raise ValueError("player1 cell out of range")
        if self.player0_cell == self.player1_cell:
            raise ValueError("both pawns occupy the same cell")
        if not 0 <= self.player0_walls <= WALLS_PER_PLAYER:
            raise ValueError("player0 walls out of range")
        if not 0 <= self.player1_walls <= WALLS_PER_PLAYER:
            raise ValueError("player1 walls out of range")
        if self.side_to_move not in (0, 1):
            raise ValueError("side_to_move must be 0 or 1")
        if self.horizontal_walls < 0 or self.vertical_walls < 0:
            raise ValueError("wall masks must be nonnegative")
        if self.horizontal_walls >> 64 or self.vertical_walls >> 64:
            raise ValueError("wall masks must fit in 64 bits")
        _validate_wall_mask(self.horizontal_walls, horizontal=True)
        _validate_wall_mask(self.vertical_walls, horizontal=False)
        _validate_cross_and_neighbor_collisions(self.horizontal_walls, self.vertical_walls)
        if require_paths:
            if not both_players_reach_goals(self):
                raise ValueError("wall layout disconnects a player from goal")

    @property
    def current_cell(self) -> int:
        return self.player0_cell if self.side_to_move == 0 else self.player1_cell

    def player_cell(self, player: int) -> int:
        return self.player0_cell if player == 0 else self.player1_cell

    def player_walls_left(self, player: int) -> int:
        return self.player0_walls if player == 0 else self.player1_walls

    def with_move(self, move: str) -> "PositionState":
        return apply_move(self, move)

    def packed_state(self) -> bytes:
        head = bytes(
            [
                POSITION_SCHEMA_VERSION,
                self.player0_cell,
                self.player1_cell,
                self.player0_walls,
                self.player1_walls,
                self.side_to_move,
                0,
                0,
            ]
        )
        return (
            head
            + int(self.horizontal_walls).to_bytes(8, "little", signed=False)
            + int(self.vertical_walls).to_bytes(8, "little", signed=False)
        )

    @staticmethod
    def unpack_state(data: bytes) -> "PositionState":
        if len(data) != 24:
            raise ValueError(f"packed state must be 24 bytes, got {len(data)}")
        version = data[0]
        if version != POSITION_SCHEMA_VERSION:
            raise ValueError(f"unsupported position schema version: {version}")
        return PositionState(
            player0_cell=data[1],
            player1_cell=data[2],
            player0_walls=data[3],
            player1_walls=data[4],
            side_to_move=data[5],
            horizontal_walls=int.from_bytes(data[8:16], "little", signed=False),
            vertical_walls=int.from_bytes(data[16:24], "little", signed=False),
        )

    def canonical_hash(self) -> bytes:
        return hashlib.sha256(self.packed_state()).digest()

    def fast_hash(self) -> int:
        return int.from_bytes(hashlib.blake2b(self.packed_state(), digest_size=8).digest(), "little")

    def terminal_winner(self) -> int | None:
        p0_row, _ = cell_to_coords(self.player0_cell)
        p1_row, _ = cell_to_coords(self.player1_cell)
        if p0_row == 8:
            return 0
        if p1_row == 0:
            return 1
        return None


def _validate_wall_mask(mask: int, *, horizontal: bool) -> None:
    for slot in iter_wall_slots(mask):
        row, col = divmod(slot, WALL_GRID_SIZE)
        if not (0 <= row < WALL_GRID_SIZE and 0 <= col < WALL_GRID_SIZE):
            ori = "horizontal" if horizontal else "vertical"
            raise ValueError(f"{ori} wall slot out of range: {slot}")


def _validate_cross_and_neighbor_collisions(horizontal_mask: int, vertical_mask: int) -> None:
    h_slots = set(iter_wall_slots(horizontal_mask))
    v_slots = set(iter_wall_slots(vertical_mask))
    for slot in h_slots:
        row, col = divmod(slot, WALL_GRID_SIZE)
        if slot - 1 in h_slots and col > 0:
            raise ValueError("neighboring horizontal walls overlap illegally")
        if slot + 1 in h_slots and col < 7:
            raise ValueError("neighboring horizontal walls overlap illegally")
        if slot in v_slots:
            raise ValueError("horizontal/vertical walls cross at same anchor")
    for slot in v_slots:
        row, col = divmod(slot, WALL_GRID_SIZE)
        if slot - 8 in v_slots and row > 0:
            raise ValueError("neighboring vertical walls overlap illegally")
        if slot + 8 in v_slots and row < 7:
            raise ValueError("neighboring vertical walls overlap illegally")


def iter_wall_slots(mask: int) -> Iterable[int]:
    bits = mask
    while bits:
        lsb = bits & -bits
        slot = lsb.bit_length() - 1
        yield slot
        bits ^= lsb


def horizontal_wall_present(state: PositionState, js_row: int, col0: int) -> bool:
    if js_row < 1 or js_row > 8 or col0 < 0 or col0 >= 8:
        return False
    bit = (js_row - 1) * 8 + col0
    return ((state.horizontal_walls >> bit) & 1) != 0


def vertical_wall_present(state: PositionState, js_row: int, col0: int) -> bool:
    if js_row < 1 or js_row > 8 or col0 < 0 or col0 >= 8:
        return False
    bit = (js_row - 1) * 8 + col0
    return ((state.vertical_walls >> bit) & 1) != 0


def pawn_can_move(state: PositionState, cell: int, dr: int, dc: int) -> bool:
    row, col = cell_to_coords(cell)
    nr = row + dr
    nc = col + dc
    if nr < 0 or nr > 8 or nc < 0 or nc > 8:
        return False
    js_from = row + 1
    js_to = nr + 1
    if dr == 1 and dc == 0:
        return not horizontal_wall_present(state, js_from, col) and (
            col == 0 or not horizontal_wall_present(state, js_from, col - 1)
        )
    if dr == -1 and dc == 0:
        return not horizontal_wall_present(state, js_to, col) and (
            col == 0 or not horizontal_wall_present(state, js_to, col - 1)
        )
    if dr == 0 and dc == 1:
        return not vertical_wall_present(state, js_from, col) and not vertical_wall_present(
            state, row, col
        )
    if dr == 0 and dc == -1:
        return not vertical_wall_present(state, js_to, nc) and not vertical_wall_present(
            state, nr, nc
        )
    return False


def valid_pawn_destinations(state: PositionState) -> list[int]:
    me = state.side_to_move
    current = state.player_cell(me)
    opponent = state.player_cell(1 - me)
    moves: list[int] = []
    for dr, dc in ((1, 0), (0, 1), (-1, 0), (0, -1)):
        if not pawn_can_move(state, current, dr, dc):
            continue
        row, col = cell_to_coords(current)
        step = coords_to_cell(row + dr, col + dc)
        if step != opponent:
            moves.append(step)
            continue
        if pawn_can_move(state, opponent, dr, dc):
            orow, ocol = cell_to_coords(opponent)
            moves.append(coords_to_cell(orow + dr, ocol + dc))
            continue
        for sdr, sdc in ((0, -1), (0, 1)) if dr else ((1, 0), (-1, 0)):
            if pawn_can_move(state, opponent, sdr, sdc):
                orow, ocol = cell_to_coords(opponent)
                diag = coords_to_cell(orow + sdr, ocol + sdc)
                if diag != current:
                    moves.append(diag)
    return moves


def _lee_direction_masks(state: PositionState) -> tuple[int, int, int, int]:
    """Source-cell masks for N/S/E/W bit-parallel Lee expansion."""
    north = south = east = west = 0
    for cell in range(CELL_COUNT):
        bit = 1 << cell
        if pawn_can_move(state, cell, 1, 0):
            north |= bit
        if pawn_can_move(state, cell, -1, 0):
            south |= bit
        if pawn_can_move(state, cell, 0, 1):
            east |= bit
        if pawn_can_move(state, cell, 0, -1):
            west |= bit
    return north, south, east, west


def _flood_reachable_bits(start_cell: int, masks: tuple[int, int, int, int]) -> int:
    """Binary flood fill: one integer frontier per exact Lee wave."""
    north, south, east, west = masks
    playable = (1 << CELL_COUNT) - 1
    reached = frontier = 1 << start_cell
    while frontier:
        expanded = (
            ((frontier & north) << BOARD_SIZE)
            | ((frontier & south) >> BOARD_SIZE)
            | ((frontier & east) << 1)
            | ((frontier & west) >> 1)
        )
        frontier = expanded & ~reached & playable
        reached |= frontier
    return reached


def _goal_row_reachable(reachable: int, player: int) -> bool:
    goal_row = 8 if player == 0 else 0
    goal_mask = ((1 << BOARD_SIZE) - 1) << (goal_row * BOARD_SIZE)
    return bool(reachable & goal_mask)


def both_players_reach_goals(state: PositionState) -> bool:
    masks = _lee_direction_masks(state)
    white_reach = _flood_reachable_bits(state.player0_cell, masks)
    if not _goal_row_reachable(white_reach, 0):
        return False
    if white_reach & (1 << state.player1_cell):
        return _goal_row_reachable(white_reach, 1)
    black_reach = _flood_reachable_bits(state.player1_cell, masks)
    return _goal_row_reachable(black_reach, 1)


def collides_with_existing_wall(state: PositionState, slot: int, horizontal: bool) -> bool:
    row, col = divmod(slot, WALL_GRID_SIZE)
    if horizontal:
        if ((state.horizontal_walls >> slot) & 1) or ((state.vertical_walls >> slot) & 1):
            return True
        if col > 0 and ((state.horizontal_walls >> (slot - 1)) & 1):
            return True
        if col < 7 and ((state.horizontal_walls >> (slot + 1)) & 1):
            return True
        return False
    if ((state.vertical_walls >> slot) & 1) or ((state.horizontal_walls >> slot) & 1):
        return True
    if row > 0 and ((state.vertical_walls >> (slot - 8)) & 1):
        return True
    if row < 7 and ((state.vertical_walls >> (slot + 8)) & 1):
        return True
    return False


def _touching_wall_candidates(slot: int, horizontal: bool) -> tuple[set[int], set[int], set[int], tuple[bool, bool]]:
    row, col = divmod(slot, WALL_GRID_SIZE)
    if horizontal:
        side_a = {64 + row * 8 + col}
        side_b = {64 + row * 8 + col + 1}
        if row < 7:
            side_a.add(64 + (row + 1) * 8 + col)
            side_b.add(64 + (row + 1) * 8 + col + 1)
        if row > 0:
            side_a.add(64 + (row - 1) * 8 + col)
            side_b.add(64 + (row - 1) * 8 + col + 1)
        side_a.add(slot - 1 if col > 0 else slot)
        side_b.add(slot + 1 if col < 7 else slot)
        middle = set()
        if row < 7:
            middle.add(64 + (row + 1) * 8 + col)
        if row > 0:
            middle.add(64 + (row - 1) * 8 + col)
        return side_a, side_b, middle, (col == 0, col == 7)
    side_a = {slot - 8 if row > 0 else slot}
    side_b = {slot + 8 if row < 7 else slot}
    side_a.update({row * 8 + col})
    side_b.update({(row + 1) * 8 + col if row < 7 else row * 8 + col})
    if col > 0:
        side_a.add(row * 8 + col - 1)
        side_b.add((row + 1) * 8 + col - 1 if row < 7 else row * 8 + col - 1)
    if col < 7:
        side_a.add(row * 8 + col + 1)
        side_b.add((row + 1) * 8 + col + 1 if row < 7 else row * 8 + col + 1)
    middle = set()
    if col > 0:
        middle.add(row * 8 + col - 1)
    if col < 7:
        middle.add(row * 8 + col + 1)
    return side_a, side_b, middle, (row == 7, row == 0)


def can_wall_block(state: PositionState, slot: int, horizontal: bool) -> bool:
    side_a, side_b, middle, edges = _touching_wall_candidates(slot, horizontal)
    occupied = state.horizontal_walls | (state.vertical_walls << 64)
    has_side_a = edges[0] or any((occupied >> idx) & 1 for idx in side_a if 0 <= idx < 128)
    has_side_b = edges[1] or any((occupied >> idx) & 1 for idx in side_b if 0 <= idx < 128)
    has_middle = any((occupied >> idx) & 1 for idx in middle if 0 <= idx < 128)
    return (has_side_a and has_side_b) or (has_side_a and has_middle) or (has_side_b and has_middle)


def is_valid_wall_placement(state: PositionState, slot: int, horizontal: bool) -> bool:
    if state.player_walls_left(state.side_to_move) <= 0:
        return False
    if collides_with_existing_wall(state, slot, horizontal):
        return False
    if not can_wall_block(state, slot, horizontal):
        return True
    move = wall_slot_to_notation(slot, horizontal)
    next_state = apply_move(state, move, assume_legal=True)
    return both_players_reach_goals(next_state)


def legal_wall_codes(state: PositionState) -> list[int]:
    codes: list[int] = []
    for slot in range(WALL_SLOT_COUNT):
        if is_valid_wall_placement(state, slot, True):
            codes.append(slot)
        if is_valid_wall_placement(state, slot, False):
            codes.append(64 + slot)
    return codes


def encode_move(state: PositionState, move: str) -> int:
    move = move.strip().lower()
    if len(move) == 3:
        code = wall_notation_to_code(move)
        horizontal = code < 64
        slot = code if horizontal else code - 64
        if not is_valid_wall_placement(state, slot, horizontal):
            raise ValueError(f"illegal wall move from state: {move}")
        return code
    if len(move) != 2:
        raise ValueError(f"bad move notation: {move!r}")
    target = notation_to_cell(move)
    legal = valid_pawn_destinations(state)
    if target not in legal:
        raise ValueError(f"illegal pawn move from state: {move}")
    from_row, from_col = cell_to_coords(state.current_cell)
    to_row, to_col = cell_to_coords(target)
    return direction_code_from_delta(to_row - from_row, to_col - from_col)


def decode_move(state: PositionState, code: int) -> str:
    if code not in VALID_MOVE_CODES:
        raise ValueError(f"unsupported move code: {code}")
    if code < 128:
        return wall_code_to_notation(code)
    from_cell = state.current_cell
    from_row, from_col = cell_to_coords(from_cell)
    candidates = []
    for target in valid_pawn_destinations(state):
        to_row, to_col = cell_to_coords(target)
        if direction_code_from_delta(to_row - from_row, to_col - from_col) == code:
            candidates.append(target)
    if len(candidates) != 1:
        raise ValueError(
            f"pawn code {code} is ambiguous or illegal in state: {[cell_to_notation(c) for c in candidates]}"
        )
    return cell_to_notation(candidates[0])


def apply_move(state: PositionState, move: str, *, assume_legal: bool = False) -> PositionState:
    move = move.strip().lower()
    if not assume_legal:
        _ = encode_move(state, move)
    if len(move) == 3:
        slot, horizontal = notation_to_wall_slot(move)
        if horizontal:
            horizontal_walls = state.horizontal_walls | (1 << slot)
            vertical_walls = state.vertical_walls
        else:
            horizontal_walls = state.horizontal_walls
            vertical_walls = state.vertical_walls | (1 << slot)
        if state.side_to_move == 0:
            next_state = PositionState(
                player0_cell=state.player0_cell,
                player1_cell=state.player1_cell,
                player0_walls=state.player0_walls - 1,
                player1_walls=state.player1_walls,
                horizontal_walls=horizontal_walls,
                vertical_walls=vertical_walls,
                side_to_move=1,
            )
        else:
            next_state = PositionState(
                player0_cell=state.player0_cell,
                player1_cell=state.player1_cell,
                player0_walls=state.player0_walls,
                player1_walls=state.player1_walls - 1,
                horizontal_walls=horizontal_walls,
                vertical_walls=vertical_walls,
                side_to_move=0,
            )
        return next_state
    target = notation_to_cell(move)
    if state.side_to_move == 0:
        return PositionState(
            player0_cell=target,
            player1_cell=state.player1_cell,
            player0_walls=state.player0_walls,
            player1_walls=state.player1_walls,
            horizontal_walls=state.horizontal_walls,
            vertical_walls=state.vertical_walls,
            side_to_move=1,
        )
    return PositionState(
        player0_cell=state.player0_cell,
        player1_cell=target,
        player0_walls=state.player0_walls,
        player1_walls=state.player1_walls,
        horizontal_walls=state.horizontal_walls,
        vertical_walls=state.vertical_walls,
        side_to_move=0,
    )


def replay_game(moves: Iterable[str], start: PositionState | None = None) -> list[PositionState]:
    states = [start or PositionState.initial()]
    for move in moves:
        states.append(apply_move(states[-1], move))
    return states


def moves_to_u8_blob(moves: Iterable[str], start: PositionState | None = None) -> bytes:
    state = start or PositionState.initial()
    out = bytearray()
    for move in moves:
        code = encode_move(state, move)
        out.append(code)
        state = apply_move(state, move)
    return bytes(out)


def moves_from_u8_blob(blob: bytes, start: PositionState | None = None) -> list[str]:
    state = start or PositionState.initial()
    out: list[str] = []
    for code in blob:
        move = decode_move(state, code)
        out.append(move)
        state = apply_move(state, move)
    return out
