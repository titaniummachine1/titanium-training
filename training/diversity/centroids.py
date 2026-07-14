"""Training-only seeded opening centroids (fixture bank + selection logic)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from diversity.canonical import reflection_canonical_position_key

# Evaluation batteries — must never appear as training seeds.
EVAL_BATTERY_IDS = frozenset({"theory-24", "frozen_gate_battery"})


@dataclass(frozen=True)
class OpeningSeed:
    seed_id: str
    moves: tuple[str, ...]
    mirrored_canonical_id: str
    two_ply_class: str
    eval_battery_disjoint: bool = True

    def __post_init__(self) -> None:
        if not self.moves:
            raise ValueError("seed must have at least one move")
        if not self.eval_battery_disjoint:
            raise ValueError("training seed overlaps evaluation battery")


def two_ply_class_from_moves(moves: tuple[str, ...]) -> str:
    """Opening-class key for diversity accounting (fixture: up to 4 plies)."""
    if len(moves) < 2:
        return f"partial:{moves[0] if moves else 'empty'}"
    if len(moves) >= 4:
        from diversity.canonical import reflection_canonical_four_ply

        return reflection_canonical_four_ply(moves[:4])
    return reflection_canonical_position_key({"moves_prefix": f"{moves[0]} {moves[1]}"})


def build_fixture_seed_bank() -> tuple[OpeningSeed, ...]:
    """Synthetic seeds spanning >16 opening classes (prep fixtures only)."""
    seeds: list[OpeningSeed] = []
    white = ("d2", "e2", "f2")
    black = ("d8", "e8", "f8")
    thirds = ("e3", "d3", "f3", "c3", "g3", "e4", "d4", "f4")
    fourths = ("e7", "d7", "f7", "c7")
    idx = 0
    for w in white:
        for b in black:
            for third in thirds:
                for fourth in fourths:
                    moves = (w, b, third, fourth)
                    seed_id = f"train-seed-{idx:04d}"
                    seeds.append(
                        OpeningSeed(
                            seed_id=seed_id,
                            moves=moves,
                            mirrored_canonical_id=reflection_canonical_position_key(
                                {"seed_id": seed_id, "moves": " ".join(moves)}
                            ),
                            two_ply_class=two_ply_class_from_moves(moves),
                        )
                    )
                    idx += 1
                    if idx >= 32:
                        return tuple(seeds)
    if len({s.two_ply_class for s in seeds}) < 16:
        raise RuntimeError("fixture seed bank must provide >=16 opening classes")
    return tuple(seeds)


def select_seed(seed_id: str, bank: tuple[OpeningSeed, ...] | None = None) -> OpeningSeed:
    bank = bank or build_fixture_seed_bank()
    for seed in bank:
        if seed.seed_id == seed_id:
            return seed
    raise KeyError(seed_id)


def iter_seeds(bank: tuple[OpeningSeed, ...] | None = None) -> Iterator[OpeningSeed]:
    yield from bank or build_fixture_seed_bank()
