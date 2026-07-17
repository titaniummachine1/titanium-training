"""Explicit bounded defaults for a supervised oracle-horizon pilot."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PilotConfig:
    max_candidate_positions: int = 10_000
    bands: tuple[int, ...] = (0, 1, 2, 3)
    retain_only_exact_or_backed: bool = True
    curriculum_mix: float = 0.10
    one_ema_continuation: bool = True
    unattended_repeat: bool = False
    book_mode_training: str = "off"
    max_generated_games: int = 100
    max_deep_search_cpu_hours: float = 4.0
    max_retained_rows: int = 10_000
    screen_games: int = 20
    full_gate_games: int = 100
    pause_on_any_safety: bool = True

    def __post_init__(self) -> None:
        if not 5_000 <= self.max_candidate_positions <= 20_000:
            raise ValueError("max_candidate_positions must be within 5,000..20,000")
        if self.unattended_repeat:
            raise ValueError("unattended_repeat must remain False for this pilot")
        if self.book_mode_training != "off":
            raise ValueError("book_mode_training must be off")
