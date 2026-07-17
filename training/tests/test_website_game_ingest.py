from __future__ import annotations

from oracle_game_factory.protocol import parse_website_game_wire
from titanium_training.store.website_games import FinishedWebsiteGame, parse_text_wire


def test_parse_text_wire_roundtrip() -> None:
    wire = "TI-GAME-1\nresult=-1\nsource=website_finished_game\nmoves=e2 e8 e3 e7 e4 e6\n"
    parsed = parse_text_wire(wire)
    assert parsed == FinishedWebsiteGame(
        moves=("e2", "e8", "e3", "e7", "e4", "e6"),
        result=-1,
        source="website_finished_game",
    )


def test_oracle_protocol_accepts_text_wire() -> None:
    wire = b"TI-GAME-1\nresult=1\nsource=website_finished_game\nmoves=e2 e8 e3 e7\n"
    payload = parse_website_game_wire(wire)
    assert payload == {
        "moves": ["e2", "e8", "e3", "e7"],
        "result": 1,
        "source": "website_finished_game",
    }
