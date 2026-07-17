"""Oracle-only Titanium self-play game factory.

This package intentionally contains no training authority.  It can generate,
spool, serve, and acknowledge self-play games; laptop-side code is responsible
for canonical import, teacher sync, training, validation, and promotion.
"""

PROTOCOL_VERSION = "titanium-oracle-game-factory/1"
GAME_SCHEMA_VERSION = "titanium-oracle-game/1"
WEIGHT_SCHEMA = "halfpw-sparse-route5-catheat-ws20-cat-v2"
