"""Train/validation split for streaming DB epochs — no position-hash leakage."""
from __future__ import annotations

from pathlib import Path

from cache_val_split import build_val_position_keys


def _canonical_hex(pos_key: str) -> str:
    if pos_key.startswith("json:"):
        return pos_key[5:]
    if pos_key.startswith("teacher:"):
        return pos_key[8:]
    return pos_key


def split_streaming_epoch_keys(
    pos_keys: list[str],
    *,
    labels_db: Path,
    val_fraction: float = 0.05,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Assign whole canonical positions to train or val (never both)."""
    if not pos_keys:
        return [], []

    games_db = labels_db.with_name("games.db")
    manifest_dir = labels_db.parent / "streaming_splits"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    ordered_hex = [_canonical_hex(k) for k in pos_keys]
    val_hex, _manifest = build_val_position_keys(
        cache_dir=manifest_dir,
        position_keys_in_order=ordered_hex,
        games_db=games_db,
        val_fraction=val_fraction,
        seed=seed,
    )

    train_keys: list[str] = []
    val_keys: list[str] = []
    for key in pos_keys:
        if _canonical_hex(key) in val_hex:
            val_keys.append(key)
        else:
            train_keys.append(key)

    # build_val_position_keys owns the deterministic small-cohort fallback and
    # persists it.  Do not substitute this epoch's first row here: that would
    # make the same canonical position change splits between epochs.
    return train_keys, val_keys
