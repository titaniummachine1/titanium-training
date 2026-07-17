#!/usr/bin/env python3
"""Mark feature caches built under the old label convention as stale."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TRAINING = REPO / "training"
sys.path.insert(0, str(TRAINING))

from label_perspective import LABEL_PERSPECTIVE_CONVENTION

CACHE_DIRS = [
    TRAINING / "data" / "feature_cache",
    TRAINING / "data" / "feature_cache_wallz",
]


def main() -> int:
    ts = datetime.now(timezone.utc).isoformat()
    for cache_dir in CACHE_DIRS:
        if not cache_dir.is_dir():
            continue
        meta = cache_dir / "meta.json"
        stale = {
            "stale": True,
            "reason": "label_perspective_pre_dataset_stm_unchanged_v1",
            "required_convention": LABEL_PERSPECTIVE_CONVENTION,
            "invalidated_at": ts,
        }
        (cache_dir / "STALE_LABEL_PERSPECTIVE.json").write_text(
            json.dumps(stale, indent=2) + "\n", encoding="utf-8"
        )
        if meta.is_file():
            data = json.loads(meta.read_text(encoding="utf-8"))
            data["stale"] = True
            data["stale_reason"] = stale["reason"]
            meta.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        print(f"invalidated {cache_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
