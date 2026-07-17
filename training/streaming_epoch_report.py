"""Per-epoch report for streaming NNUE training."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_TRAINING = Path(__file__).resolve().parent
LOG_DIR = _TRAINING / "data" / "overnight_logs"
REPORT_DIR = LOG_DIR / "epoch_reports"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_epoch_report(epoch: int, payload: dict[str, Any]) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / f"epoch_{epoch:04d}.json"
    doc = {"epoch": epoch, "recorded_at": _utc_now(), **payload}
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    latest = LOG_DIR / "latest_epoch_report.json"
    latest.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return path


def usage_distribution(con) -> dict[str, int]:
    rows = con.execute(
        """
        SELECT training_visits, COUNT(*) FROM position_usage
        GROUP BY training_visits ORDER BY training_visits
        """
    ).fetchall()
    return {str(int(v)): int(c) for v, c in rows}
