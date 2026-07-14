#!/usr/bin/env python3
"""Pre-start report and launch for continuous streaming NNUE (existing architecture)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_TRAINING = _REPO / "training"
sys.path.insert(0, str(_TRAINING))

from db_import import GAMES_DB_PATH, LABELS_DB_PATH
from game_opening_gate import TRAINING_OPENING_MIN_PREFIX, WHITE_OPENING_PAWNS, BLACK_OPENING_PAWNS
from generation_matchup import MATCHUP_PRIOR_EPOCH, MATCHUP_SELFPLAY
from streaming_checkpoint_chain import (
    ENGINE_WEIGHTS,
    INVALID_WEIGHT_PATHS,
    ensure_epoch_zero,
    sha256_file,
)

LOG_DIR = _TRAINING / "data" / "overnight_logs"
REPORT_PATH = LOG_DIR / "streaming_prestart_report.json"


def main() -> int:
    chain = ensure_epoch_zero()
    deployed_sha = sha256_file(ENGINE_WEIGHTS)
    report = {
        "architecture": "database-first split runtime (reuse, no parallel pipeline)",
        "collector": "training/local_game_pool.py -> continuous_pool.py -> self_play_overnight.py",
        "game_store": str(GAMES_DB_PATH),
        "teacher_position_store": str(LABELS_DB_PATH),
        "position_usage": "training/position_usage_db.py (training_visits, MAX=5, retired_replay_count)",
        "scheduler": "training/training_coordinator.py (TRIGGER_THRESHOLD=2048, claim_training_trigger)",
        "trainer": "training/titanium_training/training/trainer.py --labels-db --defer-usage-commit",
        "opponent_pool": [MATCHUP_SELFPLAY, MATCHUP_PRIOR_EPOCH],
        "prior_epoch_fraction": os.environ.get("STREAM_PRIOR_EPOCH_FRACTION", "0.30"),
        "mixed_opponent_fraction": f"{1.0 - float(os.environ.get('STREAM_PRIOR_EPOCH_FRACTION', '0.30')):.2f}",
        "generation_engine": os.environ.get("TITANIUM_GENERATION_ENGINE", "titanium-v17"),
        "opening_book_mode": "embedded book (TITANIUM_BOOK_DB); DIVERSITY_SPEC_V1 — no move-selection temperature",
        "training_opening_gate": (
            f"central pawn plies 0-1 in {sorted(WHITE_OPENING_PAWNS)} x "
            f"{sorted(BLACK_OPENING_PAWNS)}; canonical centroid {' '.join(TRAINING_OPENING_MIN_PREFIX)}"
        ),
        "deploy_collapse_check": "opening_sanity.py: promoted net must play e2 e8 e3 e7 (eval/deploy only)",
        "lmr_config": "titanium-v16 grafted: ACE v13 graduated LMR, dead-tail walls depth-1, backward moves depth-1, full-depth re-search on alpha raise",
        "epoch_sample_size": os.environ.get("STREAM_EPOCH_SIZE", "100000"),
        "retired_replay_fraction": os.environ.get("STREAM_RETIRED_REPLAY_FRACTION", "0.05"),
        "trigger_persistence": "labels.db training_trigger_state.pending_new_eligible + claim_training_trigger overflow",
        "usage_commit": "trainer --defer-usage-commit -> position_usage_db.commit_epoch_training_visits after ckpt save",
        "failed_epoch_rollback": "no training_visits bump on train failure; quarantined weights restored from accepted chain",
        "valid_starting_checkpoint": {
            "path": str(ENGINE_WEIGHTS),
            "sha256": deployed_sha,
        },
        "invalid_excluded_checkpoints": [str(p) for p in INVALID_WEIGHT_PATHS],
        "accepted_chain": chain,
        "files_changed_this_session": [
            "training/streaming_checkpoint_chain.py",
            "training/generation_matchup.py",
            "training/game_opening_gate.py",
            "training/streaming_epoch_validation.py",
            "training/streaming_epoch_report.py",
            "training/training_coordinator.py",
            "training/continuous_pool.py",
            "training/self_play_overnight.py",
            "training/position_usage_db.py",
            "training/titanium_training/training/trainer.py",
            "training/tools/start_streaming_nnue.ps1",
            "training/tools/start_local_game_pool_detached.ps1",
            "training/tools/start_training_coordinator_detached.ps1",
        ],
        "database_migrations": "ALTER position_usage ADD retired_replay_count if missing (runtime ensure_schema)",
        "auto_deploy": False,
    }
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if "--launch" in sys.argv:
        ps1 = _TRAINING / "tools" / "start_streaming_nnue.ps1"
        return subprocess.call(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ps1), "-SkipReport"],
            cwd=str(_REPO),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
