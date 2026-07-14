"""Registry of real-work entry points and expected prep guards (audit only)."""
from __future__ import annotations

from dataclasses import dataclass

from prep_guard import BLOCKED_CATEGORIES


@dataclass(frozen=True)
class EntryPointSpec:
    path: str
    category: str
    guarded: bool


# guarded=True means guard_real_work() is wired in that module's main().
REAL_WORK_ENTRY_POINTS: tuple[EntryPointSpec, ...] = (
    EntryPointSpec("training/local_game_pool.py", "local_self_play_pool", True),
    EntryPointSpec("training/continuous_pool.py", "corpus_generation", True),
    EntryPointSpec("training/self_play_overnight.py", "corpus_generation", True),
    EntryPointSpec("training/training_coordinator.py", "optimizer_training", True),
    EntryPointSpec("training/oracle_importer.py", "labeling", True),
    EntryPointSpec("training/oracle_game_factory/server.py", "oracle_factory", True),
    EntryPointSpec("training/db_import.py", "labeling", True),
    EntryPointSpec("training/deploy_accepted_to_website.py", "deployment", True),
    EntryPointSpec("training/strength_gate.py", "candidate_gating", True),
    EntryPointSpec("training/titanium_training/training/trainer.py", "optimizer_training", True),
    EntryPointSpec("training/titanium_training/cli.py", "optimizer_training", True),
    EntryPointSpec("training/nnue_cli.py", "optimizer_training", True),
    EntryPointSpec("training/collect_zeroink.py", "labeling", True),
    EntryPointSpec("training/build_feature_cache.py", "dataset_finalization", True),
    EntryPointSpec("training/build_cache_from_labels_db.py", "dataset_finalization", True),
    EntryPointSpec("training/score_out_labels.py", "labeling", True),
    EntryPointSpec("training/oracle_endgame_labels.py", "labeling", True),
    EntryPointSpec("training/tools/datagen/datagen.py", "corpus_generation", True),
    EntryPointSpec("training/sync_overnight_to_teacher.py", "dataset_finalization", True),
    EntryPointSpec("training/retry_failed_training.py", "optimizer_training", True),
    EntryPointSpec("training/tools/ka_teacher/ka_nn_collect_labels.py", "labeling", True),
    EntryPointSpec("training/tools/ka_teacher/ka_ab_collect_labels.py", "labeling", True),
    EntryPointSpec("training/match_eval.py", "candidate_gating", True),
    EntryPointSpec("training/ka_match.py", "candidate_gating", True),
)

UNGUARDED_FORBIDDEN = tuple(
    ep for ep in REAL_WORK_ENTRY_POINTS if ep.category in BLOCKED_CATEGORIES and not ep.guarded
)
