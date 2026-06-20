#!/usr/bin/env python3
"""One-shot training/ root relayout: git mv + import rewrites. Low resource use."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TR = ROOT / "training"

# (src relative to training/, dst relative to training/)
MOVES: list[tuple[str, str]] = [
    # tests
    ("conftest.py", "tests/conftest.py"),
    ("test_color_rotation.py", "tests/test_color_rotation.py"),
    ("test_evidence_canonical.py", "tests/test_evidence_canonical.py"),
    ("test_lmr_head_v3.py", "tests/test_lmr_head_v3.py"),
    ("test_opponent_curriculum.py", "tests/test_opponent_curriculum.py"),
    ("test_oracle_bundle.py", "tests/test_oracle_bundle.py"),
    ("test_position_store.py", "tests/test_position_store.py"),
    ("test_position_store_migration.py", "tests/test_position_store_migration.py"),
    ("test_reduction_counterfactuals.py", "tests/test_reduction_counterfactuals.py"),
    ("test_run_nnue_cycle.py", "tests/test_run_nnue_cycle.py"),
    ("test_search_importance.py", "tests/test_search_importance.py"),
    ("test_teacher_dataset.py", "tests/test_teacher_dataset.py"),
    # production package
    ("halfpw.py", "titanium_training/models/halfpw.py"),
    ("field_planes.py", "titanium_training/models/field_planes.py"),
    ("train.py", "titanium_training/training/trainer.py"),
    ("nnue_guards.py", "titanium_training/training/guards.py"),
    ("nnue_learning_metrics.py", "titanium_training/training/learning_metrics.py"),
    ("plateau_probe.py", "titanium_training/training/plateau_probe.py"),
    ("nnue_cli.py", "titanium_training/cli.py"),
    # store
    ("position_store_config.py", "titanium_training/store/config.py"),
    ("position_store_lib.py", "titanium_training/store/lib.py"),
    ("position_store_state.py", "titanium_training/store/state.py"),
    ("position_store_guards.py", "titanium_training/store/guards.py"),
    ("position_store_compact.py", "titanium_training/store/compact.py"),
    ("position_store_friend.py", "titanium_training/store/friend.py"),
    ("position_store_teacher.py", "titanium_training/store/teacher.py"),
    ("position_store_migration.py", "titanium_training/store/migration.py"),
    ("position_store_split.py", "titanium_training/store/split.py"),
    ("position_store.py", "titanium_training/store/cli.py"),
    ("move_codec.py", "titanium_training/store/move_codec.py"),
    # tools
    ("datagen.py", "tools/datagen/datagen.py"),
    ("manifest.py", "tools/maintenance/manifest.py"),
    ("housekeeping.py", "tools/maintenance/housekeeping.py"),
    ("regression_triage.py", "tools/maintenance/regression_triage.py"),
    ("verify_db_games.py", "tools/maintenance/verify_db_games.py"),
    ("parse_flamegraph.py", "tools/analysis/parse_flamegraph.py"),
    ("visualize_fields.py", "tools/analysis/visualize_fields.py"),
    ("run_benchmarks.py", "tools/operations/run_benchmarks.py"),
    ("run_infinite_benchmark.py", "tools/operations/run_infinite_benchmark.py"),
    ("run_nnue_cycle.py", "tools/operations/run_nnue_cycle.py"),
    ("run_swiss_overnight.py", "tools/operations/run_swiss_overnight.py"),
    ("supervise.py", "tools/operations/supervise.py"),
    ("swiss_tournament.py", "tools/operations/swiss_tournament.py"),
    ("coordinator.py", "tools/operations/coordinator.py"),
    ("pool_labels.py", "tools/operations/pool_labels.py"),
    ("pool_preflight.py", "tools/operations/pool_preflight.py"),
    ("opponent_curriculum.py", "tools/operations/opponent_curriculum.py"),
    ("ingest_self_match_game.py", "tools/datagen/ingest_self_match_game.py"),
    ("import_clipboard_game.py", "tools/datagen/import_clipboard_game.py"),
    ("bisect_engine_step.py", "tools/engine_parity/bisect_engine_step.py"),
    # experiments
    ("train_lmr_head_v3.py", "experiments/lmr/train_lmr_head_v3.py"),
    ("train_reduction_sidecar.py", "experiments/lmr/train_reduction_sidecar.py"),
    ("train_reduction_sidecar_v2.py", "experiments/lmr/train_reduction_sidecar_v2.py"),
    ("train_search_importance.py", "experiments/lmr/train_search_importance.py"),
    ("collect_reduction_counterfactuals.py", "experiments/lmr/collect_reduction_counterfactuals.py"),
    ("collect_reduction_counterfactuals_v3.py", "experiments/lmr/collect_reduction_counterfactuals_v3.py"),
    ("collect_search_importance.py", "experiments/lmr/collect_search_importance.py"),
    ("run_search_pressure_experiment.py", "experiments/lmr/run_search_pressure_experiment.py"),
    ("reduction_counterfactual_schema.py", "experiments/lmr/reduction_counterfactual_schema.py"),
    ("compare_halfpw.py", "experiments/evaluation/compare_halfpw.py"),
    ("compare_pressure_sources.py", "experiments/evaluation/compare_pressure_sources.py"),
    ("color_rotation.py", "experiments/features/color_rotation.py"),
    ("extend_field_planes.py", "experiments/features/extend_field_planes.py"),
    ("extend_weights.py", "experiments/features/extend_weights.py"),
    ("eyeball_inputs.py", "experiments/features/eyeball_inputs.py"),
    ("freeze_baseline_weights.py", "experiments/features/freeze_baseline_weights.py"),
    ("probe_legal_wall_signal.py", "experiments/features/probe_legal_wall_signal.py"),
    ("inspect_ka_arch.py", "experiments/features/inspect_ka_arch.py"),
    # scripts
    ("profile_titanium.ps1", "tools/scripts/profile_titanium.ps1"),
    ("pool_watch.ps1", "tools/scripts/pool_watch.ps1"),
    ("pool_watch.cmd", "tools/scripts/pool_watch.cmd"),
    ("overnight_watch.ps1", "tools/scripts/overnight_watch.ps1"),
    ("run_bisect_and_overnight.ps1", "tools/scripts/run_bisect_and_overnight.ps1"),
    ("run_bisect_continue.ps1", "tools/scripts/run_bisect_continue.ps1"),
    ("run_supervised_session.ps1", "tools/scripts/run_supervised_session.ps1"),
    ("run_supervised_session.cmd", "tools/scripts/run_supervised_session.cmd"),
    ("post_completion_teacher_import.ps1", "tools/scripts/post_completion_teacher_import.ps1"),
    ("stop_training.cmd", "tools/scripts/stop_training.cmd"),
    ("watch_progress.ps1", "tools/scripts/watch_progress.ps1"),
    ("watch_progress.cmd", "tools/scripts/watch_progress.cmd"),
    ("import_clipboard_game.cmd", "tools/scripts/import_clipboard_game.cmd"),
    ("run_ka_teacher_label.cmd", "tools/scripts/run_ka_teacher_label.cmd"),
    ("run_partial_golden_match.cmd", "tools/scripts/run_partial_golden_match.cmd"),
    # docs to docs/decisions or delete
    ("PHASE3_LMRH_RUNBOOK.md", "experiments/lmr/RUNBOOK.md"),
]

DELETES = [
    "_breakdown.py",
    "_jsonl_miss_audit.py",
    "_progress.py",
    "_stats_tmp.py",
    "_v8_recovery.py",
    "_v9_finalize.py",
    "_test_manifest_direct.py",
    "migrate_sparse_routes.py",
    "ka_api_teacher.py",
    "ka_teacher_worker.py",
    "ARCHITECTURE_HANDOFF.md",
    "AUDIT_REPORT.md",
    "CANONICAL_DATASTORE.md",
    "POSITION_STORE_RUNBOOK.md",
    "REGRESSION_BISECT.md",
    "SEARCH_PRESSURE_REPORT.md",
    "REDUCTION_SIDECAR_REPORT.md",
    "WEAK_AI_TASKS.md",
]

IMPORT_REPLACEMENTS = [
    (r"\bfrom position_store_config import\b", "from titanium_training.store.config import"),
    (r"\bimport position_store_config\b", "import titanium_training.store.config as position_store_config"),
    (r"\bfrom position_store_lib import\b", "from titanium_training.store.lib import"),
    (r"\bimport position_store_lib\b", "import titanium_training.store.lib as position_store_lib"),
    (r"\bfrom position_store_guards import\b", "from titanium_training.store.guards import"),
    (r"\bfrom position_store_compact import\b", "from titanium_training.store.compact import"),
    (r"\bfrom position_store_friend import\b", "from titanium_training.store.friend import"),
    (r"\bfrom position_store_teacher import\b", "from titanium_training.store.teacher import"),
    (r"\bfrom position_store_migration import\b", "from titanium_training.store.migration import"),
    (r"\bfrom position_store_split import\b", "from titanium_training.store.split import"),
    (r"\bfrom position_store_state import\b", "from titanium_training.store.state import"),
    (r"\bfrom move_codec import\b", "from titanium_training.store.move_codec import"),
    (r"\bfrom field_planes import\b", "from titanium_training.models.field_planes import"),
    (r"\bfrom halfpw import\b", "from titanium_training.models.halfpw import"),
    (r"\bfrom train import\b", "from titanium_training.training.trainer import"),
    (r"\bimport train\b", "import titanium_training.training.trainer as train"),
    (r"\bfrom nnue_guards import\b", "from titanium_training.training.guards import"),
    (r"\bfrom datagen import\b", "from tools.datagen.datagen import"),
    (r"\bfrom engine_identity import\b", "from titanium_training.validation.engine_identity import"),
    (r"\bfrom manifest import\b", "from tools.maintenance.manifest import"),
    (r"\bfrom plateau_probe import\b", "from titanium_training.training.plateau_probe import"),
    (r"\bfrom nnue_learning_metrics import\b", "from titanium_training.training.learning_metrics import"),
    (r"\bfrom housekeeping import\b", "from tools.maintenance.housekeeping import"),
    (r"\bfrom pool_preflight import\b", "from tools.operations.pool_preflight import"),
    (r"\bfrom pool_labels import\b", "from tools.operations.pool_labels import"),
    (r"\bfrom opponent_curriculum import\b", "from tools.operations.opponent_curriculum import"),
    (r"\bfrom swiss_tournament import\b", "from tools.operations.swiss_tournament import"),
    (r"python training/position_store\.py", "python -m titanium_training.store.cli"),
    (r"python training/train\.py", "python training/nnue_cli.py train"),
    (r"python training/supervise\.py", "python training/tools/operations/supervise.py"),
    (r"python training/validate_train_ready\.py", "python training/nnue_cli.py preflight"),
    (r"training/position_store\.py", "training/titanium_training/store/cli.py"),
    (r"training/train\.py", "training/titanium_training/training/trainer.py"),
    (r"training/supervise\.py", "training/tools/operations/supervise.py"),
    (r"training/validate_train_ready\.py", "python training/nnue_cli.py preflight"),
    (r"training/parity_check\.py", "training/titanium_training/validation/parity_check.py"),
    (r"training/engine_identity\.py", "training/titanium_training/validation/engine_identity.py"),
    (r"training/nnue_cli\.py", "training/nnue_cli.py"),
    (r"training/value_nnue_smoke\.py", "training/titanium_training/validation/smoke.py"),
    (r"training/tests/test_", "training/tests/test_"),
]

SKIP_DIRS = {".pytest-temp", "__pycache__", ".pytest_cache", "data", "runs", "checkpoints", "checkpoints_smoke"}


def run_git_mv(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        return
    if dst.exists():
        return
    subprocess.run(["git", "mv", str(src), str(dst)], cwd=str(ROOT), check=True)


def fix_store_config(text: str) -> str:
    text = text.replace(
        "ROOT = Path(__file__).resolve().parent.parent",
        "ROOT = Path(__file__).resolve().parents[3]",
    )
    text = text.replace("training/position_store", "training/titanium_training/store")
    text = text.replace("training/position_store_friend", "training/titanium_training/store/friend")
    text = text.replace("training/tests/test_position_store", "training/tests/test_position_store")
    text = text.replace("training/datagen", "training/tools/datagen")
    text = text.replace("training/coordinator", "training/tools/operations/coordinator")
    text = text.replace("training/ingest_self_match", "training/tools/datagen/ingest_self_match")
    text = text.replace("training/import_clipboard", "training/tools/datagen/import_clipboard")
    text = text.replace("training/verify_db_games", "training/tools/maintenance/verify_db_games")
    text = text.replace("training/watch_progress", "training/tools/scripts/watch_progress")
    text = text.replace("training/collect_search_importance", "training/experiments/lmr/collect_search_importance")
    text = text.replace("training/collect_reduction", "training/experiments/lmr/collect_reduction")
    text = text.replace("training/run_search_pressure_experiment", "training/experiments/lmr/run_search_pressure_experiment")
    text = text.replace("training/train_search_importance", "training/experiments/lmr/train_search_importance")
    text = text.replace("training/ka_api_teacher", "training/tools/maintenance/ka_api_teacher")
    text = text.replace("training/AUDIT_REPORT", "docs/decisions")
    return text


def rewrite_file(path: Path) -> None:
    if path.suffix not in {".py", ".md", ".ps1", ".cmd", ".sh", ".yaml", ".yml", ".ini", ".json"}:
        return
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return
    orig = text
    if path.name == "config.py" and "titanium_training" in str(path):
        text = fix_store_config(text)
    if path.name == "trainer.py":
        text = text.replace(
            "ROOT = Path(__file__).resolve().parent.parent",
            "ROOT = Path(__file__).resolve().parents[3]",
        )
        text = text.replace(
            'Path(__file__).resolve().parent.parent / "engine"',
            "ROOT / \"engine\"",
        )
    for pattern, repl in IMPORT_REPLACEMENTS:
        text = re.sub(pattern, repl, text)
    # Remove sys.path hacks pointing at training root where possible
    text = re.sub(
        r"sys\.path\.insert\(0, str\(ROOT / \"training\"\)\)\n",
        "",
        text,
    )
    text = re.sub(
        r"sys\.path\.insert\(0, str\(TRAINING\)\)\n",
        "",
        text,
    )
    if text != orig:
        path.write_text(text, encoding="utf-8")


def main() -> int:
    rewrite_only = "--rewrite-only" in __import__("sys").argv
    if not rewrite_only:
        for rel in DELETES:
            p = TR / rel
            if p.exists():
                p.unlink()
        for src, dst in MOVES:
            run_git_mv(TR / src, TR / dst)
    # package inits
    for pkg in [
        "titanium_training/models",
        "titanium_training/training",
        "titanium_training/validation",
        "titanium_training/store",
        "tools/datagen",
        "tools/maintenance",
        "tools/operations",
        "tools/engine_parity",
        "tools/analysis",
        "tools/scripts",
        "experiments/lmr",
        "experiments/features",
        "experiments/evaluation",
        "tests",
    ]:
        init = TR / pkg / "__init__.py"
        if not init.exists():
            init.write_text('"""Package."""\n', encoding="utf-8")
    for path in TR.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.is_file():
            rewrite_file(path)
    # docs + scripts outside training
    for path in list((ROOT / "docs").rglob("*")) + list((ROOT / "scripts").rglob("*")) + [ROOT / "README.md"]:
        if path.is_file():
            rewrite_file(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
