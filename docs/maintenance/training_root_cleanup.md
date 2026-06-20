# Training root cleanup (2026-06-20)

Second bookkeeping pass: reorganize loose `training/` scripts into `titanium_training/`, `tools/`, `experiments/`, and `tests/`.

## Root before

~90 loose `.py`, `.md`, `.ps1`, and `.cmd` files at `training/` root (see git history at `backup/pre-training-root-cleanup-*`).

## Root after

```text
training/
├── README.md
├── nnue_cli.py
├── pytest.ini
├── requirements.txt
├── requirements-teacher-dataset.txt
├── configs/
├── titanium_training/
├── tools/
├── experiments/
├── tests/
├── teacher_dataset/
├── zero_teacher/
├── data/          (gitignored runtime)
└── runs/          (gitignored)
```

## Actions summary

| Old path | Classification | Action | New path | Reason |
| -------- | -------------- | ------ | -------- | ------ |
| `train.py` | production | MOVE | `titanium_training/training/trainer.py` | Canonical trainer |
| `halfpw.py`, `field_planes.py` | production | MOVE | `titanium_training/models/` | Model code |
| `nnue_guards.py`, `plateau_probe.py`, `nnue_learning_metrics.py` | production | MOVE | `titanium_training/training/` | Training guards/metrics |
| `position_store*.py`, `move_codec.py` | production | MOVE | `titanium_training/store/` | Dataset store |
| `validate_train_ready.py`, `parity_check.py`, `engine_identity.py`, `value_nnue_smoke.py` | production | MOVE | `titanium_training/validation/` | Preflight/smoke |
| `nnue_cli.py` | production | MOVE | `titanium_training/cli.py` + root wrapper | Canonical CLI |
| `datagen.py`, `supervise.py`, `run_*`, `manifest.py`, … | operational | MOVE | `tools/` | Supported operator tools |
| `train_lmr_head_v3.py`, `collect_reduction_*`, … | experimental | MOVE | `experiments/` | Not Oracle production |
| `test_*.py`, `conftest.py` | tests | MOVE | `tests/` | Pytest layout |
| `_breakdown.py`, `_progress.py`, … | migration/temp | DELETE | — | v10 complete; git history |
| `migrate_sparse_routes.py`, `ka_*teacher*.py` | migration/dead | DELETE | — | Superseded/disabled |
| `ARCHITECTURE_HANDOFF.md`, `AUDIT_REPORT.md`, … | stale docs | DELETE | — | Merged into `docs/` |
| `*.ps1`, `*.cmd` at root | operational | MOVE | `tools/scripts/` | Launcher scripts |

Full machine move list: `scripts/maintenance/relayout_training.py` (`MOVES` table).

## Canonical commands

```powershell
python training/nnue_cli.py doctor
python training/nnue_cli.py verify-dataset
python training/nnue_cli.py smoke
python training/nnue_cli.py train --config training/configs/value_nnue_oracle.yaml
cd training && pytest -q
```

## Oracle bundle

Includes `titanium_training/`, `tools/` (operational subset), `nnue_cli.py`, configs, docs. Excludes `experiments/`, `runs/`, rollback dataset.

## Pytest temp and bundle invariants

After the post-relayout validation pass, these rules are enforced:

- Tests may remove only directories they created (marker subdirs under `.pytest-temp/`, not the session basetemp root).
- No test may delete pytest's session basetemp (`training/.pytest-temp/`).
- Bundle builder excludes `.pytest-temp/`, `dist/`, bundle output directory, and other forbidden prefixes (`scripts/lib/bundle_lib.py`).
- Bundle regression tests must not recursively package their own output (`test_bundle_excludes_output_directory_recursion`).
- Cleanup helpers may remove stale temp trees only when no pytest session is active.

Regression: test_bundle_excludes_pytest_temp_artifacts in training/tests/test_oracle_bundle.py.

Root cause of Run 2 errors: an Oracle bundle regression test deleted the entire session-wide pytest base-temp directory, invalidating all later `tmp_path` fixtures.
