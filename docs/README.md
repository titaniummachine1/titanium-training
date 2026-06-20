# Titanium documentation

Canonical guides for operators and developers. Start at the [repository root README](../README.md), then use this index.

| Document | Purpose |
| -------- | ------- |
| [REPOSITORY_MAP.md](REPOSITORY_MAP.md) | Folder tree, what is safe to edit, Oracle bundle inclusion |
| [ARCHITECTURE.md](ARCHITECTURE.md) | NN + search design philosophy |
| [DATASET.md](DATASET.md) | Active teacher dataset, game store, immutability rules |
| [TRAINING.md](TRAINING.md) | Value-NNUE training commands and run layout |
| [ORACLE_DEPLOYMENT.md](ORACLE_DEPLOYMENT.md) | Upload bundle, bootstrap, smoke, first production run |
| [ENGINE_INTEGRATION.md](ENGINE_INTEGRATION.md) | Engine submodule, native builds, parity |
| [OPERATIONS.md](OPERATIONS.md) | Overnight pool, supervision, checkpoints |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Common failures and recovery |
| [ROADMAP.md](ROADMAP.md) | Value NNUE → LMR sequence (LMR not production-ready) |
| [TECHNICAL_DEBT.md](TECHNICAL_DEBT.md) | Known residual debt after bookkeeping pass |
| [maintenance/training_root_cleanup.md](maintenance/training_root_cleanup.md) | Training root reorganization report |

**Do not touch casually**

- `engine/` submodule (frozen during dataset promotion work)
- `training/data/teacher_dataset/` (promoted v10 — immutable)
- `training/data/teacher_dataset_rollback_*` (local rollback until Oracle validates)
- `training/teacher_dataset/candidate_provenance/` (audit receipts)
