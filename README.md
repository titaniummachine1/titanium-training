# Titanium Quoridor — training workspace

Titanium is a Quoridor engine + **HalfPW value-NNUE** training pipeline. This repository root tracks the **training backup** (`quoridor-training` on GitHub) with embedded `engine/`, `site/`, and worker repos.

## What is where?

| Need | Location |
| ---- | -------- |
| Engine (Rust) | `engine/` — [engine integration](docs/ENGINE_INTEGRATION.md) |
| Training code | `training/` — [training guide](docs/TRAINING.md) |
| Active teacher dataset | `training/data/teacher_dataset/` — [dataset spec](docs/DATASET.md) |
| Documentation index | [docs/README.md](docs/README.md) |
| Folder map | [docs/REPOSITORY_MAP.md](docs/REPOSITORY_MAP.md) |

## Quick verification

```powershell
python scripts/maintenance/repository_doctor.py
python training/nnue_cli.py verify-dataset
```

## Oracle upload (one command)

```powershell
# Code only
python scripts/oracle/build_upload_bundle.py --output dist/oracle_upload --code-only

# Include promoted teacher dataset
python scripts/oracle/build_upload_bundle.py --output dist/oracle_upload --include-active-dataset

python scripts/oracle/verify_upload_bundle.py dist/oracle_upload
```

Full procedure: [docs/ORACLE_DEPLOYMENT.md](docs/ORACLE_DEPLOYMENT.md)

## Training smoke (one command)

```powershell
python training/nnue_cli.py smoke --config training/configs/smoke.yaml
```

## Do not touch casually

- **`engine/`** submodule — local unpushed commits; do not reset or stage pointer changes during dataset work
- **`training/data/teacher_dataset/`** — promoted audited v10 (manifest `31a422f25…`)
- **`training/data/teacher_dataset_rollback_*`** — local rollback until Oracle validates
- **`training/teacher_dataset/candidate_provenance/`** — audit receipts

## Embedded repos

| Folder | Role |
| ------ | ---- |
| `engine/` | Rust engine (UCI, WASM, ACE v13/v15) |
| `site/` | Playable UI |
| `test-client/` | Distributed match worker |

Legacy multi-repo push helpers: `setup_repos.ps1`, `push_training.ps1`.

## Native engine build

```powershell
cd engine
$env:RUSTFLAGS = "-C target-cpu=native"
cargo build --release -p titanium
```

See `.cursor/rules/titanium-native-build.mdc`.
