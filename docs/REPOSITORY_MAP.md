# Repository map

Quick orientation for the Titanium workspace (training root + embedded repos).

## Top level

| Path | Purpose | Tracked | Safe to edit? | Oracle bundle |
| ---- | ------- | ------- | ------------- | ------------- |
| `README.md` | Entry point | yes | yes | yes |
| `docs/` | Canonical documentation | yes | yes | yes |
| `scripts/` | Oracle + maintenance CLIs | yes | yes | yes |
| `training/` | NNUE training, datasets, tests | yes | mostly | yes |
| `engine/` | Rust search engine (submodule) | submodule | **no** (frozen) | no (binary built on target) |
| `tools/` | Rust position-store importer | yes | when needed | yes |
| `site/` | Website repo | submodule | separate project | no |
| `coordinator/` | Cloudflare worker | submodule | separate project | no |
| `test-client/` | Distributed match worker | submodule | separate project | no |
| `KaAiData/` | Local Ka weights / corpus | ignored | local only | no |
| `dist/` | Oracle upload staging | ignored | generated | n/a |

## Training tree

| Path | Purpose | Generated? | Oracle bundle |
| ---- | ------- | ---------- | ------------- |
| `training/nnue_cli.py` | Operator CLI wrapper | source | yes |
| `training/titanium_training/` | Production package (trainer, store, validation) | source | yes |
| `training/tools/` | Pool, datagen, parity, maintenance | source | partial |
| `training/experiments/` | LMR/feature research | source | **no** |
| `training/tests/` | Pytest suite | source | smoke-related |
| `training/teacher_dataset/` | Dataset build/audit code | source | yes |
| `training/data/teacher_dataset/` | **Active promoted v10 dataset** | local data | optional |
| `training/data/teacher_dataset_rollback_*` | Pre-promotion rollback | local | **excluded** |
| `training/configs/` | Smoke and value-NNUE YAML | source | yes |
| `training/runs/` | Run directories | gitignored | no |

Reorganization details: [maintenance/training_root_cleanup.md](maintenance/training_root_cleanup.md)

## Scripts

| Path | Entrypoint |
| ---- | ---------- |
| `scripts/oracle/build_upload_bundle.py` | Build Oracle upload package |
| `scripts/oracle/verify_upload_bundle.py` | Verify package before/after transfer |
| `scripts/oracle/bootstrap.sh` | Oracle machine bootstrap |
| `scripts/oracle/doctor.sh` | Machine + repo doctor |
| `scripts/oracle/smoke_train.sh` | Training smoke |
| `scripts/maintenance/repository_doctor.py` | Local repository health |

## Authoritative docs

Use **`docs/`** only. Legacy `training/*.md` runbooks were removed or relocated during the training root cleanup.
