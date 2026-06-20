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
| `training/teacher_dataset/` | Dataset package (build, audit, promote) | source | yes |
| `training/data/teacher_dataset/` | **Active promoted v10 dataset** | local data | optional (`--include-active-dataset`) |
| `training/data/teacher_dataset_rollback_*` | Pre-promotion rollback | local | **excluded** |
| `training/data/canonical/` | Game + teacher SQLite stores | local | no (Oracle uses Parquet dataset) |
| `training/configs/` | Smoke and value-NNUE YAML | source | yes |
| `training/runs/<run_id>/` | Self-contained run directories | gitignored | no |
| `training/checkpoints/` | Legacy checkpoint dir | gitignored | no |
| `training/nnue_cli.py` | **Canonical training CLI** | source | yes |
| `training/value_nnue_smoke.py` | Bounded end-to-end smoke | source | yes |
| `training/train.py` | HalfPW WDL fine-tune from game store | source | yes |
| `training/supervise.py` | Overnight pool supervisor | source | yes |
| `training/position_store.py` | Position graph / teacher import CLI | source | yes |

## Scripts

| Path | Entrypoint |
| ---- | ---------- |
| `scripts/oracle/build_upload_bundle.py` | Build Oracle upload package |
| `scripts/oracle/verify_upload_bundle.py` | Verify package before/after transfer |
| `scripts/oracle/bootstrap.sh` | Oracle machine bootstrap |
| `scripts/oracle/doctor.sh` | Machine + repo doctor |
| `scripts/oracle/smoke_train.sh` | Training smoke |
| `scripts/maintenance/repository_doctor.py` | Local repository health |

## Authoritative vs legacy docs

Use **`docs/`** for current procedures. Files under `training/*.md` that duplicate `docs/` are being consolidated; prefer links from `docs/README.md`.
