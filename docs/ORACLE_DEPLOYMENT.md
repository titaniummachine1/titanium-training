# Oracle deployment

End-to-end guide for packaging, transfer, bootstrap, and first value-NNUE smoke on Oracle (Linux ARM).

## 1. Build upload bundle (local Windows/Linux)

**Code only** (transfer dataset separately):

```powershell
python scripts/oracle/build_upload_bundle.py --output dist/oracle_upload --code-only
```

**Include active teacher dataset** (~13 GB+):

```powershell
python scripts/oracle/build_upload_bundle.py --output dist/oracle_upload --include-active-dataset
```

Optional archive:

```powershell
python scripts/oracle/build_upload_bundle.py --output dist/oracle_upload --include-active-dataset --archive
```

The builder:

- Verifies active manifest hash and counts (when dataset included)
- Verifies provenance / promotion receipt
- Preserves path `training/data/teacher_dataset/`
- Writes `transfer-manifest.json` + `README_FIRST.md`
- Excludes rollback trees, `.git`, caches, checkpoints, `.partial` trees

## 2. Verify before transfer

```powershell
python scripts/oracle/verify_upload_bundle.py dist/oracle_upload
```

Fails closed on hash mismatch, missing required files, or forbidden paths.

## 3. Transfer

Copy `dist/oracle_upload/` (or the `.tar.gz`) to Oracle. Extract preserving directory layout.

## 4. Bootstrap on Oracle

```bash
bash scripts/oracle/bootstrap.sh
bash scripts/oracle/doctor.sh
```

Bootstrap creates `.venv`, installs pinned requirements, checks disk/Python, and confirms dataset visibility.

Copy `.env.example` → `.env` only for non-secret path overrides.

## 5. Smoke test

```bash
bash scripts/oracle/smoke_train.sh
```

Equivalent:

```bash
python training/nnue_cli.py smoke --config training/configs/smoke.yaml
```

## 6. Start value training (after smoke passes)

```bash
bash scripts/oracle/start_value_training.sh training/configs/value_nnue_oracle.yaml
```

Resume:

```bash
bash scripts/oracle/resume_value_training.sh training/runs/value_oracle/checkpoints/best.pt
```

Collect artifacts:

```bash
bash scripts/oracle/collect_results.sh training/runs/value_oracle dist/oracle_results
```

## Bundle contents summary

| Category | Included (code bundle) | Included (with `--include-active-dataset`) |
| -------- | ---------------------- | -------------------------------------------- |
| `docs/`, `scripts/`, `training/` source | yes | yes |
| `tools/position_store_importer/` | yes | yes |
| `training/data/teacher_dataset/` | no | **yes** |
| Rollback dataset | no | no |
| `engine/` source | no | no |
| Local checkpoints / logs | no | no |

## Do not

- Modify files under `training/data/teacher_dataset/` after packaging
- Delete local rollback until Oracle checkpoint validates
- Start LMR production training before value NNUE is frozen (see [ROADMAP.md](ROADMAP.md))
