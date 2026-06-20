# Dataset

## Active teacher dataset (promoted v10)

| Property | Value |
| -------- | ----- |
| Path | `training/data/teacher_dataset/` |
| Manifest SHA256 | `31a422f25a8c701ebfa72410f59fab9dff52c2717e30985a3f8e6929be007d02` |
| Positions | 1,405,888 |
| Labels | 2,281,163 |
| Observations | 1,454,824 |
| Unique policies | 1,927,597 |
| Labels with policy | 2,275,885 |
| Labels without policy | 5,278 |

**Rules**

- Do not modify artifacts, rewrite the manifest, rename the tree, or re-run promotion during normal operations.
- Oracle packaging must preserve the repository-relative path `training/data/teacher_dataset/`.
- Future candidates should use manifest paths relative to the manifest file to avoid promotion-time rewrites.

## Rollback (local only)

`training/data/teacher_dataset_rollback_20260620T111616Z/` — keep until the first real Oracle checkpoint validates. Excluded from Oracle upload bundles.

## Provenance and audit

| File | Role |
| ---- | ---- |
| `training/teacher_dataset/candidate_provenance/teacher_dataset_v10.json` | Candidate provenance |
| `training/teacher_dataset/candidate_provenance/teacher_dataset_v10_promotion_receipt.json` | Promotion receipt (SHA256 `97e69900…`) |

Audit timestamp `20260620T101843Z`; final evidence envelope SHA256 `7b2d9297…`.

## Game store (WDL micro-train path)

| Store | Path |
| ----- | ---- |
| Game store | `training/data/canonical/game_store.db` |
| Teacher SQLite (reference) | `training/data/canonical/position_teacher_store.db` |

Normal `train.py` reads the **game store** today. Teacher Parquet is verified in smoke and prepared for value-target training; see [TRAINING.md](TRAINING.md).

## Verification commands

```powershell
python training/nnue_cli.py verify-dataset
python scripts/maintenance/repository_doctor.py
```

Deeper audits live in `training/teacher_dataset/` (gate audits, artifact verification, loader smoke).

## Legacy reference

Position-store administration: `python -m titanium_training.store.cli --help` (from `training/` with `PYTHONPATH=.`).
