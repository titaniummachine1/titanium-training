# Troubleshooting

## Repository doctor fails on dataset

- Confirm `training/data/teacher_dataset/manifest.json` exists locally (gitignored but required on disk).
- Expected manifest SHA256: `31a422f25a8c701ebfa72410f59fab9dff52c2717e30985a3f8e6929be007d02`
- Run `python training/nnue_cli.py verify-dataset` for details.

## Oracle bundle verification fails

- Re-build with the same flags (`--code-only` vs `--include-active-dataset`).
- Do not edit files inside the bundle after `transfer-manifest.json` is written.
- Check for accidental `.partial` directories under `training/data/`.

## Smoke fails at train phase

- Requires `engine/target/release/titanium` built with native CPU flags.
- Requires `training/data/canonical/game_store.db` with at least one game.
- Run `python training/nnue_cli.py preflight`.

## Training blocked by guards

- Read message from `training/data/nnue_train.log`.
- Artifact cap: prune old checkpoints under `training/checkpoints/`.
- Engine stamp mismatch: rebuild engine and refresh stamp via preflight.

## Submodule / coordinator warning

`fatal: no submodule mapping found in .gitmodules for path 'coordinator'` — coordinator checkout is optional; does not block Oracle value training.

## Rollback recovery

If promotion must be reversed manually, the rollback tree is:

```text
training/data/teacher_dataset_rollback_20260620T111616Z/
```

Do not delete until Oracle validates a real checkpoint. Restoration is a manual operator procedure — not automated in this pass.
