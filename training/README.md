# Training

Production value-NNUE code lives in **`titanium_training/`**. Operator entrypoint:

```powershell
python training/nnue_cli.py doctor
python training/nnue_cli.py verify-dataset
python training/nnue_cli.py smoke
python training/nnue_cli.py train --config training/configs/value_nnue_local.yaml
```

## Layout

| Path | Purpose |
| ---- | ------- |
| `titanium_training/` | Production package (trainer, models, validation, store) |
| `tools/` | Supported operator utilities (pool, datagen, parity, maintenance) |
| `experiments/` | Non-production LMR/feature research (excluded from Oracle bundle) |
| `tests/` | Pytest suite |
| `teacher_dataset/` | Immutable dataset tooling (not active data) |
| `zero_teacher/` | External MCTS label mining |
| `configs/` | Smoke and value-NNUE YAML |
| `data/` | Runtime data (gitignored) |
| `runs/` | Training run output (gitignored) |

Canonical documentation: [docs/README.md](../docs/README.md)

Run tests from this directory:

```powershell
cd training
pytest -q
```
