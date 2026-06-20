# Training

## Canonical CLI

All operator actions go through **`training/nnue_cli.py`** (shell scripts under `scripts/oracle/` are thin wrappers).

```powershell
python training/nnue_cli.py doctor
python training/nnue_cli.py verify-dataset
python training/nnue_cli.py preflight
python training/nnue_cli.py smoke --config training/configs/smoke.yaml
python training/nnue_cli.py train --config training/configs/value_nnue_oracle.yaml
python training/nnue_cli.py resume --checkpoint training/runs/<run_id>/checkpoints/best.pt
python training/nnue_cli.py export --checkpoint training/runs/<run_id>/checkpoints/best.pt
```

## Configurations

| Config | Use |
| ------ | --- |
| `training/configs/smoke.yaml` | Bounded smoke (minutes) |
| `training/configs/value_nnue_oracle.yaml` | Oracle production defaults |
| `training/configs/value_nnue_local.yaml` | Local WDL fine-tune |

## Smoke test

Smoke verifies:

1. Active teacher dataset manifest + artifact sample + loader IO
2. Micro-train on a **tiny game subset** (WDL path)
3. Checkpoint write, resume, engine-format export

It does **not** start a multi-hour campaign. Output: `training/runs/smoke_<timestamp>/`.

```powershell
python training/nnue_cli.py smoke
# or
bash scripts/oracle/smoke_train.sh
```

## Production runs

Each real run should use a dedicated directory under the runs folder (gitignored at repository root).

```text
training/runs/<run_id>/
  resolved_config.json
  checkpoints/
  net_weights_export.bin
```

## Value NNUE vs LMR

| Track | Status | Entry |
| ----- | ------ | ----- |
| Value NNUE | **Current** — Oracle first run | `nnue_cli.py train` + teacher dataset verification |
| LMR / reduction head | **Experimental** | training/experiments/lmr/ — not production |

Do not mix LMR commands into the Oracle first-run guide.

## Dependencies

```powershell
pip install -r training/requirements.txt
pip install -r training/requirements-teacher-dataset.txt
```

Supported Python: **3.11–3.12**. CPU-only is expected on Oracle ARM; CUDA optional locally.

## Legacy pipeline docs

Overnight pool, parity, guards: [training/README.md](../training/README.md) and [OPERATIONS.md](OPERATIONS.md).
