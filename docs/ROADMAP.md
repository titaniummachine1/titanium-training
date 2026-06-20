# Roadmap

## Sequence (mandatory order)

1. **Train and validate value NNUE** on the promoted teacher dataset (Oracle first run).
2. **Freeze** the selected value network and search configuration.
3. **Generate LMR supervision** using the frozen engine + value net environment.
4. **Conservative LMR head**: predict **+1 ply reduction** first.
5. **Expand** to reduction-magnitude prediction only after holdout validation.

## Current status

| Item                                   | Status                                                                                        |
| -------------------------------------- | --------------------------------------------------------------------------------------------- |
| Teacher dataset v10                    | Promoted — `training/data/teacher_dataset/`                                                   |
| Value NNUE Oracle packaging            | Ready — see [ORACLE_DEPLOYMENT.md](ORACLE_DEPLOYMENT.md)                                      |
| Teacher-value featurization in trainer | **Ready (packed-state)** — `eval-packed-batch` + `smoke-teacher`; preflight FULL-CORPUS READY |
| LMR head (experiments/lmr/)            | **Experimental** — see experiments/lmr/RUNBOOK.md                                             |
| LMR production config                  | **Not created** intentionally                                                                 |

## Where LMR work belongs later

- Code: training/experiments/lmr/
- Docs: training/experiments/lmr/RUNBOOK.md (experimental runbook only)
- Config: add an experimental LMR YAML under training/configs/ when the pipeline is validated — not before

Do not present LMR as ready for Oracle production until value NNUE is frozen and validated.
