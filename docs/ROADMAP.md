# Roadmap

## Sequence (mandatory order)

1. **Train and validate value NNUE** on the promoted teacher dataset (Oracle first run).
2. **Freeze** the selected value network and search configuration.
3. **Generate LMR supervision** using the frozen engine + value net environment.
4. **Conservative LMR head**: predict **+1 ply reduction** first.
5. **Expand** to reduction-magnitude prediction only after holdout validation.

## Current status

| Item | Status |
| ---- | ------ |
| Teacher dataset v10 | Promoted — `training/data/teacher_dataset/` |
| Value NNUE Oracle packaging | Ready — see [ORACLE_DEPLOYMENT.md](ORACLE_DEPLOYMENT.md) |
| Teacher-value featurization in `train.py` | **Not wired** — smoke uses game-store WDL micro-train + dataset verification |
| LMR head (`train_lmr_head_v3.py`, sidecars) | **Experimental** — see `training/PHASE3_LMRH_RUNBOOK.md` |
| LMR production config | **Not created** intentionally |

## Where LMR work belongs later

- Code: `training/train_lmr_head_v3.py`, `training/collect_reduction_counterfactuals*.py`
- Docs: `training/PHASE3_LMRH_RUNBOOK.md` (experimental runbook only)
- Config: add an experimental LMR YAML under training/configs/ when the pipeline is validated — not before

Do not present LMR as ready for Oracle production until value NNUE is frozen and validated.
