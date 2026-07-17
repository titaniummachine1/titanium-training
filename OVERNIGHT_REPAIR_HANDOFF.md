# Overnight Repair Handoff

**Do not change label resolution, source priorities, confidence formulas, frequency weighting, phase quotas, acceptance thresholds, or model architecture overnight.**

## Current Intended Setup

| Item                       | Value                                                                   |
| -------------------------- | ----------------------------------------------------------------------- |
| Parent weights             | epoch 37 h96, hash prefix `beb3a746`                                    |
| `STREAM_REPAIR_MODE`       | `1`                                                                     |
| Training LR                | `0.0002`                                                                |
| Optimizer                  | Fresh each cycle; **do not** resume optimizer state                     |
| `OPENING_WALLS_PLACED_MAX` | `7`                                                                     |
| Pool                       | Already running                                                         |
| Coordinator                | Restarted with repair env after failed-trigger fix                      |
| Tests                      | 19 passing (label + weight diagnostics + coordinator failure semantics) |

---

## OVERNIGHT TASK: Run Exactly One Controlled Repair Epoch (Validation Epoch)

The first repair epoch is the **validation epoch**. Collect evidence before any continuation decision.

### 1. Verify Before Running

- Confirm `OPENING_WALLS_PLACED_MAX` exists in `training/label_weights.py` and imports correctly:
  ```powershell
  $env:PYTHONPATH = "training"
  python -c "from label_weights import OPENING_WALLS_PLACED_MAX; print(OPENING_WALLS_PLACED_MAX)"
  ```
- Run:
  ```powershell
  python -m pytest training/tests/test_epoch_weight_diagnostics.py training/tests/test_label_resolution.py training/tests/test_training_coordinator_failures.py -q
  ```
- Confirm all tests pass.
- Check the failed run did **not** leave:
  - a candidate checkpoint treated as complete,
  - a partial cache marked valid,
  - an advanced epoch number (epoch 38 must **not** be in accepted chain),
  - incomplete diagnostics mistaken for final output.

## Known Blockers Fixed (2026-07-08)

| Issue                                             | Fix                                                                                                       |
| ------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `OPENING_WALLS_PLACED_MAX` NameError              | Restored constant `= 7`                                                                                   |
| `train_failed` consumed 450-game trigger          | Coordinator preserves `last_train_games_rowid`, releases pending claim, backoff retry                     |
| Tiny epoch slices (247–682 rows) on games trigger | Games trigger now uses `max(claimed_count, epoch_size)`; repair mode enables `--stream-full-active-epoch` |
| Stale h48 `best.pt` broke validation              | Repair mode deletes `best.pt` before each cycle; validation failures rollback cleanly                     |
| Binary rows in JSON path                          | Skip non-UTF8 `position_data` in `_featurize_records`                                                     |

- Restored parent weights from epoch 37
- Removed `pending_usage_keys.json` and partial `cycle_0038_candidate.*`
- Re-armed games trigger (`games_since_last_train = 450`)
- Released pending claim (+14976)

### 2. Failed-Trigger Accounting (Operational Only)

A trainer crash must **not** consume the 450-game trigger.

Required behavior (implemented in `training/training_coordinator.py`):

- preserve `games_since_last_train` on `train_failed`
- preserve parent weights
- do not advance epoch
- do not mark dataset snapshot consumed (`release_pending_claim`)
- retry after short backoff (`STREAM_TRAIN_FAILED_BACKOFF_SEC`, default 90s)

Manual re-arm if needed (run while coordinator is running — it reloads these keys each poll):

```powershell
$env:PYTHONPATH = "training"
python training/tools/retry_failed_training.py
```

Coordinator reloads `last_train_games_rowid` and `train_failed_retry_after` from the state file each poll so external retry scripts take effect without restart.

### 3. Retry Immediately

Do **not** wait for another 450 games after a `train_failed` crash. The trigger was re-armed; coordinator should fire on next poll.

Run exactly one repair epoch with:

- LR `0.0002`
- fresh optimizer
- corrected weighted resolver
- phase quota sampling
- hash-based validation split

### 4. Failure Handling

If obvious code/configuration blocker:

- fix **only** the direct blocker
- rerun relevant tests
- retry at most **twice**

If CUDA/VRAM OOM:

- reduce batch size only
- do not alter model, labels, weighting, or dataset

If NaNs, corrupted cache, unexplained loss explosion, or repeated failure:

- **stop training**
- preserve all logs and artifacts
- do not guess or automatically retune anything

### 5. After Successful Training — Collect Evidence

- fixed-node match result against epoch 37
- opening move evaluation for plies 0–20
- `epoch_weight_diagnostics_0001.json`
- `epoch_diagnostics_0001.json`
- prediction/label distribution
- global phase row share and loss-mass share
- per-tier row share and loss-mass share

**Evaluation priority:**

1. Fixed-node strength vs epoch 37
2. Opening quality at plies 0–20
3. Tactical/endgame regressions
4. Value calibration and saturation
5. Weighted validation loss

**Do not reject or accept based only on validation loss.**

Interpret diagnostics:

- Ishtar 3% rows / 10–15% loss mass may be intentional
- Suspicious: 2% rows / 45% loss mass; or titanium anchored 60% rows / 70% loss mass despite caps
- Opening: ~28% rows; ~32–36% loss mass reasonable with 1.25× multiplier; 50%+ is suspicious

### 6. Promotion Policy (First Epoch)

- Strength gate fails → quarantine candidate
- Strength passes but opening plies 0–20 clearly regresses → quarantine
- Opening eval cannot complete automatically → do not promote; leave for manual review
- Do **not** start a second repair epoch until first-epoch evidence is reviewed **unless** continuation criteria below are met

---

## OVERNIGHT CONTINUATION POLICY

The first repair epoch is the validation epoch.

If the first repair epoch:

- completes without errors,
- passes the fixed-node strength gate against epoch 37,
- shows no clear opening regression in plies 0–20,
- shows no tactical/endgame collapse,
- has sane prediction/value distributions,
- and diagnostics show no pathological tier or phase dominance,

then **promote it and continue training for the rest of the night**.

### Continuation Rules

1. Train only **one epoch at a time**.
2. Each new epoch must start from the **latest accepted** model, never from a quarantined candidate.
3. Keep the same repaired pipeline:
   - `STREAM_REPAIR_MODE=1`
   - `LR=0.0002`
   - fresh optimizer per cycle
   - unchanged label resolver
   - unchanged source weights
   - unchanged frequency formula
   - unchanged phase quota
4. Do **not** modify formulas, architecture, dataset semantics, or thresholds overnight.
5. Continue generating pool games between epochs.
6. Use the normal **450-game trigger** after the first forced retry.
7. **Maximum overnight continuation:**
   - up to **4 additional** successful repair epochs after the validation epoch,
   - or until approximately **10 hours** have elapsed,
   - **whichever comes first**

**Hard cap: five total repair epochs tonight** (1 validation + 4 continuations).

### After Every Epoch

- compare candidate against its direct accepted parent
- run the fixed-node strength gate
- inspect opening plies 0–20
- check value-output saturation
- read phase/tier loss-mass diagnostics

**Accept and continue only if:**

- strength gate passes
- opening quality is neutral or better
- no major tactical/endgame regression
- no NaNs or output collapse
- no suspicious diagnostic concentration

**Stop further training immediately if:**

- one candidate clearly loses to its parent
- **two consecutive** candidates fail the strength gate
- opening quality clearly regresses
- predictions saturate near ±1
- loss becomes NaN or explodes
- Titanium or one phase dominates weighted loss unexpectedly
- repeated trainer or pool failures occur

**On failure:**

- quarantine the candidate
- preserve the last accepted model
- keep the pool running if stable
- do not retune or invent a fix overnight

**Do not:**

- auto-switch back to `LR=0.001` overnight
- start from scratch overnight
- rewind to another checkpoint unless the first repair epoch fails badly and the reason is clearly damaged weights rather than an implementation bug

### Final Overnight Report

Must contain:

- all attempted epochs
- accepted and rejected model hashes
- parent-child match results
- opening evaluation results
- diagnostic JSON paths
- prediction distributions
- failures and fixes
- final accepted model
- whether training stopped early and why

---

## Key Paths

| Purpose           | Path                                                           |
| ----------------- | -------------------------------------------------------------- |
| Parent weights    | `training/runs/v16/accepted/epoch_0037.bin`                    |
| Coordinator state | `training/data/overnight_logs/training_coordinator_state.json` |
| Coordinator log   | `training/data/overnight_logs/training_coordinator.log`        |
| Retry helper      | `training/tools/retry_failed_training.py`                      |
| Start coordinator | `training/tools/start_training_coordinator_detached.ps1`       |
| Stop training     | `stop_training.ps1`                                            |

## Commands Reference

```powershell
# Status
Get-Content training\data\overnight_logs\training_coordinator_state.json

# Re-arm after train_failed
$env:PYTHONPATH = "training"
python training/tools/retry_failed_training.py

# Restart coordinator (repair env baked into start script)
$pid = (Get-Content training\data\overnight_logs\training_coordinator.pid).Trim()
Stop-Process -Id $pid -Force
powershell -ExecutionPolicy Bypass -File training\tools\start_training_coordinator_detached.ps1

# Tail log
Get-Content training\data\overnight_logs\training_coordinator.log -Tail 40
```

---

**Important:** Do not spend the night "improving" formulas after seeing one suspicious diagnostic number.
