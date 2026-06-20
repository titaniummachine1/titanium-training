# Operations

Day-to-day training pool operations on the Windows workstation.

## Overnight pool

```powershell
python training/supervise.py --start-pool
python training/supervise.py --once
```

Logs: `training/data/supervisor.log` (gitignored).

Stop workers:

```powershell
training/stop_training.cmd
```

## Scoreboard / manifest

```powershell
python training/manifest.py
```

Tracked metadata examples: `training/data/manifest.json`, `training/data/nnue_guard_state.json`.

## Position store administration

```powershell
python training/position_store.py --help
```

Runbook: [training/POSITION_STORE_RUNBOOK.md](../training/POSITION_STORE_RUNBOOK.md)

## Repository health

```powershell
python scripts/maintenance/repository_doctor.py
python scripts/maintenance/verify_docs.py
```

## Oracle vs local pool

| Environment | Focus |
| ----------- | ----- |
| Oracle | Value-NNUE smoke + long run from [ORACLE_DEPLOYMENT.md](ORACLE_DEPLOYMENT.md) |
| Local Windows | Game pool + micro-train via `run_nnue_cycle.py` / `supervise.py` |

Do not point the overnight pool at Oracle-only run directories.
