# Operations

Day-to-day training pool operations on the Windows workstation.

## Overnight pool

```powershell
python training/tools/operations/supervise.py --start-pool
python training/tools/operations/supervise.py --once
```

Logs: `training/data/supervisor.log` (gitignored).

Stop workers:

```powershell
training/tools/scripts/stop_training.cmd
```

## Scoreboard / manifest

```powershell
python training/tools/maintenance/manifest.py
```

Tracked metadata examples: `training/data/manifest.json`, `training/data/nnue_guard_state.json`.

## Position store administration

```powershell
python -m titanium_training.store.cli --help
```

```powershell
cd training
python -m titanium_training.store.cli --help
```

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
