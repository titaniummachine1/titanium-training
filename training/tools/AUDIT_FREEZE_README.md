# Training pipeline freeze audit

Detects live workers that may mutate training state while `TRAINING_PREP_ONLY=1`.

## Usage

```powershell
$env:TRAINING_PREP_ONLY = "1"
python training/tools/audit_training_freeze.py
```

Exit codes: `0` = PASS, `2` = BLOCK, `3` = INVALID

## Manual stop (stale workers started before freeze)

```powershell
# Coordinator (example PID — read current from audit output or pid file)
Stop-Process -Id 24892 -Force -ErrorAction SilentlyContinue

# Generic: stop all registered training workers from pid files
@(
  "training\data\overnight_logs\training_coordinator.pid",
  "training\data\overnight_logs\local_game_pool.pid",
  "training\data\overnight_logs\oracle_importer.pid"
) | ForEach-Object {
  if (Test-Path $_) {
    $pid = [int](Get-Content $_ -Raw).Trim()
    Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
    Write-Host "Stopped pid=$pid from $_"
  }
}
```
