$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$LogDir = Join-Path $Repo "training\data\overnight_logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$OutLog = Join-Path $LogDir "continuous_pool.log"
$ErrLog = Join-Path $LogDir "continuous_pool_err.log"
$PidFile = Join-Path $LogDir "continuous_pool.pid"
$TokenFile = Join-Path $env:LOCALAPPDATA "titanium-oracle-api-token"

if (-not (Test-Path $TokenFile)) {
    throw "Oracle token missing: $TokenFile"
}
$token = (Get-Content $TokenFile -Raw).Trim()

$env:RUSTFLAGS = "-C target-cpu=native"
$env:PYTHONPATH = Join-Path $Repo "training"
$env:PYTHONUNBUFFERED = "1"

# Legacy coupled pool is deprecated — use start_oracle_importer_detached.ps1 +
# start_local_game_pool_detached.ps1.  Stop zombie continuous_pool processes
# without blindly deleting the lock file.
$killScript = Join-Path $Repo "training\tools\kill_legacy_pool_processes.py"
python $killScript

$py = (Get-Command python).Source
$script = Join-Path $Repo "training\local_game_pool.py"
$argList = @(
    "-u `"$script`"",
    "--threads 8 --time 5 --nodes 200000",
    "--train-after-new-positions 2048 --position-trigger --batch-games 999999",
    "--no-initial-epoch --no-parity --same-net-pct 0.7",
    "--recent-replay-fraction 0.6 --recent-window-games 256",
    "--opening-exploration",
    "--opening-temperature-initial 1.0 --opening-temperature-after-ply4 0.7",
    "--opening-temperature-decay-per-ply 0.08 --opening-temperature-min-while-known 0.15",
    "--opening-exploration-max-ply 16 --novel-prefix-temperature 0.0",
    "--explore-max-loss-cp 80 --explore-candidate-count 6 --explore-top-n 4",
    "--explore-wall-bonus-cp 12 --opening-prob-floor 0.02 --opening-pct 0.05",
    "--opening-line `"e2,e8,e3,e7,e4,e6,h3h,d6h,d3h,f3v`""
) -join " "

$OutLog = Join-Path $LogDir "local_game_pool.log"
$ErrLog = Join-Path $LogDir "local_game_pool_err.log"
$PidFile = Join-Path $LogDir "local_game_pool.pid"

$lock = Join-Path $LogDir "continuous_pool.lock.json"
if (Test-Path $lock) { Remove-Item $lock -Force -EA SilentlyContinue }
$legacyLock = Join-Path $LogDir "local_game_pool.lock.json"
if (Test-Path $legacyLock) { Remove-Item $legacyLock -Force -EA SilentlyContinue }

$p = Start-Process -FilePath $py `
    -ArgumentList $argList `
    -WorkingDirectory $Repo `
    -RedirectStandardOutput $OutLog `
    -RedirectStandardError $ErrLog `
    -PassThru `
    -WindowStyle Hidden

$p.Id | Set-Content -Encoding ascii $PidFile
# Back-compat for supervisor scripts still reading continuous_pool.pid
$PidFileLegacy = Join-Path $LogDir "continuous_pool.pid"
$p.Id | Set-Content -Encoding ascii $PidFileLegacy
Write-Host "Detached local_game_pool pid=$($p.Id) (legacy pid file updated)"
