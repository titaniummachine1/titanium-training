param(
    [int]$Threads = 4
)

$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$LogDir = Join-Path $Repo "training\data\overnight_logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$OpeningGate = Join-Path $LogDir "opening_exploration_enabled.json"
if (-not (Test-Path $OpeningGate)) {
    '{"enabled":true}' | Set-Content -Encoding ascii $OpeningGate
}
$OutLog = Join-Path $LogDir "local_game_pool.log"
$ErrLog = Join-Path $LogDir "local_game_pool_err.log"
$PidFile = Join-Path $LogDir "local_game_pool.pid"

if (Test-Path $PidFile) {
    $existingPid = [int](Get-Content $PidFile -Raw).Trim()
    $existingProc = Get-CimInstance Win32_Process -Filter "ProcessId=$existingPid" -EA SilentlyContinue
    if ($existingProc -and $existingProc.CommandLine -like "*local_game_pool.py*") {
        Write-Host "local_game_pool already running pid=$existingPid - skipping launch"
        exit 0
    }
}

$env:TRAINING_PREP_ONLY = "1"
$env:TITANIUM_GENERATION_ENGINE = "titanium-v17"
$env:RUSTFLAGS = "-C target-cpu=native"
$env:PYTHONPATH = Join-Path $Repo "training"
$env:PYTHONUNBUFFERED = "1"
# Simplified generation_matchup.py: no mixed external-opponent pool anymore,
# just a continuous ~30% current-vs-immediately-previous-accepted fraction at
# all times (self-play the rest). This is also the real strength signal read
# by streaming_epoch_validation.py's accept gate.
$env:STREAM_PRIOR_EPOCH_FRACTION = "0.30"

$py = (Get-Command python).Source
$script = Join-Path $Repo "training\local_game_pool.py"
$argList = @(
    "-u `"$script`"",
    "--threads $Threads --time 1 --nodes 550000",
    "--train-after-new-positions 0 --batch-games 999999",  # training owned by training_coordinator.py
    "--no-initial-epoch --no-parity",
    "--explore-chance 0",
    "--recent-replay-fraction 0.0 --recent-window-games 0"
) -join " "

$p = Start-Process -FilePath $py `
    -ArgumentList $argList `
    -WorkingDirectory $Repo `
    -RedirectStandardOutput $OutLog `
    -RedirectStandardError $ErrLog `
    -PassThru `
    -WindowStyle Hidden

$p.Id | Set-Content -Encoding ascii $PidFile
Write-Host "Detached local_game_pool pid=$($p.Id) threads=$Threads"
