# Continuous streaming NNUE — pre-start report + launch (existing DB-first architecture).
param(
    [switch]$SkipReport,
    [switch]$SkipSmoke,
    [int]$Threads = 4
)

$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Repo

$env:RUSTFLAGS = "-C target-cpu=native"
$env:PYTHONPATH = Join-Path $Repo "training"
$env:TITANIUM_GENERATION_ENGINE = "titanium-v17"
$env:STREAM_PRIOR_EPOCH_FRACTION = "0.30"
$env:STREAM_RETIRED_REPLAY_FRACTION = "0.05"
$env:NNUE_DEPLOY_EVERY = "0"
$env:ORACLE_PUSH_EACH_EPOCH = "0"

if (-not $SkipReport) {
    Write-Host "=== STREAMING NNUE PRE-START REPORT ==="
    python training/tools/start_streaming_nnue.py
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

if (-not $SkipSmoke) {
    Write-Host "Running streaming_db_smoke..."
    python training/tools/streaming_db_smoke.py
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host "Starting database-first runtime (importer + local pool + coordinator)..."
powershell -NoProfile -ExecutionPolicy Bypass -File "$Repo\start_overnight_pool.ps1"
Write-Host "Streaming NNUE started. Monitor: training\data\overnight_logs\"
