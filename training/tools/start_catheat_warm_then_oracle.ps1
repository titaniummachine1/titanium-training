param(
    [string]$RepoRoot = "C:\gitProjects\Quoridor best AI",
    [int]$Epochs = 2,
    [int]$EpochSize = 1412018,
    [double]$RetiredReplayFraction = 0.05
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path $RepoRoot).Path
$Training = Join-Path $RepoRoot "training"
$LogDir = Join-Path $Training "data\overnight_logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
$RunDir = Join-Path $Training ("runs\catheat_warm_" + $stamp)
$StatusPath = Join-Path $LogDir "catheat_warm_then_oracle_state.json"
$TranscriptPath = Join-Path $LogDir ("catheat_warm_then_oracle_" + $stamp + ".log")

function Write-State {
    param([string]$State, [string]$Message = "")
    [ordered]@{
        updated_at = (Get-Date).ToUniversalTime().ToString("o")
        state = $State
        message = $Message
        run_dir = $RunDir
        epochs = $Epochs
        epoch_size = $EpochSize
        retired_replay_fraction = $RetiredReplayFraction
    } | ConvertTo-Json -Depth 3 | Set-Content -Encoding utf8 -LiteralPath $StatusPath
}

function Invoke-Step {
    param([string]$Title, [string[]]$Command)
    Write-Host ""
    Write-Host "=== $Title ==="
    Write-Host ($Command -join " ")
    & $Command[0] $Command[1..($Command.Length - 1)]
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed: $Title (exit $LASTEXITCODE)"
    }
}

Start-Transcript -Path $TranscriptPath -Append | Out-Null
try {
    Set-Location $RepoRoot
    $env:PYTHONPATH = $Training
    $env:PYTHONUNBUFFERED = "1"
    $env:RUSTFLAGS = "-C target-cpu=native"

    Write-State "WARM_TRAINING" "Training CAT heat net from current engine weights."
    Invoke-Step "Warm CAT-heat retrain" @(
        "python",
        "training\titanium_training\training\trainer.py",
        "--labels-db", "training\data\canonical\labels.db",
        "--weights", "engine\src\titanium\net_weights.bin",
        "--out-dir", $RunDir,
        "--epochs", "$Epochs",
        "--batch", "512",
        "--lr", "0.001",
        "--weight-decay", "0.00001",
        "--grad-clip", "1.0",
        "--stream-epoch-size", "$EpochSize",
        "--stream-featurize-chunk", "4096",
        "--stream-retired-replay-fraction", "$RetiredReplayFraction",
        "--val-split", "0.05",
        "--checkpoint-steps", "999999",
        "--patience", "0",
        "--cpu",
        "--no-parity",
        "--log-every", "100",
        "--log-interval-sec", "30"
    )

    $Best = Join-Path $RunDir "net_weights_best.bin"
    if (-not (Test-Path $Best)) {
        throw "Warm training completed without net_weights_best.bin"
    }

    Write-State "DEPLOYING_WARM_WEIGHTS" "Copying warm best weights into engine source."
    Copy-Item -LiteralPath $Best -Destination (Join-Path $RepoRoot "engine\src\titanium\net_weights.bin") -Force

    Write-State "REBUILDING_ENGINE" "Rebuilding native engine with warm weights."
    Push-Location (Join-Path $RepoRoot "engine")
    try {
        Invoke-Step "Rebuild native titanium" @(
            "cargo",
            "build",
            "--release",
            "--bin",
            "titanium"
        )
    }
    finally {
        Pop-Location
    }

    Write-State "STARTING_ORACLE_IMPORTER" "Starting Oracle importer."
    Invoke-Step "Start Oracle importer" @(
        "powershell",
        "-ExecutionPolicy", "Bypass",
        "-File", "training\tools\start_oracle_importer_detached.ps1"
    )

    Write-State "STARTING_LOCAL_POOL" "Starting local game pool."
    Invoke-Step "Start local game pool" @(
        "powershell",
        "-ExecutionPolicy", "Bypass",
        "-File", "training\tools\start_local_game_pool_detached.ps1"
    )

    Write-State "STARTING_TRAINING_COORDINATOR" "Starting 2048-position training coordinator."
    Invoke-Step "Start training coordinator" @(
        "powershell",
        "-ExecutionPolicy", "Bypass",
        "-File", "training\tools\start_training_coordinator_detached.ps1"
    )

    Write-State "RUNNING" "Warm retrain complete; Oracle/game generation/coordinator launched."
}
catch {
    Write-State "FAILED" $_.Exception.Message
    throw
}
finally {
    Stop-Transcript | Out-Null
}
