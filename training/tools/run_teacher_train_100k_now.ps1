param(
    [string]$RepoRoot = "C:\gitProjects\Quoridor best AI"
)

$ErrorActionPreference = "Stop"

$stamp = Get-Date -Format "yyyyMMddTHHmmssZ"
$runDir = Join-Path $RepoRoot ("training\runs\value_teacher_100k_" + $stamp)
$cacheDir = Join-Path $RepoRoot "training\data\feature_cache_teacher_100k_now"
$logDir = Join-Path $runDir "logs"
$transcript = Join-Path $logDir ("train_100k_" + $stamp + ".log")

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Invoke-Step {
    param(
        [string]$Title,
        [string[]]$Command
    )
    Write-Host ""
    Write-Host "=== $Title ==="
    Write-Host ($Command -join " ")
    & $Command[0] $Command[1..($Command.Length - 1)]
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed: $Title (exit $LASTEXITCODE)"
    }
}

Start-Transcript -Path $transcript -Append | Out-Null
try {
    Set-Location $RepoRoot
    $env:PYTHONPATH = "training"

    Write-Host "Run dir: $runDir"
    Write-Host "Cache:   $cacheDir"
    Write-Host "Dataset: training\data\teacher_dataset_good"

    Invoke-Step "Verify active teacher dataset" @(
        "python",
        "training\nnue_cli.py",
        "verify-dataset"
    )

    if (Test-Path $cacheDir) {
        Remove-Item -LiteralPath $cacheDir -Recurse -Force
    }

    Invoke-Step "Build 100k teacher feature cache" @(
        "python",
        "training\build_feature_cache.py",
        "--force",
        "--dataset",
        "training\data\teacher_dataset_good",
        "--cache-dir",
        "training\data\feature_cache_teacher_100k_now",
        "--max-positions",
        "100000",
        "--workers",
        "2",
        "--batch-size",
        "1024",
        "--eval-timeout-sec",
        "1800"
    )

    Invoke-Step "Train from 100k teacher cache" @(
        "python",
        "-u",
        "training\titanium_training\training\trainer.py",
        "--data",
        "training\data\teacher_dataset_good",
        "--cache-dir",
        "training\data\feature_cache_teacher_100k_now",
        "--out-dir",
        $runDir,
        "--cpu",
        "--epochs",
        "3000",
        "--batch",
        "512",
        "--lr",
        "0.001",
        "--checkpoint-steps",
        "1000",
        "--val-split",
        "0.05",
        "--min-val",
        "64",
        "--patience",
        "100"
    )
}
finally {
    Stop-Transcript | Out-Null
}
