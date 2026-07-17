param(
    [string]$RepoRoot = "C:\gitProjects\Quoridor best AI"
)

$ErrorActionPreference = "Stop"

$stamp = Get-Date -Format "yyyyMMddTHHmmssZ"
$runDir = Join-Path $RepoRoot ("training\runs\value_teacher_now_" + $stamp)
$logDir = Join-Path $runDir "logs"
$transcript = Join-Path $logDir ("train_" + $stamp + ".log")

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

Start-Transcript -Path $transcript -Append | Out-Null
try {
    Set-Location $RepoRoot
    $env:PYTHONPATH = "training"

    Write-Host "Run dir: $runDir"
    Write-Host "Dataset: training\data\teacher_dataset_good"
    Write-Host "Mode: direct packed-state featurization, 200000 samples"

    python training\nnue_cli.py verify-dataset
    if ($LASTEXITCODE -ne 0) {
        throw "active teacher dataset verification failed"
    }

    python -u training\titanium_training\training\trainer.py `
        --data training\data\teacher_dataset_good `
        --out-dir $runDir `
        --cpu `
        --epochs 3000 `
        --batch 512 `
        --lr 0.001 `
        --checkpoint-steps 1000 `
        --val-split 0.05 `
        --min-val 64 `
        --coverage-min 0.999 `
        --max-samples 200000 `
        --patience 100
    if ($LASTEXITCODE -ne 0) {
        throw "teacher training failed"
    }
}
finally {
    Stop-Transcript | Out-Null
}
