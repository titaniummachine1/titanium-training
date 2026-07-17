param(
    [string]$RepoRoot = "C:\gitProjects\Quoridor best AI"
)

$ErrorActionPreference = "Stop"

$stamp = Get-Date -Format "yyyyMMddTHHmmssZ"
$runDir = Join-Path $RepoRoot ("training\runs\value_oracle_live_" + $stamp)
$logDir = Join-Path $runDir "refresh_logs"
$transcript = Join-Path $logDir ("teacher_refresh_" + $stamp + ".log")
$cacheDir = Join-Path $RepoRoot "training\data\feature_cache"
$datasetDir = Join-Path $RepoRoot "training\data\teacher_dataset_good"

New-Item -ItemType Directory -Force -Path $runDir | Out-Null
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

function Clear-FeatureCacheArtifacts {
    param([string]$Dir)
    if (-not (Test-Path $Dir)) {
        return
    }
    Get-ChildItem $Dir -Force -ErrorAction SilentlyContinue | ForEach-Object {
        if ($_.PSIsContainer) {
            Remove-Item -LiteralPath $_.FullName -Recurse -Force -ErrorAction SilentlyContinue
        } else {
            Remove-Item -LiteralPath $_.FullName -Force -ErrorAction SilentlyContinue
        }
    }
}

function Invoke-FeatureCacheBuild {
    param(
        [int]$Workers,
        [int]$BatchSize,
        [int]$TimeoutSec
    )
    Clear-FeatureCacheArtifacts -Dir $cacheDir
    Invoke-Step "Rebuild feature cache from teacher_dataset_good (workers=$Workers batch=$BatchSize timeout=$TimeoutSec)" @(
        "python",
        "training\build_feature_cache.py",
        "--force",
        "--dataset",
        "training\data\teacher_dataset_good",
        "--cache-dir",
        "training\data\feature_cache",
        "--workers",
        "$Workers",
        "--batch-size",
        "$BatchSize",
        "--eval-timeout-sec",
        "$TimeoutSec"
    )
}

Start-Transcript -Path $transcript -Append | Out-Null
try {
    Set-Location $RepoRoot
    $env:PYTHONPATH = "training"

    Write-Host "Run dir: $runDir"
    Write-Host "Dataset: $datasetDir"
    Write-Host "Cache:   $cacheDir"

    Invoke-Step "Verify active teacher dataset" @(
        "python",
        "training\nnue_cli.py",
        "verify-dataset"
    )

    $attempts = @(
        @{ Workers = 2; BatchSize = 1024; TimeoutSec = 1800 },
        @{ Workers = 1; BatchSize = 1024; TimeoutSec = 1800 },
        @{ Workers = 1; BatchSize = 512;  TimeoutSec = 2400 }
    )

    $built = $false
    foreach ($attempt in $attempts) {
        try {
            Invoke-FeatureCacheBuild -Workers $attempt.Workers -BatchSize $attempt.BatchSize -TimeoutSec $attempt.TimeoutSec
            $built = $true
            break
        } catch {
            Write-Warning $_
        }
    }

    if (-not $built) {
        throw "All feature-cache build attempts failed."
    }

    $resolvedConfig = [ordered]@{
        data = "training/data/teacher_dataset_good"
        teacher_dataset = "training/data/teacher_dataset_good"
        active_manifest_sha256 = "810fe8c5db540447aafd89399c5dcc3d8916ec800ade5bc97759a1bfd45bb08d"
        cache_dir = "training/data/feature_cache"
        out_dir = ($runDir.Replace($RepoRoot + "\", "") -replace "\\","/")
        epochs = 3000
        patience = 100
        batch = 512
        lr = 0.001
        checkpoint_steps = 1000
        val_split = 0.05
        min_val = 64
        seed = 0
        lmr_ready = $false
    }
    $resolvedConfig | ConvertTo-Json -Depth 4 | Set-Content -Path (Join-Path $runDir "resolved_config.json") -Encoding utf8

    Invoke-Step "Start long teacher training" @(
        "python",
        "training\titanium_training\training\trainer.py",
        "--data",
        "training\data\teacher_dataset_good",
        "--cache-dir",
        "training\data\feature_cache",
        "--out-dir",
        $runDir,
        "--resume",
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
