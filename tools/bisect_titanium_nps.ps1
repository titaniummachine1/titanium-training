# Native Titanium search-speed bisect helper.
# Writes results to %USERPROFILE%\titanium_nps_bisect.csv (outside repo).
#
# Usage (from repo root, inside engine submodule for commit-aware builds):
#   git bisect start
#   git bisect bad HEAD
#   git bisect good <older-commit>
#   git bisect run powershell -NoProfile -ExecutionPolicy Bypass -File ..\tools\bisect_titanium_nps.ps1

param(
    [double]$GoodNps = 0,
    [double]$BadNps = 0,
    [double]$ThresholdNps = 0
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$engineDir = Join-Path $repoRoot "engine"
if (-not (Test-Path (Join-Path $engineDir "Cargo.toml"))) {
    $engineDir = Join-Path $repoRoot "engine"
}

$csv = Join-Path $env:USERPROFILE "titanium_nps_bisect.csv"
$commit = (git -C $engineDir rev-parse HEAD 2>$null)
if (-not $commit) { exit 125 }

$env:RUSTFLAGS = "-C target-cpu=native -C force-frame-pointers=yes"
Push-Location $engineDir
try {
    cargo build --release --bin search_bench 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) { exit 125 }
    $bench = Join-Path $engineDir "target\release\search_bench.exe"
    if (-not (Test-Path $bench)) { exit 125 }

    $json = & $bench time --sec 10 --runs 3 2>$null | Select-Object -Last 1
    if (-not $json) { exit 125 }
    $obj = $json | ConvertFrom-Json
    $nps = [double]$obj.median_nps
    $nodes = [int64]$obj.median_nodes
    $depth = [int]$obj.median_depth
    $line = "$(Get-Date -Format o),$commit,$nps,$nodes,$depth"
    Add-Content -Path $csv -Value $line

    if ($ThresholdNps -le 0) {
        # Auto threshold only when endpoints supplied via env.
        $thr = $env:TITANIUM_BISECT_THRESHOLD_NPS
        if ($thr) { $ThresholdNps = [double]$thr }
    }
    if ($ThresholdNps -le 0) {
        # Default conservative: treat < 60k as bad for startpos 10s pinned-TT bench.
        $ThresholdNps = 60000
    }
    if ($nps -ge $ThresholdNps) { exit 0 } else { exit 1 }
}
finally {
    Pop-Location
}
