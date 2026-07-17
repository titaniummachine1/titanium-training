<#
.SYNOPSIS
  Symbolicated CPU flamegraph for native Titanium search (profiling profile = release + debuginfo).

  Requires: cargo-flamegraph, elevated PowerShell (ETW on Windows).
  Install:  cargo install flamegraph --locked

  Output:   training/data/profiles/titanium-search-<name>-<timestamp>.svg
            training/data/profiles/titanium-search-<name>-<timestamp>-top20.txt

  Example (Run as Administrator):
    powershell -ExecutionPolicy Bypass -File tools\flamegraph_titanium_search.ps1
    powershell -ExecutionPolicy Bypass -File tools\flamegraph_titanium_search.ps1 -Seconds 30 -Name startpos-10s
#>
[CmdletBinding()]
param(
    [int]$Seconds = 30,
    [string]$Name = "startpos-profile",
    [string]$Position = "startpos",
    [int]$Frequency = 997,
    [switch]$UseGenmove,
    [switch]$Full
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$EngineRoot = Join-Path $RepoRoot "engine"
$OutputRoot = Join-Path $RepoRoot "training\data\profiles"
$TargetRoot = Join-Path $EngineRoot "target-profile"
$Parser = Join-Path $RepoRoot "training\tools\analysis\parse_flamegraph.py"
$Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$SvgName = "titanium-search-$Name-$Stamp.svg"
$SvgPath = Join-Path $OutputRoot $SvgName
$ReportPath = Join-Path $OutputRoot ("titanium-search-$Name-$Stamp-top20.txt")

$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = [Security.Principal.WindowsBuiltInRole]::Administrator
if (-not ([Security.Principal.WindowsPrincipal]::new($identity).IsInRole($principal))) {
    Write-Warning @"
cargo flamegraph on Windows needs Administrator (ETW).
Re-run: Right-click PowerShell -> Run as administrator, then:
  Set-Location '$RepoRoot'
  powershell -ExecutionPolicy Bypass -File tools\flamegraph_titanium_search.ps1 -Seconds $Seconds -Name $Name
"@
}

if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    throw "cargo not on PATH"
}
$fgVer = cargo flamegraph --version 2>&1
if ($LASTEXITCODE -ne 0) {
    throw "cargo-flamegraph missing. Install: cargo install flamegraph --locked"
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$env:CARGO_TARGET_DIR = $TargetRoot
Remove-Item Env:TITANIUM_ALLOW_SUBOPTIMAL -ErrorAction SilentlyContinue
$env:RUSTFLAGS = "-C target-cpu=native -C force-frame-pointers=yes"

Write-Host "Build: --profile profiling (release opts + debug=2, strip=false)" -ForegroundColor Cyan
Write-Host "RUSTFLAGS: $env:RUSTFLAGS" -ForegroundColor DarkGray
Write-Host "flamegraph: $fgVer" -ForegroundColor DarkGray

Push-Location $EngineRoot
try {
    if ($UseGenmove) {
        cargo build --profile profiling --bin titanium
        if ($LASTEXITCODE -ne 0) { throw "build failed" }
        $benchArgs = @(
            "genmove", "--engine", "titanium-v15", "--threads", "1", "--time", "$Seconds"
        )
        $binLabel = "titanium"
    }
    else {
        cargo build --profile profiling --features bench-instrument --bin search_bench
        if ($LASTEXITCODE -ne 0) { throw "build failed" }
        $benchArgs = @("profile", "--sec", "$Seconds", "--position", $Position)
        if ($Full) { $benchArgs += "--full" }
        $binLabel = "search_bench"
    }

    Write-Host "`nProfiling $binLabel ($Name, ${Seconds}s) -> $SvgPath" -ForegroundColor Green
    cargo flamegraph `
        --profile profiling `
        --deterministic `
        --freq $Frequency `
        --output $SvgPath `
        --bin $binLabel `
        -- @benchArgs

    if ($LASTEXITCODE -ne 0) {
        throw "cargo flamegraph failed (exit $LASTEXITCODE). Try elevated shell."
    }
    if (-not (Test-Path $SvgPath)) {
        throw "SVG not written: $SvgPath"
    }

    $meta = @{
        name = $Name
        seconds = $Seconds
        profile = "profiling"
        rustflags = $env:RUSTFLAGS
        binary = $binLabel
        svg = $SvgPath
        commit = (git rev-parse HEAD 2>$null)
        full_search = [bool]$Full
    } | ConvertTo-Json
    $metaPath = Join-Path $OutputRoot ("titanium-search-$Name-$Stamp-meta.json")
    Set-Content -Path $metaPath -Value $meta -Encoding utf8

    if ((Test-Path $Parser) -and (Get-Command python -ErrorAction SilentlyContinue)) {
        Write-Host "`nTop-20 exclusive stacks:" -ForegroundColor Cyan
        python $Parser $SvgPath --top 20 | Tee-Object -FilePath $ReportPath
        Write-Host "Report: $ReportPath" -ForegroundColor DarkGray
    }

    Write-Host "`nDONE" -ForegroundColor Green
    Write-Host "  SVG:    $SvgPath"
    Write-Host "  Meta:   $metaPath"
    Write-Host "  Open in browser (file://) or https://profiler.firefox.com (import profile if using samply later)"
}
finally {
    Pop-Location
}
