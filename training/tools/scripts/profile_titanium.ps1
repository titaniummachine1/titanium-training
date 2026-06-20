[CmdletBinding()]
param(
    [switch]$IncludeScalar,
    [int]$Frequency = 997
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$EngineRoot = Join-Path $RepoRoot "engine"
$OutputRoot = Join-Path $RepoRoot "training\data\profiles"
$TargetRoot = Join-Path $EngineRoot "target-profile"
$Parser = Join-Path $PSScriptRoot "parse_flamegraph.py"

$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = [Security.Principal.WindowsPrincipal]::new($identity)
$isAdmin = $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    throw "cargo flamegraph needs elevated ETW access on Windows. Right-click PowerShell, choose 'Run as administrator', then run: & '$PSCommandPath'"
}
if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    throw "cargo is not available on PATH"
}
if (-not (cargo flamegraph --version 2>$null)) {
    throw "cargo-flamegraph is missing. Install it with: cargo install flamegraph --locked"
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$env:CARGO_TARGET_DIR = $TargetRoot
$env:CARGO_PROFILE_RELEASE_DEBUG = "true"
$env:RUSTFLAGS = "-C target-cpu=native"
Remove-Item Env:TITANIUM_ALLOW_SUBOPTIMAL -ErrorAction SilentlyContinue

Push-Location $EngineRoot
try {
    cargo build --release --bin titanium
    $profiles = @(
        @{ Name = "titanium-v15-start"; Args = @("genmove", "--engine", "titanium-v15", "--time", "3") },
        @{ Name = "titanium-v15-c3h"; Args = @("genmove", "--engine", "titanium-v15", "--time", "10", "e2", "e8", "e3", "e7", "e4", "e6", "c3h") },
        @{
            Name = "titanium-v15-wall-maze"
            Args = @(
                "genmove", "--engine", "titanium-v15", "--time", "10",
                "e2", "e8", "e3", "e7", "e4", "e6", "e3h", "e4h", "d4", "c4h", "e5v",
                "a5h", "h8h", "d6", "b5v", "f3v", "e7v", "c3h", "d7h", "b2v", "h6h"
            )
        }
    )
    foreach ($profile in $profiles) {
        $output = Join-Path $OutputRoot ($profile.Name + ".svg")
        Write-Host "`nProfiling $($profile.Name) -> $output" -ForegroundColor Cyan
        cargo flamegraph --deterministic --freq $Frequency --output $output --bin titanium -- @($profile.Args)
    }
    if ($IncludeScalar) {
        Remove-Item Env:RUSTFLAGS -ErrorAction SilentlyContinue
        $env:TITANIUM_ALLOW_SUBOPTIMAL = "1"
        $output = Join-Path $OutputRoot "titanium-v15-scalar-c3h.svg"
        Write-Host "`nProfiling titanium-v15-scalar-c3h -> $output" -ForegroundColor Cyan
        cargo flamegraph --deterministic --freq $Frequency --output $output --bin titanium -- `
            genmove --engine titanium-v15 --time 10 e2 e8 e3 e7 e4 e6 c3h
    }
} finally {
    Pop-Location
}

$svgs = Get-ChildItem -Path $OutputRoot -Filter "titanium-v15-*.svg" | Sort-Object Name
if ($svgs) {
    python $Parser @($svgs.FullName)
}
