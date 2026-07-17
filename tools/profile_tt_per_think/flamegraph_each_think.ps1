<#
.SYNOPSIS
  One Inferno flamegraph SVG per Titanium think (from collect_thinks.py JSONL).

  Requires Administrator (ETW). Uses --profile profiling search_bench (debug=2).

  Parallel: -Workers 8 assigns WHOLE GAMES to workers (game % Workers).
  Within a worker, that game's plies run sequentially.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File tools\profile_tt_per_think\flamegraph_each_think.ps1 `
    -ThinksJsonl ...\thinks.jsonl -OutDir ... -Workers 8
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ThinksJsonl,
    [Parameter(Mandatory = $true)]
    [string]$OutDir,
    [int]$Skip = 0,
    [int]$Limit = 0,
    [int]$Frequency = 997,
    [int]$Workers = 8,
    # Internal: which shard this process owns. -1 = parent launcher.
    [int]$WorkerIndex = -1
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$EngineRoot = Join-Path $RepoRoot "engine"
$TargetRoot = Join-Path $EngineRoot "target"
$FgDir = Join-Path $OutDir "flamegraphs"
$ThinksPath = (Resolve-Path $ThinksJsonl).Path
$ScriptPath = $MyInvocation.MyCommand.Path

function Ensure-Admin {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($identity)
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw "Administrator required for ETW flamegraphs. Re-run elevated."
    }
}

function Get-ProfilingExe {
    param([switch]$Build)
    Remove-Item Env:CARGO_TARGET_DIR -ErrorAction SilentlyContinue
    Remove-Item Env:TITANIUM_ALLOW_SUBOPTIMAL -ErrorAction SilentlyContinue
    $env:RUSTFLAGS = "-C target-cpu=native -C force-frame-pointers=yes"
    $Exe = Join-Path $TargetRoot "profiling\search_bench.exe"

    if ($Build -or -not (Test-Path $Exe)) {
        Write-Host "Build search_bench --profile profiling (debug=2)..." -ForegroundColor Cyan
        Push-Location $EngineRoot
        try {
            cargo build --profile profiling --bin search_bench
            if ($LASTEXITCODE -ne 0) { throw "search_bench profiling build failed" }
        }
        finally {
            Pop-Location
        }
    }

    if (-not (Test-Path $Exe)) { throw "missing $Exe" }
    $smoke = & $Exe think --ms 1 2>&1 | Out-String
    if ($smoke -match "unknown mode") {
        throw "search_bench lacks think mode: $Exe"
    }
    Write-Host "exe: $Exe mtime=$((Get-Item $Exe).LastWriteTime)" -ForegroundColor DarkGray
    return $Exe
}

function Run-Shard {
    param(
        [string]$Exe,
        [string[]]$Lines,
        [int]$Skip,
        [int]$End,
        [int]$Workers,
        [int]$WorkerIndex,
        [string]$FgDir,
        [int]$Frequency
    )

    $done = 0
    $failed = 0
    $skipped = 0
    for ($i = $Skip; $i -lt $End; $i++) {
        $rec = $Lines[$i] | ConvertFrom-Json
        $game = [int]$rec.game
        # Whole-game sharding: all plies of a game stay on one worker.
        if (($game % $Workers) -ne $WorkerIndex) { continue }

        $ply = [int]$rec.ply
        $side = [int]$rec.side
        $ms = [Math]::Max(1, [int][Math]::Round([double]$rec.allotted_ms))
        $movesArr = @($rec.moves)
        $movesStr = if ($movesArr.Count -eq 0) { "" } else { ($movesArr -join " ") }

        $stem = ("g{0:D3}_ply{1:D3}_s{2}" -f $game, $ply, $side)
        $svg = Join-Path $FgDir "$stem.svg"
        $sideJson = Join-Path $FgDir "$stem.json"

        if (Test-Path $svg) {
            Write-Host "[w$WorkerIndex g$game] skip $stem" -ForegroundColor DarkGray
            $skipped++
            continue
        }

        Write-Host "[w$WorkerIndex g$game ply$ply] $stem ms=$ms" -ForegroundColor Cyan
        $t0 = Get-Date
        $fgArgs = @("-o", $svg, "--freq", "$Frequency", "--", $Exe, "think", "--ms", "$ms")
        if ($movesStr -ne "") { $fgArgs += @("--moves", $movesStr) }

        $prevEap = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            & flamegraph @fgArgs 1>$null 2>$null
            $code = $LASTEXITCODE
        }
        finally {
            $ErrorActionPreference = $prevEap
        }
        if ($null -eq $code) { $code = 1 }

        $meta = [ordered]@{
            index = $i
            worker = $WorkerIndex
            game = $game
            ply = $ply
            side = $side
            allotted_ms = $ms
            exit_code = $code
            profile_wall_sec = [Math]::Round(((Get-Date) - $t0).TotalSeconds, 3)
            svg_exists = [bool](Test-Path $svg)
            finished_utc = (Get-Date).ToUniversalTime().ToString("o")
        }
        ($meta | ConvertTo-Json -Depth 5) | Set-Content -Path $sideJson -Encoding utf8

        if ($code -ne 0 -or -not (Test-Path $svg)) {
            Write-Warning "[w$WorkerIndex] FAILED $stem exit=$code"
            $failed++
        }
        else {
            $done++
        }
    }

    Write-Host "WORKER $WorkerIndex DONE ok=$done failed=$failed skipped=$skipped" -ForegroundColor Green
    exit 0
}

Ensure-Admin
if (-not (Get-Command flamegraph -ErrorAction SilentlyContinue)) {
    throw "flamegraph CLI missing. Install: cargo install flamegraph --locked"
}
New-Item -ItemType Directory -Force -Path $FgDir | Out-Null

$Workers = [Math]::Max(1, $Workers)
$lines = Get-Content -Path $ThinksPath
$total = $lines.Count
$end = if ($Limit -gt 0) { [Math]::Min($Skip + $Limit, $total) } else { $total }

# Parent: build once, spawn one worker per shard (whole games).
if ($WorkerIndex -lt 0 -and $Workers -gt 1) {
    $exe = Get-ProfilingExe -Build
    Write-Host "launching $Workers game-sharded workers (10 games -> worker = game % $Workers)" -ForegroundColor Green

    $procs = @()
    $logDir = Join-Path $OutDir "worker_logs"
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    for ($w = 0; $w -lt $Workers; $w++) {
        $arg = "-NoProfile -ExecutionPolicy Bypass -File `"$ScriptPath`" -ThinksJsonl `"$ThinksPath`" -OutDir `"$OutDir`" -Skip $Skip -Limit $Limit -Frequency $Frequency -Workers $Workers -WorkerIndex $w"
        $outLog = Join-Path $logDir ("worker_{0}.out.log" -f $w)
        $errLog = Join-Path $logDir ("worker_{0}.err.log" -f $w)
        $p = Start-Process -FilePath "powershell.exe" -ArgumentList $arg -PassThru -WindowStyle Hidden `
            -RedirectStandardOutput $outLog -RedirectStandardError $errLog
        $procs += $p
        Write-Host "worker $w pid=$($p.Id) games: $w,$($w+8),... log=$outLog" -ForegroundColor DarkGray
    }

    while ($true) {
        $alive = @($procs | Where-Object { -not $_.HasExited })
        $svgsNow = @(Get-ChildItem -Path $FgDir -Filter "*.svg" -ErrorAction SilentlyContinue).Count
        Write-Host ("progress alive={0}/{1} svgs={2}/{3}" -f $alive.Count, $Workers, $svgsNow, $total) -ForegroundColor DarkGray
        if ($alive.Count -eq 0) { break }
        Start-Sleep -Seconds 15
    }

    $failedWorkers = 0
    foreach ($p in $procs) {
        $p.Refresh()
        if ($p.ExitCode -ne 0 -and $null -ne $p.ExitCode) {
            Write-Warning "worker pid=$($p.Id) exit=$($p.ExitCode)"
            $failedWorkers++
        }
    }

    $svgs = @(Get-ChildItem -Path $FgDir -Filter "*.svg" -ErrorAction SilentlyContinue).Count
    Write-Host "DONE workers=$Workers failed_workers=$failedWorkers svgs=$svgs/$total" -ForegroundColor Green
    if ($svgs -lt [Math]::Max(1, [int]($total * 0.5)) -and $failedWorkers -gt 0) { exit 1 }
    exit 0
}

# Child / single-worker path: never rebuild; reuse profiling exe.
$shard = if ($WorkerIndex -lt 0) { 0 } else { $WorkerIndex }
$exe = Get-ProfilingExe
Write-Host "shard=$shard/$Workers (owns games where game%$Workers==$shard) end=$end" -ForegroundColor DarkGray
Run-Shard -Exe $exe -Lines $lines -Skip $Skip -End $end -Workers $Workers -WorkerIndex $shard -FgDir $FgDir -Frequency $Frequency
