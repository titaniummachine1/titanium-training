<#
.SYNOPSIS
  100-game A/B: baseline 835c9dd vs broke-side e0d47d3.
  Local: 4 workers, shards 0..3. Oracle: 13 workers, shards 4..16.
  Engines forced RealTime (Windows) / best-effort RT+nice (Linux) with
  fixed affinity mask chosen once from the quietest local CPUs.
#>
[CmdletBinding()]
param(
    [string] $SshKeyPath = "$env:USERPROFILE\.ssh\oracle_titanium.key",
    [string] $OracleHost = "92.5.77.92",
    [string] $OracleUser = "ubuntu",
    [int] $Games = 100,
    [int] $MaxGames = 450,
    [double] $ClockSec = 60,
    [int] $Seed = 20260717,
    [string] $BaselineSha = "835c9dd",
    [string] $CandidateSha = "e0d47d3",
    [string] $EngineFlag = "titanium-v17",
    [string] $Note = "",
    [switch] $SkipLocalBuild,
    [string] $RunId = "",
    [string] $AffinityMask = "",
    [string[]] $ResumeFrom = @(),
    [int] $LocalWorkers = 4,
    [int] $OracleWorkers = 13,
    [double] $LocalGpt10 = 0,
    [double] $OracleGpt10 = 0,
    [switch] $LocalOnly,
    [switch] $OracleOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$EngineRoot = Join-Path $Repo "engine"
$BinDir = Join-Path $Repo "tools\binary_match\bin"
if (-not $RunId) { $RunId = "broke_side_ab_{0}_g{1}" -f (Get-Date -Format "yyyyMMdd_HHmmss"), $Games }
$LocalRunDir = Join-Path $Repo "tools\binary_match\runs\$RunId"
New-Item -ItemType Directory -Force -Path $BinDir, $LocalRunDir | Out-Null

$BaselineBin = Join-Path $BinDir "titanium_baseline_$BaselineSha.exe"
$CandidateBin = Join-Path $BinDir "titanium_broke_$CandidateSha.exe"
$BaselineLinuxBin = Join-Path $BinDir "titanium_baseline_$BaselineSha"
$CandidateLinuxBin = Join-Path $BinDir "titanium_broke_$CandidateSha"

function Build-WinSha([string]$Sha, [string]$OutExe) {
    Write-Host "WIN build $Sha -> $OutExe" -ForegroundColor Cyan
    Push-Location $EngineRoot
    try {
        git checkout --detach $Sha
        if ($LASTEXITCODE -ne 0) { throw "checkout $Sha failed" }
        $env:RUSTFLAGS = "-C target-cpu=native"
        Remove-Item Env:TITANIUM_ALLOW_SUBOPTIMAL -ErrorAction SilentlyContinue
        cargo build --release -p titanium --bin titanium
        if ($LASTEXITCODE -ne 0) { throw "build $Sha failed" }
        Copy-Item -Force (Join-Path $EngineRoot "target\release\titanium.exe") $OutExe
    }
    finally { Pop-Location }
}

function Build-LinuxSha([string]$Sha, [string]$OutBin) {
    Write-Host "LINUX build $Sha -> $OutBin" -ForegroundColor Cyan
    Push-Location $EngineRoot
    try {
        git checkout --detach $Sha
        if ($LASTEXITCODE -ne 0) { throw "checkout $Sha failed" }
        $env:RUSTFLAGS = "-C target-cpu=x86-64-v3"
        Remove-Item Env:TITANIUM_ALLOW_SUBOPTIMAL -ErrorAction SilentlyContinue
        $bundledZigDir = Join-Path $PSScriptRoot "zig"
        $bundledZigExe = Join-Path $bundledZigDir "zig.exe"
        if (Test-Path $bundledZigExe) {
            $env:PATH = $bundledZigDir + [IO.Path]::PathSeparator + $env:PATH
        }
        if (-not (Get-Command cargo-zigbuild -ErrorAction SilentlyContinue)) {
            throw "cargo-zigbuild is required for Linux ELF builds; install it with 'cargo install cargo-zigbuild'"
        }
        cargo zigbuild --release -p titanium --bin titanium --target x86_64-unknown-linux-gnu
        if ($LASTEXITCODE -ne 0) { throw "Linux build $Sha failed" }
        Copy-Item -Force (Join-Path $EngineRoot "target\x86_64-unknown-linux-gnu\release\titanium") $OutBin
    }
    finally { Pop-Location }
}

function Write-EngineTar([string]$Sha, [string]$OutTar) {
    Push-Location $EngineRoot
    try {
        git archive --format=tar.gz -o $OutTar $Sha
        if ($LASTEXITCODE -ne 0) { throw "git archive $Sha failed" }
    }
    finally { Pop-Location }
}

function Copy-ToOracle([string[]]$SshArgs, [string]$Target, [string]$Local, [string]$Remote) {
    & scp.exe @SshArgs $Local "${Target}:$Remote"
    if ($LASTEXITCODE -ne 0) { throw "scp failed $Local" }
}

function Get-QuietAffinityMask([int]$NeedCpus) {
    $n = (Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors
    $paths = 0..($n - 1) | ForEach-Object { "\Processor($_)\% Processor Time" }
    $c = Get-Counter -Counter $paths -SampleInterval 1 -MaxSamples 3
    $avgs = @{}
    0..($n - 1) | ForEach-Object { $avgs[$_] = New-Object System.Collections.Generic.List[double] }
    foreach ($sample in $c.CounterSamples) {
        if ($sample.Path -match '\\processor\((\d+)\)\\') {
            $avgs[[int]$Matches[1]].Add([double]$sample.CookedValue)
        }
    }
    $objects = @()
    for ($i = 0; $i -lt $n; $i++) {
        $objects += [pscustomobject]@{
            cpu = $i
            load = ($avgs[$i] | Measure-Object -Average).Average
        }
    }
    $sorted = @($objects | Sort-Object load)
    $pick = @($sorted | Select-Object -First $NeedCpus | ForEach-Object { [int]$_.cpu })
    $mask = [int64]0
    foreach ($cpu in $pick) { $mask = $mask -bor (1L -shl $cpu) }
    Write-Host ("Quiet CPUs={0} mask=0x{1:X}" -f ($pick -join ","), $mask)
    return @{ cpus = $pick; maskHex = ("0x{0:X}" -f $mask) }
}

# --- builds ---
if (-not $SkipLocalBuild -and -not $OracleOnly) {
    Build-WinSha $BaselineSha $BaselineBin
    Build-WinSha $CandidateSha $CandidateBin
    git -C $EngineRoot checkout --detach $CandidateSha | Out-Null
}
if (-not $LocalOnly) {
    if (-not $SkipLocalBuild -or -not (Test-Path $BaselineLinuxBin) -or -not (Test-Path $CandidateLinuxBin)) {
        Build-LinuxSha $BaselineSha $BaselineLinuxBin
        Build-LinuxSha $CandidateSha $CandidateLinuxBin
    }
}

if (-not $OracleOnly) {
    if (-not (Test-Path $BaselineBin) -or -not (Test-Path $CandidateBin)) {
        throw "Windows bins missing; build first"
    }
}
if (-not $LocalOnly) {
    if (-not (Test-Path $BaselineLinuxBin) -or -not (Test-Path $CandidateLinuxBin)) {
        throw "Linux bins missing; build first"
    }
}

if ($AffinityMask) {
    $aff = @{ cpus = @(); maskHex = $AffinityMask }
} else {
    $aff = Get-QuietAffinityMask 8
}
$env:TITANIUM_PROCESS_PRIORITY = "realtime"
$env:TITANIUM_AFFINITY_MASK = $aff.maskHex
$env:TITANIUM_BOOK_MODE = "off"

$shardPlanPy = Join-Path $PSScriptRoot "shard_plan.py"
$shardPlanPath = Join-Path $LocalRunDir "shard_plan.json"
$planArgs = @(
    $shardPlanPy, "--games", "$Games",
    "--local-workers", "$LocalWorkers", "--oracle-workers", "$OracleWorkers",
    "--out", $shardPlanPath
)
if ($LocalGpt10 -gt 0) { $planArgs += @("--local-gpt10", "$LocalGpt10") }
if ($OracleGpt10 -gt 0) { $planArgs += @("--oracle-gpt10", "$OracleGpt10") }
& python @planArgs
if ($LASTEXITCODE -ne 0) { throw "shard_plan.py failed" }
$plan = Get-Content $shardPlanPath -Raw | ConvertFrom-Json
Write-Host ("Shard plan: local {0} games (~{1} min) | oracle {2} games (~{3} min) | imbalance {4} min" -f `
    $plan.local.games, $plan.local.eta_minutes, $plan.oracle.games, $plan.oracle.eta_minutes, `
    $plan.finish_imbalance_minutes) -ForegroundColor Cyan

$meta = [ordered]@{
    run_id = $RunId
    baseline_sha = $BaselineSha
    candidate_sha = $CandidateSha
    engine_flag_both = $EngineFlag
    note = if ($Note) { $Note } else { "A=candidate($CandidateSha) B=baseline($BaselineSha); mirrored pairs" }
    games = $Games; max_games = $MaxGames; clock_sec = $ClockSec; seed = $Seed
    local_shard = $plan.local_shard
    oracle_shard = $plan.oracle_shard
    local_workers = $LocalWorkers
    oracle_workers = $OracleWorkers
    shard_plan = $plan
    affinity = $aff
    priority = "realtime"
    started_utc = (Get-Date).ToUniversalTime().ToString("o")
}
if (Test-Path $BaselineBin) {
    $meta.baseline_sha256 = (Get-FileHash $BaselineBin -Algorithm SHA256).Hash
    $meta.candidate_sha256 = (Get-FileHash $CandidateBin -Algorithm SHA256).Hash
}
$meta | ConvertTo-Json -Depth 6 | Set-Content (Join-Path $LocalRunDir "run_meta.json") -Encoding utf8

$matchPy = Join-Path $Repo "tools\binary_match\parallel_engine_match.py"
$book = Join-Path $Repo "training\data\opening_book\non_titanium_10ply.json"
$claustro = Join-Path $Repo "training\external_sources\claustrophobia\repo\runs\openings\human_openings.jsonl"

# --- LOCAL ---
if (-not $OracleOnly) {
    $localOut = Join-Path $LocalRunDir "local"
    New-Item -ItemType Directory -Force -Path $localOut | Out-Null
    $localLog = Join-Path $LocalRunDir "local_match.log"
    $errLog = Join-Path $LocalRunDir "local_match.err.log"
    $matchArgs = @(
        "`"$matchPy`"",
        "--engine-a", $EngineFlag, "--engine-b", $EngineFlag,
        "--engine-bin-a", "`"$CandidateBin`"", "--engine-bin-b", "`"$BaselineBin`"",
        "--games", "$Games", "--clock-sec", "$ClockSec", "--open-plies", "8", "--book-cap-plies", "12",
        "--seed", "$Seed", "--engine-threads", "1",
        "--workers", "$LocalWorkers",
        "--shard-count", "$($plan.shard_count)", "--shard-offset", "$($plan.local.offset)", "--shard-span", "$($plan.local.span)",
        "--no-early-elimination",
        "--opening-book", "`"$book`"", "--out-dir", "`"$localOut`""
    ) -join ' '
    foreach ($resume in $ResumeFrom) {
        if (Test-Path $resume) { $matchArgs += " --resume-from `"$resume`"" }
    }
    Write-Host "LOCAL start workers=$LocalWorkers shard=$($plan.local_shard)" -ForegroundColor Green
    $p = Start-Process -FilePath "python" -ArgumentList $matchArgs -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput $localLog -RedirectStandardError $errLog
    try { $p.PriorityClass = "High" } catch {}
    Set-Content (Join-Path $LocalRunDir "local.pid") $p.Id
    Write-Host "local pid=$($p.Id)"
}

# --- ORACLE ---
if (-not $LocalOnly) {
    if (-not (Test-Path $SshKeyPath)) { throw "missing SSH key $SshKeyPath" }
    $ssh = @("-i", $SshKeyPath, "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3")
    $target = "$OracleUser@$OracleHost"
    $job = "/var/lib/titanium-match-jobs/$RunId"
    $out = "$job/out"

    Write-Host "ORACLE pause factory + prepare job dir" -ForegroundColor Cyan
    & (Join-Path $Repo "tools\oracle_compute.ps1") -Mode factory-pause -SshKeyPath $SshKeyPath

    & ssh @ssh $target "sudo install -d -o $OracleUser -g $OracleUser /var/lib/titanium-match-jobs && mkdir -p $job/training/titanium_training/store $out"
    if ($LASTEXITCODE -ne 0) { throw "oracle mkdir failed" }

    Copy-ToOracle $ssh $target $matchPy "$job/parallel_engine_match.py"
    Copy-ToOracle $ssh $target $book "$job/non_titanium_10ply.json"
    Copy-ToOracle $ssh $target $claustro "$job/human_openings.jsonl"
    Copy-ToOracle $ssh $target (Join-Path $Repo "training\engine_session.py") "$job/training/engine_session.py"
    Copy-ToOracle $ssh $target (Join-Path $Repo "training\self_play_overnight.py") "$job/training/self_play_overnight.py"
    Copy-ToOracle $ssh $target (Join-Path $Repo "training\titanium_training\paths.py") "$job/training/titanium_training/paths.py"
    Copy-ToOracle $ssh $target (Join-Path $Repo "training\db_import.py") "$job/training/db_import.py"
    Copy-ToOracle $ssh $target (Join-Path $Repo "training\titanium_training\store\config.py") "$job/training/titanium_training/store/config.py"
    Copy-ToOracle $ssh $target (Join-Path $Repo "training\titanium_training\store\state.py") "$job/training/titanium_training/store/state.py"
    # empty __init__ so imports resolve
    & ssh @ssh $target "printf '' > $job/training/__init__.py; printf '' > $job/training/titanium_training/__init__.py; printf '' > $job/training/titanium_training/store/__init__.py"
    Copy-ToOracle $ssh $target $CandidateLinuxBin "$job/titanium_a"
    Copy-ToOracle $ssh $target $BaselineLinuxBin "$job/titanium_b"

    # Use a fixed 16-CPU mask on oracle; affinity selection is best-effort.
    $remoteAff = "0xffff"
    Write-Host "Oracle affinity mask=$remoteAff"

    $oracleScript = Join-Path $LocalRunDir "oracle_run.sh"
    $buildAndRun = @"
#!/usr/bin/env bash
set -eu -o pipefail
cd $job
chmod +x $job/titanium_a $job/titanium_b
sudo setcap 'cap_sys_nice=eip' $job/titanium_a $job/titanium_b || true
sha256sum $job/titanium_a $job/titanium_b | tee $job/binary.sha256
export TITANIUM_PROCESS_PRIORITY=realtime
export TITANIUM_AFFINITY_MASK=$remoteAff
export TITANIUM_BOOK_MODE=off
export TITANIUM_GAME_FACTORY_ROOT=$job
export TITANIUM_OPENING_BOOK=$job/non_titanium_10ply.json
export TITANIUM_CLAUSTRO_OPENINGS=$job/human_openings.jsonl
export PYTHONPATH=$job/training
nohup python3 $job/parallel_engine_match.py \
  --engine-a $EngineFlag --engine-b $EngineFlag \
  --engine-bin-a $job/titanium_a --engine-bin-b $job/titanium_b \
  --games $Games --clock-sec $ClockSec --open-plies 8 --book-cap-plies 12 --seed $Seed --engine-threads 1 \
  --workers $OracleWorkers --shard-count $($plan.shard_count) --shard-offset $($plan.oracle.offset) --shard-span $($plan.oracle.span) \
  --no-early-elimination \
  --opening-book $job/non_titanium_10ply.json \
  --out-dir $out --stop-file $job/STOP \
  > $job/coordinator.log 2>&1 &
echo `$! > $job/coordinator.pid
sleep 2
kill -0 `$(cat $job/coordinator.pid)
# setcap above enables sched_setscheduler from engine_session when permitted.
echo ORACLE_STARTED pid=`$(cat $job/coordinator.pid)
"@
    # Bash rejects CRLF (`pipefail\r`); force LF-only for the remote script.
    $unixScript = ($buildAndRun -replace "`r`n", "`n" -replace "`r", "`n")
    [System.IO.File]::WriteAllText($oracleScript, $unixScript)
    Copy-ToOracle $ssh $target $oracleScript "$job/oracle_run.sh"

    Write-Host "ORACLE start $OracleWorkers workers (shard $($plan.oracle_shard))..." -ForegroundColor Green
    & ssh @ssh $target "chmod +x $job/oracle_run.sh && nohup bash $job/oracle_run.sh > $job/oracle_run.log 2>&1 &"
    if ($LASTEXITCODE -ne 0) {
        & (Join-Path $Repo "tools\oracle_compute.ps1") -Mode factory-resume -SshKeyPath $SshKeyPath -ErrorAction SilentlyContinue
        throw "oracle start failed"
    }
    $deadline = (Get-Date).AddMinutes(2)
    $started = $false
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 10
        $probe = & ssh @ssh $target "grep -q ORACLE_STARTED $job/oracle_run.log 2>/dev/null && echo ok || echo wait"
        if (($probe -join "") -match "ok") { $started = $true; break }
        $fail = & ssh @ssh $target "tail -n 3 $job/oracle_run.log 2>/dev/null"
        Write-Host ("oracle start... {0}" -f (($fail | Select-Object -Last 1) -join ""))
    }
    if (-not $started) {
        & (Join-Path $Repo "tools\oracle_compute.ps1") -Mode factory-resume -SshKeyPath $SshKeyPath -ErrorAction SilentlyContinue
        throw "oracle start timed out (no ORACLE_STARTED in start log)"
    }
    Set-Content (Join-Path $LocalRunDir "oracle_job.txt") $job
    Write-Host "Oracle job $job running. Factory stays paused until match-pull/stop."
}

Write-Host ""
$gatePolicy = [ordered]@{
    candidate = "A=$CandidateSha (broke)"
    baseline = "B=$BaselineSha"
    initial_games = 100
    current_games = $Games
    max_games = $MaxGames
    mirrored_pairs = $true
    wilson = "95% Wilson; successes=A wins + 0.5*draws"
    rules = @("KEEP: Wilson LB >= 0.5", "REJECT: Wilson UB < 0.5",
        "EXTEND: even ceil(current*1.5), capped at max_games",
        "At cap: KEEP iff score_a >= 0.5")
    no_early_elimination = $true
}
$gatePolicy | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $LocalRunDir "gate_policy.json") -Encoding utf8
Write-Host "RUN_ID=$RunId"
Write-Host "DIR=$LocalRunDir"
Write-Host "A=candidate $CandidateSha (broke)  B=baseline $BaselineSha"
Write-Host "Monitor local: Get-Content '$LocalRunDir\local_match.log' -Wait -Tail 20"
Write-Host "Monitor oracle: ssh ... 'tail -f $job/coordinator.log'"
