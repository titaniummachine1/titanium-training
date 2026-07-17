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
    [switch] $SkipLocalBuild,
    [string] $RunId = "",
    [string] $AffinityMask = "",
    [string[]] $ResumeFrom = @(),
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
$BaselineTar = Join-Path $LocalRunDir "engine_$BaselineSha.tar.gz"
$CandidateTar = Join-Path $LocalRunDir "engine_$CandidateSha.tar.gz"

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
    Write-EngineTar $BaselineSha $BaselineTar
    Write-EngineTar $CandidateSha $CandidateTar
}

if (-not $OracleOnly) {
    if (-not (Test-Path $BaselineBin) -or -not (Test-Path $CandidateBin)) {
        throw "Windows bins missing; build first"
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

$meta = [ordered]@{
    run_id = $RunId
    baseline_sha = $BaselineSha
    candidate_sha = $CandidateSha
    engine_flag_both = $EngineFlag
    note = "A=candidate(broke) B=baseline; mirrored pairs"
    games = $Games; max_games = $MaxGames; clock_sec = $ClockSec; seed = $Seed
    local_shard = "0+4/17"; oracle_shard = "4+13/17"
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
        "--games", "$Games", "--clock-sec", "$ClockSec", "--open-plies", "4",
        "--seed", "$Seed", "--engine-threads", "1",
        "--workers", "4", "--shard-count", "17", "--shard-offset", "0", "--shard-span", "4",
        "--no-early-elimination",
        "--opening-book", "`"$book`"", "--out-dir", "`"$localOut`""
    ) -join ' '
    foreach ($resume in $ResumeFrom) {
        if (Test-Path $resume) { $matchArgs += " --resume-from `"$resume`"" }
    }
    Write-Host "LOCAL start workers=4 shards=0..3" -ForegroundColor Green
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

    & ssh @ssh $target "sudo install -d -o $OracleUser -g $OracleUser /var/lib/titanium-match-jobs && mkdir -p $job/training/titanium_training/store $out $job/build_a $job/build_b"
    if ($LASTEXITCODE -ne 0) { throw "oracle mkdir failed" }

    Copy-ToOracle $ssh $target $matchPy "$job/parallel_engine_match.py"
    Copy-ToOracle $ssh $target $book "$job/non_titanium_10ply.json"
    Copy-ToOracle $ssh $target (Join-Path $Repo "training\engine_session.py") "$job/training/engine_session.py"
    Copy-ToOracle $ssh $target (Join-Path $Repo "training\self_play_overnight.py") "$job/training/self_play_overnight.py"
    Copy-ToOracle $ssh $target (Join-Path $Repo "training\titanium_training\paths.py") "$job/training/titanium_training/paths.py"
    Copy-ToOracle $ssh $target (Join-Path $Repo "training\db_import.py") "$job/training/db_import.py"
    Copy-ToOracle $ssh $target (Join-Path $Repo "training\titanium_training\store\config.py") "$job/training/titanium_training/store/config.py"
    Copy-ToOracle $ssh $target (Join-Path $Repo "training\titanium_training\store\state.py") "$job/training/titanium_training/store/state.py"
    # empty __init__ so imports resolve
    & ssh @ssh $target "printf '' > $job/training/__init__.py; printf '' > $job/training/titanium_training/__init__.py; printf '' > $job/training/titanium_training/store/__init__.py"
    Copy-ToOracle $ssh $target $CandidateTar "$job/engine_a.tar.gz"
    Copy-ToOracle $ssh $target $BaselineTar "$job/engine_b.tar.gz"

    # Use a fixed 16-CPU mask on oracle; affinity selection is best-effort.
    $remoteAff = "0xffff"
    Write-Host "Oracle affinity mask=$remoteAff"

    $buildAndRun = @"
set -euo pipefail
cd $job
tar -xzf engine_a.tar.gz -C build_a
tar -xzf engine_b.tar.gz -C build_b
# git archive may extract flat or with top dir — detect Cargo.toml
find_cargo() { find "`$1" -maxdepth 3 -name Cargo.toml | head -1 | xargs -I{} dirname {}; }
A_SRC=`$(find_cargo build_a)
B_SRC=`$(find_cargo build_b)
export RUSTFLAGS='-C target-cpu=native'
( cd "`$A_SRC" && cargo build --release -p titanium --bin titanium )
( cd "`$B_SRC" && cargo build --release -p titanium --bin titanium )
cp "`$A_SRC/target/release/titanium" $job/titanium_a
cp "`$B_SRC/target/release/titanium" $job/titanium_b
chmod +x $job/titanium_a $job/titanium_b
sha256sum $job/titanium_a $job/titanium_b | tee $job/binary.sha256
export TITANIUM_PROCESS_PRIORITY=realtime
export TITANIUM_AFFINITY_MASK=$remoteAff
export TITANIUM_BOOK_MODE=off
export TITANIUM_GAME_FACTORY_ROOT=$job
export TITANIUM_OPENING_BOOK=$job/non_titanium_10ply.json
export PYTHONPATH=$job/training
nohup python3 $job/parallel_engine_match.py \
  --engine-a $EngineFlag --engine-b $EngineFlag \
  --engine-bin-a $job/titanium_a --engine-bin-b $job/titanium_b \
  --games $Games --clock-sec $ClockSec --open-plies 4 --seed $Seed --engine-threads 1 \
  --workers 13 --shard-count 17 --shard-offset 4 --shard-span 13 \
  --no-early-elimination \
  --opening-book $job/non_titanium_10ply.json \
  --out-dir $out --stop-file $job/STOP \
  > $job/coordinator.log 2>&1 &
echo `$! > $job/coordinator.pid
sleep 2
kill -0 `$(cat $job/coordinator.pid)
echo ORACLE_STARTED pid=`$(cat $job/coordinator.pid)
"@
    Write-Host "ORACLE remote build both SHAs + start 13 workers (this takes several minutes)..." -ForegroundColor Green
    & ssh @ssh $target $buildAndRun
    if ($LASTEXITCODE -ne 0) {
        & (Join-Path $Repo "tools\oracle_compute.ps1") -Mode factory-resume -SshKeyPath $SshKeyPath -ErrorAction SilentlyContinue
        throw "oracle start failed"
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
