<#
.SYNOPSIS
  Adaptive A/B supervisor for the broke-side Titanium binary.

  A is e0d47d3 (candidate), B is 835c9dd (baseline).  The launcher is
  intentionally reused for each stage; its stable RunId makes each stage
  resume the same local and oracle JSONL files.
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
    [switch] $SkipLocalBuild
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$BinDir = Join-Path $Repo "tools\binary_match\bin"
$Launcher = Join-Path $PSScriptRoot "launch_broke_side_ab_100.ps1"
$Gate = Join-Path $PSScriptRoot "adaptive_ab_gate.py"
$RunId = "broke_side_adaptive_{0}" -f (Get-Date -Format "yyyyMMdd_HHmmss")
$RunDir = Join-Path $Repo "tools\binary_match\runs\$RunId"
$BaselineBin = Join-Path $BinDir "titanium_baseline_$BaselineSha.exe"
$CandidateBin = Join-Path $BinDir "titanium_broke_$CandidateSha.exe"

function Build-WinSha([string]$Sha, [string]$Output) {
    $engineRoot = Join-Path $Repo "engine"
    Push-Location $engineRoot
    try {
        git checkout --detach $Sha | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "checkout failed: $Sha" }
        $env:RUSTFLAGS = "-C target-cpu=native"
        Remove-Item Env:TITANIUM_ALLOW_SUBOPTIMAL -ErrorAction SilentlyContinue
        cargo build --release -p titanium --bin titanium
        if ($LASTEXITCODE -ne 0) { throw "native build failed: $Sha" }
        New-Item -ItemType Directory -Force -Path (Split-Path $Output) | Out-Null
        Copy-Item -Force (Join-Path $engineRoot "target\release\titanium.exe") $Output
    } finally { Pop-Location }
}

function Get-QuietAffinityMask([int]$NeedCpus) {
    $n = (Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors
    $paths = 0..($n - 1) | ForEach-Object { "\Processor($_)\% Processor Time" }
    $sample = Get-Counter -Counter $paths -SampleInterval 1 -MaxSamples 2
    $loads = @{}
    foreach ($row in $sample.CounterSamples) {
        if ($row.Path -match '\\processor\((\d+)\)\\') { $loads[[int]$Matches[1]] = [double]$row.CookedValue }
    }
    $mask = [int64]0
    foreach ($cpu in @($loads.GetEnumerator() | Sort-Object Value | Select-Object -First $NeedCpus)) {
        $mask = $mask -bor (1L -shl [int]$cpu.Key)
    }
    return ("0x{0:X}" -f $mask)
}

function Invoke-Stage([int]$Target) {
    & powershell -NoProfile -ExecutionPolicy Bypass -File $Launcher `
        -SshKeyPath $SshKeyPath -OracleHost $OracleHost -OracleUser $OracleUser `
        -Games $Target -MaxGames $MaxGames -ClockSec $ClockSec -Seed $Seed `
        -BaselineSha $BaselineSha -CandidateSha $CandidateSha -RunId $RunId `
        -AffinityMask $script:AffinityMask -SkipLocalBuild
    if ($LASTEXITCODE -ne 0) { throw "stage launch failed: $Target games" }
}

function Get-RemoteStatus([string]$Job, [string[]]$Ssh) {
    $target = "$OracleUser@$OracleHost"
    $text = & ssh @Ssh $target "if test -f '$Job/out/status.json'; then cat '$Job/out/status.json'; else echo '{}'; fi"
    if ($LASTEXITCODE -ne 0) { throw "cannot read oracle status" }
    try { return ($text -join "`n" | ConvertFrom-Json) } catch { return $null }
}

function Test-MatchTerminal([string]$Job, [string[]]$Ssh) {
    $localStatus = Join-Path $RunDir "local\status.json"
    if (-not (Test-Path $localStatus)) { return $false }
    try {
        $local = Get-Content $localStatus -Raw | ConvertFrom-Json
        if ([bool]$local.running) { return $false }
    } catch { return $false }
    try {
        $remote = Get-RemoteStatus $Job $Ssh
        return $null -ne $remote -and $remote.running -eq $false
    } catch { return $false }
}

function Resume-Factory {
    & (Join-Path $Repo "tools\oracle_compute.ps1") -Mode factory-resume `
        -SshKeyPath $SshKeyPath -ErrorAction SilentlyContinue
}

function Wait-Stage([string]$Job, [string[]]$Ssh) {
    $localStatus = Join-Path $RunDir "local\status.json"
    do {
        Start-Sleep -Seconds 5
        $localDone = $false
        if (Test-Path $localStatus) {
            try { $localDone = -not [bool]((Get-Content $localStatus -Raw | ConvertFrom-Json).running) } catch {}
        }
        $remote = Get-RemoteStatus $Job $Ssh
        $remoteDone = $remote -and ($remote.running -eq $false)
        Write-Host "stage status: local_done=$localDone oracle_done=$remoteDone"
    } until ($localDone -and $remoteDone)
}

$ssh = @("-i", $SshKeyPath, "-o", "BatchMode=yes",
    "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=20")
$job = "/var/lib/titanium-match-jobs/$RunId"
$oracleResult = Join-Path $RunDir "oracle\results_shard_4_13.jsonl"
$localResult = Join-Path $RunDir "local\results_shard_0_4.jsonl"

try {
    New-Item -ItemType Directory -Force -Path $BinDir, (Join-Path $RunDir "oracle") | Out-Null
    # -SkipLocalBuild means "reuse existing binaries"; missing artifacts still
    # have to be built because a runnable adaptive match is the contract.
    if (-not (Test-Path $BaselineBin)) { Build-WinSha $BaselineSha $BaselineBin }
    if (-not (Test-Path $CandidateBin)) { Build-WinSha $CandidateSha $CandidateBin }
    if (-not (Test-Path $BaselineBin) -or -not (Test-Path $CandidateBin)) {
        throw "missing local binaries; build them or remove -SkipLocalBuild"
    }
    $env:TITANIUM_PROCESS_PRIORITY = "realtime"
    $script:AffinityMask = Get-QuietAffinityMask 8
    $env:TITANIUM_AFFINITY_MASK = $script:AffinityMask
    $target = [Math]::Max(2, $Games)
    if ($target % 2) { $target++ }
    $stage = 1
    while ($true) {
        Invoke-Stage $target
        Wait-Stage $job $ssh
        $targetRemote = "$OracleUser@$OracleHost`:$job/out/results_shard_4_13.jsonl"
        & scp.exe @ssh $targetRemote $oracleResult
        if ($LASTEXITCODE -ne 0) { throw "oracle results pull failed" }
        $decisionJson = & python $Gate --results $localResult $oracleResult `
            --current-games $target --max-games $MaxGames
        if ($LASTEXITCODE -ne 0) { throw "adaptive gate failed" }
        $decision = $decisionJson | ConvertFrom-Json
        $decisionPath = Join-Path $RunDir ("stage_{0}_decision.json" -f $stage)
        $decisionJson | Set-Content $decisionPath -Encoding utf8
        Write-Host "stage $target decision=$($decision.decision): $($decision.reason)"
        if ($decision.decision -eq "KEEP") {
            "KEEP`n$($decision.reason)" | Set-Content (Join-Path $RunDir "PROMOTE.txt")
            Resume-Factory
            exit 0
        }
        if ($decision.decision -eq "REJECT") {
            "REJECT`n$($decision.reason)" | Set-Content (Join-Path $RunDir "REJECT.txt")
            Resume-Factory
            exit 1
        }
        $target = [int]$decision.next_games
        if ($target -gt $MaxGames) { $target = $MaxGames }
        $stage++
    }
} catch {
    # A failed launch may leave the local/oracle match running.  Never resume
    # the factory in that state; recover with launcher -OracleOnly if needed.
    if (Test-MatchTerminal $job $ssh) {
        Resume-Factory
    }
    throw
}
