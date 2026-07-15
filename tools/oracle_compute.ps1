[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("deploy", "status", "match-start", "match-status", "match-pull", "match-stop", "factory-pause", "factory-resume")]
    [string] $Mode,
    [ValidateNotNullOrEmpty()][string] $OracleHost = "92.5.77.92",
    [ValidateNotNullOrEmpty()][string] $OracleUser = "ubuntu",
    [Parameter(Mandatory = $true)]
    [ValidateScript({ Test-Path -LiteralPath $_ -PathType Leaf })]
    [string] $SshKeyPath,
    [string] $RunId,
    [string] $LocalRunDir,
    [ValidateScript({ -not $_ -or (Test-Path -LiteralPath $_ -PathType Leaf) })]
    [string] $ResumeFrom,
    [ValidateNotNullOrEmpty()][string] $DeployedRoot = "/opt/titanium-game-factory",
    [ValidateNotNullOrEmpty()][string] $JobsRoot = "/var/lib/titanium-match-jobs",
    [ValidateNotNullOrEmpty()][string] $FactoryService = "titanium-game-factory",
    [ValidateNotNullOrEmpty()][string] $EngineA = "titanium-v17-lmp-ace",
    [ValidateNotNullOrEmpty()][string] $EngineB = "titanium-v17",
    [ValidateRange(2, 1000000)][int] $Games = 200,
    [ValidateRange(0.1, 86400)][double] $ClockSec = 60.0,
    [ValidateRange(0, 128)][int] $OpenPlies = 4,
    [ValidateRange(1, 4096)][int] $MaxPlies = 512,
    [ValidateRange(0, [int]::MaxValue)][int] $Seed = 1337,
    [ValidateRange(1, 64)][int] $EngineThreads = 1,
    [ValidateRange(1, 300)][int] $StopWaitSec = 30,
    [ValidateRange(1, 5)][int] $UploadAttempts = 3,
    [switch] $NoFactoryResume,
    [switch] $DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$matchScript = Join-Path $repo "tools\binary_match\parallel_engine_match.py"
$openingBook = Join-Path $repo "training\data\opening_book\non_titanium_10ply.json"
$buildBundle = Join-Path $repo "training\tools\build_oracle_bundle.ps1"
$deployWorker = Join-Path $repo "deploy_oracle_worker.ps1"
$sshKey = (Resolve-Path -LiteralPath $SshKeyPath).Path
$remoteBinary = "$DeployedRoot/engine/target/release/titanium"

# This is the complete Python import closure needed by parallel_engine_match.py.
# It intentionally excludes databases, weights, trainer code, and engine source.
$dependencyFiles = @(
    "training\engine_session.py",
    "training\self_play_overnight.py",
    "training\db_import.py",
    "training\titanium_training\__init__.py",
    "training\titanium_training\paths.py",
    "training\titanium_training\store\__init__.py",
    "training\titanium_training\store\config.py",
    "training\titanium_training\store\state.py"
)

function ConvertTo-PosixQuoted([string] $Value) {
    return "'" + ($Value -replace "'", "'\''") + "'"
}

function Get-SshArgs([string] $RemoteCommand) {
    return @(
        "-i", $sshKey, "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=15",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=3",
        "$OracleUser@$OracleHost", $RemoteCommand
    )
}

function Invoke-Ssh([string] $RemoteCommand) {
    $sshArguments = Get-SshArgs $RemoteCommand
    if ($DryRun) {
        Write-Host ("DRY-RUN ssh " + (($sshArguments | ForEach-Object { ConvertTo-PosixQuoted $_ }) -join " "))
        return ""
    }
    $output = & ssh @sshArguments
    if ($LASTEXITCODE -ne 0) { throw "ssh failed with exit code $LASTEXITCODE" }
    return ($output -join [Environment]::NewLine)
}

function Get-ScpArgs([string] $LocalPath, [string] $RemotePath) {
    return @(
        "-i", $sshKey, "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=15",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=3",
        $LocalPath, "$OracleUser@$OracleHost`:$RemotePath"
    )
}

function Copy-ToRemote([string] $LocalPath, [string] $RemotePath) {
    $scpArguments = Get-ScpArgs $LocalPath $RemotePath
    if ($DryRun) {
        Write-Host ("DRY-RUN scp " + (($scpArguments | ForEach-Object { ConvertTo-PosixQuoted $_ }) -join " "))
        return
    }
    & scp @scpArguments
    if ($LASTEXITCODE -ne 0) { throw "scp failed with exit code $LASTEXITCODE" }
}

function Copy-FromRemote([string] $RemotePath, [string] $LocalPath) {
    if (-not $DryRun) {
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $LocalPath) | Out-Null
    }
    # Reverse the source/destination order for a download.
    $scpArguments = @(
        "-i", $sshKey, "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=15",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=3",
        "$OracleUser@$OracleHost`:$RemotePath", $LocalPath
    )
    if ($DryRun) {
        Write-Host ("DRY-RUN scp " + (($scpArguments | ForEach-Object { ConvertTo-PosixQuoted $_ }) -join " "))
        return
    }
    & scp @scpArguments
    if ($LASTEXITCODE -ne 0) { throw "scp download failed with exit code $LASTEXITCODE" }
}

function Copy-ToRemoteVerified([string] $LocalPath, [string] $RemotePath) {
    $file = Get-Item -LiteralPath $LocalPath
    $sha = (Get-FileHash -LiteralPath $LocalPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $temp = "$RemotePath.uploading.$([guid]::NewGuid().ToString('N'))"
    for ($attempt = 1; $attempt -le $UploadAttempts; $attempt++) {
        try {
            Copy-ToRemote $LocalPath $temp
            $qTemp = ConvertTo-PosixQuoted $temp
            $qDest = ConvertTo-PosixQuoted $RemotePath
            $check = 'test "$(' +
                "sha256sum $qTemp | awk '{print `$1}'" +
                ')" = ' + (ConvertTo-PosixQuoted $sha) +
                ' && test "$(' + "wc -c < $qTemp" + ')" -eq ' +
                $file.Length + ' && mv -f ' + $qTemp + ' ' + $qDest
            Invoke-Ssh $check | Out-Null
            return
        }
        catch {
            if ($attempt -eq $UploadAttempts) { throw }
            if (-not $DryRun) { Start-Sleep -Seconds 2 }
        }
    }
}

function Resolve-RunId {
    if (-not $RunId) {
        if ($Mode -eq "match-start") {
            $RunId = "oracle_" + (Get-Date).ToUniversalTime().ToString("yyyyMMdd_HHmmss") + "_" +
                [guid]::NewGuid().ToString("N").Substring(0, 8)
        }
        else { throw "-RunId is required for $Mode." }
    }
    if ($RunId -notmatch "^[A-Za-z0-9][A-Za-z0-9_-]{2,63}$") {
        throw "RunId must contain 3-64 safe filename characters and start alphanumeric."
    }
    return $RunId
}

function Invoke-Deploy {
    if ($DryRun) {
        Write-Host "DRY-RUN powershell $buildBundle -OutputDir dist"
        Write-Host "DRY-RUN powershell $deployWorker -OracleHost $OracleHost -KeyPath $sshKey -User $OracleUser"
        Write-Host "DRY-RUN verify systemd service and native binary SHA256"
        return
    }
    & powershell -NoProfile -ExecutionPolicy Bypass -File $buildBundle
    if ($LASTEXITCODE -ne 0) { throw "Oracle bundle build failed with exit code $LASTEXITCODE" }
    & powershell -NoProfile -ExecutionPolicy Bypass -File $deployWorker `
        -OracleHost $OracleHost -KeyPath $sshKey -User $OracleUser
    if ($LASTEXITCODE -ne 0) { throw "Oracle worker deployment failed with exit code $LASTEXITCODE" }
    # The installer intentionally leaves the service stopped after replacing
    # the native binary. Start it before the active-service verification.
    Invoke-Ssh "sudo systemctl start $(ConvertTo-PosixQuoted $FactoryService)"
    $verify = Invoke-Ssh "systemctl is-active --quiet $(ConvertTo-PosixQuoted $FactoryService) && test -x $(ConvertTo-PosixQuoted $remoteBinary) && sha256sum $(ConvertTo-PosixQuoted $remoteBinary)"
    Write-Host "Oracle factory is active; native deployed binary SHA256: $verify"
}

function Invoke-Factory([ValidateSet("pause", "resume", "status")][string] $Action) {
    $service = ConvertTo-PosixQuoted $FactoryService
    if ($Action -eq "pause") {
        Invoke-Ssh "sudo systemctl stop $service"
        Write-Host "Oracle factory paused."
    }
    elseif ($Action -eq "resume") {
        Invoke-Ssh "sudo systemctl start $service"
        Write-Host "Oracle factory resumed."
    }
    else {
        $state = Invoke-Ssh "systemctl is-active $service; systemctl show -p ActiveState -p SubState $service"
        Write-Host $state
    }
}

function Get-RemoteJob([string] $Id) {
    return "$JobsRoot/$Id"
}

$needsRunId = $Mode -in @("match-start", "match-status", "match-pull", "match-stop")
if ($needsRunId) { $RunId = Resolve-RunId }
$remoteJob = Get-RemoteJob $RunId
$remoteOut = "$remoteJob/out"
$remotePid = "$remoteJob/coordinator.pid"
$remoteStop = "$remoteJob/STOP"
$remoteLog = "$remoteJob/coordinator.log"
$remoteStatus = "$remoteOut/status.json"
$remoteMeta = "$remoteJob/command.json"
$remoteSha = "$remoteJob/binary.sha256"

try {
    switch ($Mode) {
        "deploy" { Invoke-Deploy; break }
        "factory-pause" { Invoke-Factory "pause"; break }
        "factory-resume" { Invoke-Factory "resume"; break }
        "status" {
            Invoke-Ssh "systemctl is-active $(ConvertTo-PosixQuoted $FactoryService); test -x $(ConvertTo-PosixQuoted $remoteBinary) && sha256sum $(ConvertTo-PosixQuoted $remoteBinary)"
            break
        }
        "match-start" {
            foreach ($path in @($matchScript, $openingBook) + ($dependencyFiles | ForEach-Object { Join-Path $repo $_ })) {
                if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Required match upload file is missing: $path" }
            }
            if ($Games % 2 -ne 0) { throw "Games must be even because the harness runs mirrored opening pairs." }
            $localDir = if ($LocalRunDir) { [IO.Path]::GetFullPath($LocalRunDir) } else { Join-Path $repo "tools\binary_match\runs\$RunId" }
            $qJob = ConvertTo-PosixQuoted $remoteJob
            $qPid = ConvertTo-PosixQuoted $remotePid
            $qBinary = ConvertTo-PosixQuoted $remoteBinary
            $activeCheck = "if [ -e $qPid ] && kill -0 `$(cat $qPid) 2>/dev/null; then echo ACTIVE; exit 42; fi; test ! -e $(ConvertTo-PosixQuoted $remoteMeta)"
            $paused = $false
            try {
                Invoke-Factory "pause"
                $paused = $true
                Invoke-Ssh "test -x $qBinary && sha256sum $qBinary && $activeCheck && sudo install -d -o $(ConvertTo-PosixQuoted $OracleUser) -g $(ConvertTo-PosixQuoted $OracleUser) $(ConvertTo-PosixQuoted $JobsRoot) && mkdir -p $qJob/training/titanium_training/store $qJob/out"
                if ($ResumeFrom) {
                    Copy-ToRemoteVerified $ResumeFrom "$remoteOut/results_shard_4_13.jsonl"
                }
                Copy-ToRemoteVerified $matchScript "$remoteJob/parallel_engine_match.py"
                Copy-ToRemoteVerified $openingBook "$remoteJob/non_titanium_10ply.json"
                foreach ($relative in $dependencyFiles) {
                    $destination = "$remoteJob/$($relative.Replace('\', '/'))"
                    $remoteParent = $destination.Substring(0, $destination.LastIndexOf('/'))
                    Invoke-Ssh "mkdir -p $(ConvertTo-PosixQuoted $remoteParent)"
                    Copy-ToRemoteVerified (Join-Path $repo $relative) $destination
                }
                $metadata = [ordered]@{
                    run_id = $RunId; engine_a = $EngineA; engine_b = $EngineB
                    games = $Games; clock_sec = $ClockSec; open_plies = $OpenPlies
                    max_plies = $MaxPlies; seed = $Seed; engine_threads = $EngineThreads
                    workers = 13; shard_count = 17; shard_offset = 4; shard_span = 13
                    uploaded_files = @("parallel_engine_match.py", "non_titanium_10ply.json") + $dependencyFiles
                    resume_from = $ResumeFrom
                    source_archive_uploaded = $false; local_run_dir = $localDir
                    deployed_binary = $remoteBinary
                    started_at_utc = (Get-Date).ToUniversalTime().ToString("o")
                }
                $command = "cd $(ConvertTo-PosixQuoted $remoteJob) && env TITANIUM_GAME_FACTORY_ROOT=$(ConvertTo-PosixQuoted $remoteJob) TITANIUM_ENGINE_BIN=$qBinary TITANIUM_OPENING_BOOK=$(ConvertTo-PosixQuoted "$remoteJob/non_titanium_10ply.json") PYTHONPATH=$(ConvertTo-PosixQuoted "$remoteJob/training") python3 $(ConvertTo-PosixQuoted "$remoteJob/parallel_engine_match.py") --engine-a $(ConvertTo-PosixQuoted $EngineA) --engine-b $(ConvertTo-PosixQuoted $EngineB) --games $Games --clock-sec $ClockSec --open-plies $OpenPlies --max-plies $MaxPlies --seed $Seed --engine-threads $EngineThreads --workers 13 --shard-count 17 --shard-offset 4 --shard-span 13 --opening-book $(ConvertTo-PosixQuoted "$remoteJob/non_titanium_10ply.json") --out-dir $(ConvertTo-PosixQuoted $remoteOut) --stop-file $(ConvertTo-PosixQuoted $remoteStop)"
                $metadata.command = $command
                if (-not $DryRun) {
                    New-Item -ItemType Directory -Force -Path $localDir | Out-Null
                    $metadata | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $localDir "command.json") -Encoding UTF8
                }
                $metadataJson = $metadata | ConvertTo-Json -Compress
                $writeMetadata = "printf '%s' " + (ConvertTo-PosixQuoted $metadataJson) +
                    " > " + (ConvertTo-PosixQuoted $remoteMeta)
                Invoke-Ssh "sha256sum $qBinary > $(ConvertTo-PosixQuoted $remoteSha) && $writeMetadata && nohup sh -c $(ConvertTo-PosixQuoted ($command + " > " + (ConvertTo-PosixQuoted $remoteLog) + " 2>&1")) </dev/null >/dev/null 2>&1 & echo `$! > $qPid && sleep 1 && kill -0 `$(cat $qPid)"
                Write-Host "Oracle match $RunId started with one coordinator (13 workers, remote slots 4..16)."
                $paused = $false
            }
            finally {
                if ($paused) {
                    try { Invoke-Factory "resume" } catch { Write-Warning "Failed to resume factory after match-start failure: $($_.Exception.Message)" }
                }
            }
            break
        }
        "match-status" {
            Invoke-Ssh "if [ -f $(ConvertTo-PosixQuoted $remotePid) ] && kill -0 `$(cat $(ConvertTo-PosixQuoted $remotePid)) 2>/dev/null; then echo coordinator=ACTIVE; else echo coordinator=INACTIVE; fi; if [ -f $(ConvertTo-PosixQuoted $remoteStatus) ]; then cat $(ConvertTo-PosixQuoted $remoteStatus); else echo status.json=missing; fi; echo '--- log tail ---'; test -f $(ConvertTo-PosixQuoted $remoteLog) && tail -n 30 $(ConvertTo-PosixQuoted $remoteLog) || true"
            break
        }
        "match-pull" {
            Invoke-Ssh "if [ -f $(ConvertTo-PosixQuoted $remotePid) ] && kill -0 `$(cat $(ConvertTo-PosixQuoted $remotePid)) 2>/dev/null; then exit 42; fi"
            $archive = "$remoteJob/pull.tar.gz"
            Invoke-Ssh "tar -czf $(ConvertTo-PosixQuoted $archive) -C $(ConvertTo-PosixQuoted $remoteJob) --ignore-failed-read coordinator.pid coordinator.log command.json binary.sha256 STOP out"
            if (-not $LocalRunDir) { $LocalRunDir = Join-Path $repo "tools\binary_match\runs\$RunId" }
            $localDir = [IO.Path]::GetFullPath($LocalRunDir)
            $localArchive = Join-Path $localDir "oracle-results.tar.gz"
            Copy-FromRemote $archive $localArchive
            if (-not $DryRun) { tar -xzf $localArchive -C $localDir }
            if (-not $NoFactoryResume) { Invoke-Factory "resume" }
            Write-Host "Pulled completed Oracle match $RunId into $localDir and resumed factory."
            break
        }
        "match-stop" {
            Invoke-Ssh "touch $(ConvertTo-PosixQuoted $remoteStop)"
            $deadline = (Get-Date).ToUniversalTime().AddSeconds($StopWaitSec)
            do {
                $alive = Invoke-Ssh "if [ -f $(ConvertTo-PosixQuoted $remotePid) ] && kill -0 `$(cat $(ConvertTo-PosixQuoted $remotePid)) 2>/dev/null; then echo yes; else echo no; fi"
                if ($alive.Trim() -notmatch "yes") { break }
                if ((Get-Date).ToUniversalTime() -ge $deadline) { break }
                if (-not $DryRun) { Start-Sleep -Seconds 2 }
            } while ($true)
            $remaining = Invoke-Ssh "if [ -f $(ConvertTo-PosixQuoted $remotePid) ] && kill -0 `$(cat $(ConvertTo-PosixQuoted $remotePid)) 2>/dev/null; then echo yes; else echo no; fi"
            if ($remaining.Trim() -match "yes") {
                Invoke-Ssh "kill `$(cat $(ConvertTo-PosixQuoted $remotePid))"
                Write-Warning "Stop timeout reached; killed only recorded coordinator PID."
            }
            if (-not $NoFactoryResume) { Invoke-Factory "resume" }
            Write-Host "Oracle match $RunId stopped; factory resume requested."
            break
        }
    }
}
catch {
    throw
}
