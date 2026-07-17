[CmdletBinding()]
param(
    [ValidateNotNullOrEmpty()][string] $OracleHost = "92.5.77.92",
    [ValidateNotNullOrEmpty()][string] $OracleUser = "ubuntu",
    [Parameter(Mandatory = $true)]
    [ValidateScript({ Test-Path -LiteralPath $_ -PathType Leaf })]
    [string] $SshKeyPath,
    [ValidateNotNullOrEmpty()][string] $EngineA = "titanium-v17-lmp-ace",
    [ValidateNotNullOrEmpty()][string] $EngineB = "titanium-v17",
    [ValidateRange(2, 1000000)][int] $Games = 200,
    [ValidateRange(0.1, 86400)][double] $ClockSec = 60.0,
    [ValidateRange(0, 128)][int] $OpenPlies = 4,
    [ValidateRange(1, 4096)][int] $MaxPlies = 512,
    [ValidateRange(0, [int]::MaxValue)][int] $Seed = 1337,
    [ValidateRange(1, 64)][int] $EngineThreads = 1,
    [ValidateNotNullOrEmpty()][string] $DeployedRoot = "/opt/titanium-game-factory",
    [ValidateNotNullOrEmpty()][string] $RemoteRoot = "/tmp/quoridor-oracle-deployed",
    [string] $RunId,
    [string] $LocalRunDir,
    [ValidateSet("launch", "status", "pull")][string] $Mode = "launch",
    [ValidateRange(1, 5)][int] $UploadAttempts = 3,
    [ValidateScript({ -not $_ -or (Test-Path -LiteralPath $_ -PathType Leaf) })]
    [string] $WeightsA,
    [ValidateScript({ -not $_ -or (Test-Path -LiteralPath $_ -PathType Leaf) })]
    [string] $WeightsB,
    [switch] $DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$script:ToolDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$script:MatchScript = Join-Path $script:ToolDir "parallel_engine_match.py"
$script:OpeningBook = Join-Path (Resolve-Path (Join-Path $script:ToolDir "..\..")) `
    "training\data\opening_book\non_titanium_10ply.json"
$script:SshKey = (Resolve-Path -LiteralPath $SshKeyPath).Path

function ConvertTo-PosixQuoted([string] $Value) {
    return "'" + ($Value -replace "'", "'\''") + "'"
}

function Invoke-Ssh([string] $RemoteCommand) {
    $sshArgs = @("-i", $script:SshKey, "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=15",
        "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3",
        "$($OracleUser)@$OracleHost", $RemoteCommand)
    if ($DryRun) {
        Write-Host ("DRY-RUN ssh " + (($sshArgs | ForEach-Object { ConvertTo-PosixQuoted $_ }) -join " "))
        return
    }
    & ssh @sshArgs
    if ($LASTEXITCODE -ne 0) { throw "ssh failed with exit code $LASTEXITCODE" }
}

function Copy-ToRemote([string] $LocalPath, [string] $RemotePath) {
    $scpArgs = @("-i", $script:SshKey, "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=15",
        "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3",
        $LocalPath, "$($OracleUser)@$OracleHost`:$RemotePath")
    if ($DryRun) {
        Write-Host ("DRY-RUN scp " + (($scpArgs | ForEach-Object { ConvertTo-PosixQuoted $_ }) -join " "))
        return
    }
    & scp @scpArgs
    if ($LASTEXITCODE -ne 0) { throw "scp failed with exit code $LASTEXITCODE" }
}

function Copy-ToRemoteVerified([string] $LocalPath, [string] $RemotePath) {
    $info = Get-Item -LiteralPath $LocalPath
    $sha = (Get-FileHash -LiteralPath $LocalPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $bytes = [int64]$info.Length
    $temp = "$RemotePath.uploading.$([guid]::NewGuid().ToString('N'))"
    for ($attempt = 1; $attempt -le $UploadAttempts; $attempt++) {
        try {
            Copy-ToRemote $LocalPath $temp
            $qTemp = ConvertTo-PosixQuoted $temp
            $qDest = ConvertTo-PosixQuoted $RemotePath
            $qSha = ConvertTo-PosixQuoted $sha
            $remoteShaCommand = '$(' + "sha256sum $qTemp | awk '{print `$1}'" + ')'
            $remoteBytesCommand = '$(' + "wc -c < $qTemp" + ')'
            $remoteCheck = 'test "' + $remoteShaCommand + '" = ' + $qSha +
                ' && test "' + $remoteBytesCommand + '" -eq ' + $bytes
            Invoke-Ssh "$remoteCheck && mv -f $qTemp $qDest"
            return
        }
        catch {
            if ($attempt -eq $UploadAttempts) {
                throw ("verified upload failed after {0} attempts for {1}: {2}" -f
                    $UploadAttempts, $LocalPath, $_.Exception.Message)
            }
            Write-Warning "Upload attempt $attempt/$UploadAttempts failed; retrying after 2 seconds."
            if (-not $DryRun) { Start-Sleep -Seconds 2 }
        }
    }
}

function Copy-FromRemote([string] $RemotePath, [string] $LocalPath) {
    if (-not $DryRun) {
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $LocalPath) | Out-Null
    }
    $scpArgs = @("-i", $script:SshKey, "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=15",
        "$($OracleUser)@$OracleHost`:$RemotePath", $LocalPath)
    if ($DryRun) {
        Write-Host ("DRY-RUN scp " + (($scpArgs | ForEach-Object { ConvertTo-PosixQuoted $_ }) -join " "))
        return
    }
    & scp @scpArgs
    if ($LASTEXITCODE -ne 0) { throw "scp pull failed with exit code $LASTEXITCODE" }
}

if (-not $RunId) {
    if ($Mode -ne "launch") { throw "-RunId is required for status and pull modes." }
    $RunId = "oracle_" + (Get-Date).ToUniversalTime().ToString("yyyyMMdd_HHmmss") + "_" +
        [System.IO.Path]::GetRandomFileName().Replace(".", "").Substring(0, 8)
}
if ($RunId -notmatch "^[A-Za-z0-9][A-Za-z0-9_-]{2,63}$") {
    throw "RunId must contain 3-64 letters, digits, underscore, or hyphen and start alphanumeric."
}
foreach ($label in @($EngineA, $EngineB)) {
    if ($label -notmatch "^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$") {
        throw "Engine labels may contain only letters, digits, underscore, dot, and hyphen: $label"
    }
}
if ($Games % 2 -ne 0) { throw "Games must be even because the harness runs mirrored opening pairs." }
if (-not $LocalRunDir) { $LocalRunDir = Join-Path $script:ToolDir ("runs\" + $RunId) }
$LocalRunDir = [System.IO.Path]::GetFullPath($LocalRunDir)
$remoteDir = "$RemoteRoot/$RunId"
$remoteScript = "$remoteDir/parallel_engine_match.py"
$remoteBook = "$remoteDir/non_titanium_10ply.json"
$remoteBinary = "$DeployedRoot/engine/target/release/titanium"
$remoteTraining = "$DeployedRoot/training"
$remoteOut = "$remoteDir/out"
$remoteStop = "$remoteDir/STOP"

if ($Mode -eq "launch") {
    foreach ($required in @($script:MatchScript, $script:OpeningBook)) {
        if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
            throw "Required upload file is missing: $required"
        }
    }
    $qRoot = ConvertTo-PosixQuoted $DeployedRoot
    $qBinary = ConvertTo-PosixQuoted $remoteBinary
    Invoke-Ssh "test -d $qRoot && test -x $qBinary && test -f $qRoot/training/engine_session.py && test -f $qRoot/training/titanium_training/paths.py && test -f $qRoot/training/self_play_overnight.py && mkdir -p $(ConvertTo-PosixQuoted $remoteDir) $(ConvertTo-PosixQuoted $remoteOut)"
    Copy-ToRemoteVerified $script:MatchScript $remoteScript
    Copy-ToRemoteVerified $script:OpeningBook $remoteBook
    $uploaded = @("parallel_engine_match.py", "non_titanium_10ply.json")
    $weightArgs = ""
    if ($WeightsA) {
        Copy-ToRemoteVerified $WeightsA "$remoteDir/weights_a.bin"
        $weightArgs += " --weights-a $(ConvertTo-PosixQuoted "$remoteDir/weights_a.bin")"
        $uploaded += "weights_a.bin"
    }
    if ($WeightsB) {
        Copy-ToRemoteVerified $WeightsB "$remoteDir/weights_b.bin"
        $weightArgs += " --weights-b $(ConvertTo-PosixQuoted "$remoteDir/weights_b.bin")"
        $uploaded += "weights_b.bin"
    }
    $manifest = [ordered]@{
        run_id = $RunId
        host = "$OracleUser@$OracleHost"
        deployed_root = $DeployedRoot
        remote_engine_binary = $remoteBinary
        engine_a = $EngineA
        engine_b = $EngineB
        games = $Games
        clock_sec = $ClockSec
        open_plies = $OpenPlies
        max_plies = $MaxPlies
        seed = $Seed
        engine_threads = $EngineThreads
        uploaded_files = $uploaded
        weights_a = $WeightsA
        weights_b = $WeightsB
        source_archive_uploaded = $false
        local_run_dir = $LocalRunDir
        launched_at_utc = (Get-Date).ToUniversalTime().ToString("o")
    }
    if (-not $DryRun) {
        New-Item -ItemType Directory -Force -Path $LocalRunDir | Out-Null
        $manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $LocalRunDir "launcher_manifest.json") -Encoding UTF8
    } else {
        Write-Host "DRY-RUN manifest: deployed native binary; no archive or build operation."
    }
    for ($offset = 4; $offset -le 16; $offset++) {
        $remoteLog = "$remoteDir/shard_$offset.log"
        $shardOut = "$remoteOut/shard_$offset"
        $command = @(
            "cd $(ConvertTo-PosixQuoted $remoteDir)",
            "env TITANIUM_GAME_FACTORY_ROOT=$(ConvertTo-PosixQuoted $DeployedRoot)",
            "TITANIUM_ENGINE_BIN=$qBinary",
            "TITANIUM_BOOK_MODE=off",
            "PYTHONPATH=$(ConvertTo-PosixQuoted $remoteTraining)",
            "python3 $(ConvertTo-PosixQuoted $remoteScript)",
            "--engine-a $(ConvertTo-PosixQuoted $EngineA) --engine-b $(ConvertTo-PosixQuoted $EngineB)",
            "--games $Games --clock-sec $ClockSec --open-plies $OpenPlies --max-plies $MaxPlies",
            "--seed $Seed --engine-threads $EngineThreads --workers 1 --shard-count 17 --shard-offset $offset --shard-span 1",
            "--opening-book $(ConvertTo-PosixQuoted $remoteBook) --out-dir $(ConvertTo-PosixQuoted $shardOut)",
            "--stop-file $(ConvertTo-PosixQuoted $remoteStop)$weightArgs > $(ConvertTo-PosixQuoted $remoteLog) 2>&1",
            "& echo `$!"
        ) -join " "
        Invoke-Ssh "nohup sh -c $(ConvertTo-PosixQuoted $command)"
        Write-Host "Launched deployed-binary shard offset $offset/17."
    }
    Write-Host "Oracle run $RunId launched using $remoteBinary; no local workers or build were started."
}
elseif ($Mode -eq "status") {
    Invoke-Ssh "for f in $(ConvertTo-PosixQuoted $remoteOut)/shard_*/status.json; do if [ -f `"`$f`" ]; then printf '%s\n' `"`$f`"; cat `"`$f`"; fi; done"
}
else {
    for ($offset = 4; $offset -le 16; $offset++) {
        Copy-FromRemote "$remoteOut/shard_$offset/status.json" (Join-Path $LocalRunDir "status_shard_$offset.json")
        Copy-FromRemote "$remoteOut/shard_$offset/results_shard_${offset}_1.jsonl" (Join-Path $LocalRunDir "results_shard_${offset}_1.jsonl")
        Copy-FromRemote "$remoteDir/shard_$offset.log" (Join-Path $LocalRunDir "shard_$offset.log")
    }
    Write-Host "Pulled deployed Oracle run $RunId into $LocalRunDir."
}
