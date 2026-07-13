[CmdletBinding()]
param(
    [ValidateNotNullOrEmpty()]
    [string] $OracleHost = "92.5.77.92",

    [ValidateNotNullOrEmpty()]
    [string] $OracleUser = "ubuntu",

    [Parameter(Mandatory = $true)]
    [ValidateScript({ Test-Path -LiteralPath $_ -PathType Leaf })]
    [string] $SshKeyPath,

    [ValidateNotNullOrEmpty()]
    [string] $EngineA = "titanium-v17",

    [ValidateNotNullOrEmpty()]
    [string] $EngineB = "titanium-v16",

    [ValidateRange(2, 1000000)]
    [int] $Games = 200,

    [ValidateRange(0.1, 86400)]
    [double] $ClockSec = 60.0,

    [ValidateRange(0, 128)]
    [int] $OpenPlies = 4,

    [ValidateRange(1, 4096)]
    [int] $MaxPlies = 512,

    [ValidateRange(0, [int]::MaxValue)]
    [int] $Seed = 1337,

    [ValidateRange(1, 64)]
    [int] $EngineThreads = 1,

    [string] $EngineGitRef,

    [string[]] $ResumeFrom = @(),

    [ValidateNotNullOrEmpty()]
    [string] $RemoteRoot = "/tmp/quoridor-oracle",

    [string] $RunId,

    [string] $LocalRunDir,

    [ValidateSet("launch", "status", "pull")]
    [string] $Mode = "launch",

    [switch] $DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$script:ToolDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$script:RepoRoot = Resolve-Path (Join-Path $script:ToolDir "..\..")
$script:MatchScript = Join-Path $script:ToolDir "parallel_engine_match.py"
$script:EnginePackageFiles = @(
    "engine\Cargo.toml",
    "engine\Cargo.lock",
    "engine\build.rs"
)
$script:EnginePackageDirectories = @(
    "engine\build",
    "engine\src",
    "engine\benches"
)
$script:CargoConfigDirectories = @(
    @(
        ".cargo",
        "engine\.cargo"
    ) | Where-Object {
        Test-Path -LiteralPath (Join-Path $script:RepoRoot $_) -PathType Container
    }
)
$script:OpeningBook = Join-Path $script:RepoRoot "training\data\opening_book\non_titanium_10ply.json"
$script:TrainingFiles = @(
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

function Invoke-Ssh([string] $RemoteCommand) {
    $args = @(
        "-i", $script:SshKey,
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=15",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=3",
        "$($script:OracleUser)@$($script:OracleHost)",
        $RemoteCommand
    )
    if ($script:DryRun) {
        Write-Host ("DRY-RUN ssh " + (($args | ForEach-Object { ConvertTo-PosixQuoted $_ }) -join " "))
        return
    }
    & ssh @args
    if ($LASTEXITCODE -ne 0) {
        throw "ssh failed with exit code $LASTEXITCODE"
    }
}

function Copy-ToRemote([string] $LocalPath, [string] $RemotePath) {
    $args = @(
        "-i", $script:SshKey,
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=15",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=3",
        $LocalPath,
        "$($script:OracleUser)@$($script:OracleHost):$RemotePath"
    )
    if ($script:DryRun) {
        Write-Host ("DRY-RUN scp " + (($args | ForEach-Object { ConvertTo-PosixQuoted $_ }) -join " "))
        return
    }
    & scp @args
    if ($LASTEXITCODE -ne 0) {
        throw "scp failed with exit code $LASTEXITCODE"
    }
}

function Copy-FromRemote([string] $RemotePath, [string] $LocalPath) {
    $parent = Split-Path -Parent $LocalPath
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
    $args = @(
        "-i", $script:SshKey,
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=15",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=3",
        "$($script:OracleUser)@$($script:OracleHost):$RemotePath",
        $LocalPath
    )
    if ($script:DryRun) {
        Write-Host ("DRY-RUN scp " + (($args | ForEach-Object { ConvertTo-PosixQuoted $_ }) -join " "))
        return
    }
    & scp @args
    if ($LASTEXITCODE -ne 0) {
        throw "scp pull failed with exit code $LASTEXITCODE"
    }
}

function New-EngineArchive([string] $ArchivePath) {
    if ($script:EngineGitCommit) {
        & git -C (Join-Path $script:RepoRoot "engine") archive `
            --format=tar.gz --prefix=engine/ -o $ArchivePath $script:EngineGitCommit
        if ($LASTEXITCODE -ne 0) {
            throw "git archive failed for engine commit $($script:EngineGitCommit)"
        }
        return
    }
    # Exclude all Cargo artifact trees (target, target-profile, target-alt*, …).
    # Only --exclude=engine/target missed ~3.7 GB of debug/profile builds → 1.5 GB tarballs.
    $tarArgs = @(
        "-czf", $ArchivePath,
        "-C", $script:RepoRoot,
        "--exclude=engine/target",
        "--exclude=engine/target-*",
        "engine"
    ) + $script:CargoConfigDirectories
    & tar @tarArgs
    if ($LASTEXITCODE -ne 0) {
        throw "tar failed with exit code $LASTEXITCODE"
    }
}

function Copy-ToRemoteWithRetry([string] $LocalPath, [string] $RemotePath, [int] $Attempts = 3) {
    for ($attempt = 1; $attempt -le $Attempts; $attempt++) {
        try {
            Invoke-Ssh "rm -f $(ConvertTo-PosixQuoted $RemotePath)"
            Copy-ToRemote $LocalPath $RemotePath
            return
        }
        catch {
            if ($attempt -eq $Attempts) {
                throw "scp upload failed after $Attempts attempts: $($_.Exception.Message)"
            }
            Write-Warning "scp upload attempt $attempt/$Attempts failed; retrying after 2 seconds."
            if (-not $script:DryRun) {
                Start-Sleep -Seconds 2
            }
        }
    }
}

if (-not $RunId) {
    if ($Mode -ne "launch") {
        throw "-RunId is required for status and pull modes."
    }
    $RunId = "oracle_" + (Get-Date).ToUniversalTime().ToString("yyyyMMdd_HHmmss") + "_" +
    [System.IO.Path]::GetRandomFileName().Replace(".", "").Substring(0, 8)
}
if ($RunId -notmatch "^[A-Za-z0-9][A-Za-z0-9_-]{2,63}$") {
    throw "RunId must contain 3-64 letters, digits, underscore, or hyphen and start alphanumeric."
}

$script:SshKey = (Resolve-Path -LiteralPath $SshKeyPath).Path
$script:EngineGitCommit = $null
if ($EngineGitRef) {
    $script:EngineGitCommit = (& git -C (Join-Path $script:RepoRoot "engine") rev-parse --verify "$EngineGitRef`^{commit}").Trim()
    if ($LASTEXITCODE -ne 0 -or $script:EngineGitCommit -notmatch "^[0-9a-f]{40}$") {
        throw "Invalid engine git ref: $EngineGitRef"
    }
}
$resolvedResume = @($ResumeFrom | ForEach-Object { (Resolve-Path -LiteralPath $_).Path })
$script:RemoteDir = "$RemoteRoot/$RunId"
$script:RemoteTraining = "$script:RemoteDir/training"
if (-not $LocalRunDir) {
    $LocalRunDir = Join-Path $script:ToolDir ("runs\" + $RunId)
}
$LocalRunDir = [System.IO.Path]::GetFullPath($LocalRunDir)

if ($Mode -eq "launch") {
    $requiredPaths = @(
        $script:MatchScript,
        $script:OpeningBook,
        ($script:EnginePackageFiles | ForEach-Object { Join-Path $script:RepoRoot $_ }),
        ($script:EnginePackageDirectories | ForEach-Object { Join-Path $script:RepoRoot $_ })
    )
    if ($script:CargoConfigDirectories.Count -gt 0) {
        $requiredPaths += $script:CargoConfigDirectories | ForEach-Object {
            Join-Path $script:RepoRoot $_
        }
    }
    foreach ($required in $requiredPaths) {
        if (-not (Test-Path -LiteralPath $required)) {
            throw "Required package path is missing: $required"
        }
    }
    if ($Games % 2 -ne 0) {
        throw "Games must be even because the harness runs mirrored opening pairs."
    }
    $remotePython = "$script:RemoteDir/venv/bin/python3"
    $remoteScript = "$script:RemoteDir/parallel_engine_match.py"
    $remoteBinary = "$script:RemoteDir/engine/target/release/titanium"
    $remoteBook = "$script:RemoteTraining/data/opening_book/non_titanium_10ply.json"
    $remoteOut = "$script:RemoteDir/out"
    $remoteStop = "$script:RemoteDir/STOP"
    if (-not $DryRun) {
        New-Item -ItemType Directory -Force -Path $LocalRunDir | Out-Null
    }
    $engineArchive = Join-Path $LocalRunDir "engine-source.tar.gz"
    if ((Test-Path -LiteralPath $engineArchive -PathType Leaf) -and -not $DryRun) {
        throw "Refusing to overwrite existing local source archive: $engineArchive"
    }
    $engineHashes = [ordered]@{}
    foreach ($relative in $script:EnginePackageFiles) {
        $path = Join-Path $script:RepoRoot $relative
        $engineHashes[$relative.Replace("\", "/")] = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    $engineSourceFiles = @()
    foreach ($directory in ($script:EnginePackageDirectories + $script:CargoConfigDirectories)) {
        $root = Join-Path $script:RepoRoot $directory
        foreach ($file in (Get-ChildItem -LiteralPath $root -File -Recurse | Sort-Object FullName)) {
            $relative = $file.FullName.Substring($script:RepoRoot.Path.TrimEnd("\").Length + 1).Replace("\", "/")
            $engineSourceFiles += [ordered]@{
                path   = $relative
                sha256 = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
            }
        }
    }

    $manifest = [ordered]@{
        run_id               = $RunId
        host                 = "$OracleUser@$OracleHost"
        remote_dir           = $script:RemoteDir
        local_run_dir        = $LocalRunDir
        engine_source_root   = "engine"
        engine_package_files = $engineHashes
        engine_source_files  = $engineSourceFiles
        remote_engine_binary = $remoteBinary
        remote_build_command = "RUSTFLAGS='-C target-cpu=native' cargo build --release --locked --manifest-path engine/Cargo.toml --bin titanium"
        engine_a             = $EngineA
        engine_b             = $EngineB
        games                = $Games
        clock_sec            = $ClockSec
        open_plies           = $OpenPlies
        max_plies            = $MaxPlies
        seed                 = $Seed
        engine_threads       = $EngineThreads
        engine_git_commit    = $script:EngineGitCommit
        resume_inputs        = @($resolvedResume | ForEach-Object {
            [ordered]@{
                path = $_
                sha256 = (Get-FileHash -LiteralPath $_ -Algorithm SHA256).Hash.ToLowerInvariant()
            }
        })
        workers              = 1
        shard_count          = 17
        shard_offsets        = @(4..16)
        shard_span           = 1
        launched_at_utc      = (Get-Date).ToUniversalTime().ToString("o")
    }

    $quotedRemoteDir = ConvertTo-PosixQuoted $script:RemoteDir
    $remoteArchive = "$script:RemoteDir/engine-source.tar.gz"
    $remoteArchiveTemp = "$remoteArchive.uploading"
    Invoke-Ssh "mkdir -p $quotedRemoteDir/training/data/opening_book $quotedRemoteDir/training/titanium_training/store $quotedRemoteDir/out"
    if ($DryRun) {
        $tarArgs = @(
            "-czf", $engineArchive,
            "-C", $script:RepoRoot,
            "--exclude=engine/target",
            "engine"
        ) + $script:CargoConfigDirectories
        Write-Host ("DRY-RUN tar " + (($tarArgs | ForEach-Object { ConvertTo-PosixQuoted $_ }) -join " "))
    }
    else {
        New-EngineArchive $engineArchive
        $archiveInfo = Get-Item -LiteralPath $engineArchive
        $manifest.local_archive_name = Split-Path -Leaf $engineArchive
        $manifest.local_archive_sha256 = (Get-FileHash -LiteralPath $engineArchive -Algorithm SHA256).Hash.ToLowerInvariant()
        $manifest.local_archive_bytes = [int64]$archiveInfo.Length
        $manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $LocalRunDir "launcher_manifest.json") -Encoding UTF8
    }

    if ($DryRun) {
        $manifest.local_archive_name = Split-Path -Leaf $engineArchive
        $archiveSha256 = "<local-sha256>"
        $archiveBytes = "<local-bytes>"
        Write-Host "DRY-RUN archive SHA-256 and byte size will be recorded after creation."
    }
    else {
        $archiveSha256 = $manifest.local_archive_sha256
        $archiveBytes = [string]$manifest.local_archive_bytes
    }
    Copy-ToRemoteWithRetry $engineArchive $remoteArchiveTemp
    $remoteSha256 = ConvertTo-PosixQuoted $archiveSha256
    $remoteBytes = ConvertTo-PosixQuoted $archiveBytes
    $remoteTempQuoted = ConvertTo-PosixQuoted $remoteArchiveTemp
    $remoteArchiveQuoted = ConvertTo-PosixQuoted $remoteArchive
    Invoke-Ssh "test `"`$(sha256sum $remoteTempQuoted | awk '{print `$1}')`" = $remoteSha256 && test `"`$(wc -c < $remoteTempQuoted)`" -eq $remoteBytes && mv -f $remoteTempQuoted $remoteArchiveQuoted"
    Invoke-Ssh "tar -xf $(ConvertTo-PosixQuoted $remoteArchive) -C $quotedRemoteDir && rm -f $(ConvertTo-PosixQuoted $remoteArchive)"
    Copy-ToRemote $script:MatchScript $remoteScript
    Copy-ToRemote $script:OpeningBook $remoteBook
    $remoteResumeArgs = @()
    for ($resumeIndex = 0; $resumeIndex -lt $resolvedResume.Count; $resumeIndex++) {
        $remoteResume = "$script:RemoteDir/resume_$resumeIndex.jsonl"
        Copy-ToRemote $resolvedResume[$resumeIndex] $remoteResume
        $remoteResumeArgs += "--resume-from $(ConvertTo-PosixQuoted $remoteResume)"
    }
    foreach ($relative in $script:TrainingFiles) {
        $remotePath = "$script:RemoteDir/$($relative -replace '\\', '/')"
        $remoteParent = ($remotePath -replace "/[^/]+$", "")
        Invoke-Ssh "mkdir -p $(ConvertTo-PosixQuoted $remoteParent)"
        Copy-ToRemote (Join-Path $script:RepoRoot $relative) $remotePath
    }
    Invoke-Ssh "cd $quotedRemoteDir && RUSTFLAGS='-C target-cpu=native' cargo build --release --locked --manifest-path engine/Cargo.toml --bin titanium"
    Invoke-Ssh "test -x $(ConvertTo-PosixQuoted $remoteBinary) && sha256sum $(ConvertTo-PosixQuoted $remoteBinary) > $(ConvertTo-PosixQuoted "$script:RemoteDir/engine_artifact.sha256")"

    foreach ($offset in 4..16) {
        $remoteLog = "$script:RemoteDir/shard_$offset.log"
        $remoteOut = "$script:RemoteDir/out/shard_$offset"
        $command = @(
            "cd $(ConvertTo-PosixQuoted $script:RemoteDir)",
            "env TITANIUM_GAME_FACTORY_ROOT=$(ConvertTo-PosixQuoted $script:RemoteDir)",
            "TITANIUM_ENGINE_BIN=$(ConvertTo-PosixQuoted $remoteBinary)",
            "PYTHONPATH=$(ConvertTo-PosixQuoted $script:RemoteTraining)",
            "python3 $(ConvertTo-PosixQuoted $remoteScript)",
            "--engine-a $(ConvertTo-PosixQuoted $EngineA)",
            "--engine-b $(ConvertTo-PosixQuoted $EngineB)",
            "--games $Games",
            "--clock-sec $ClockSec",
            "--open-plies $OpenPlies",
            "--max-plies $MaxPlies",
            "--seed $Seed",
            "--engine-threads $EngineThreads",
            "--workers 1 --shard-count 17 --shard-offset $offset --shard-span 1",
            "--out-dir $(ConvertTo-PosixQuoted $remoteOut)",
            "--stop-file $(ConvertTo-PosixQuoted $remoteStop)",
            ($remoteResumeArgs -join " "),
            "> $(ConvertTo-PosixQuoted $remoteLog) 2>&1",
            "& echo \$!"
        ) -join " "
        Invoke-Ssh "nohup sh -c $(ConvertTo-PosixQuoted $command)"
        Write-Host "Launched remote shard offset $offset/17 (one worker)."
    }
    Write-Host "Oracle run $RunId launched; no local workers were started."
}
elseif ($Mode -eq "status") {
    $statusRoot = ConvertTo-PosixQuoted "$script:RemoteDir/out"
    Invoke-Ssh "for f in $statusRoot/shard_*/status.json; do if [ -f `"`$f`" ]; then printf '%s\n' `"`$f`"; cat `"`$f`"; fi; done"
}
else {
    New-Item -ItemType Directory -Force -Path $LocalRunDir | Out-Null
    foreach ($offset in 4..16) {
        Copy-FromRemote "$script:RemoteDir/out/shard_$offset/status.json" (Join-Path $LocalRunDir "status_shard_$offset.json")
        Copy-FromRemote "$script:RemoteDir/out/shard_$offset/results_shard_${offset}_1.jsonl" (Join-Path $LocalRunDir "results_shard_${offset}_1.jsonl")
        Copy-FromRemote "$script:RemoteDir/shard_$offset.log" (Join-Path $LocalRunDir "shard_$offset.log")
    }
    Write-Host "Pulled Oracle run $RunId into $LocalRunDir."
}
