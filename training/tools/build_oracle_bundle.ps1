param(
    [string]$OutputDir = "dist",
    [string]$Version = ""
)

# Oracle bundle — game-generator only.
#
# Included:
#   training/oracle_game_factory/   worker code, shell scripts, systemd unit
#   engine/src/                     Rust source (built on Oracle at install time)
#   engine/build/                   build helper code
#   engine/build.rs                 Cargo build script
#   engine/Cargo.toml               package manifest
#   engine/Cargo.lock               pinned dependencies for reproducible builds
#   ORACLE_SETUP.md
#
# Excluded (enforced):
#   engine/.git/                    git history — irrelevant and large
#   engine/worktrees/               git worktrees — irrelevant and large
#   engine/target*/                 build artifacts — must be built on Oracle natively
#   training/titanium_training/     PyTorch trainer — laptop only
#   training/tools/                 operational scripts — laptop only
#   training/data/                  any local data if present — never ship
#   training/db_import.py           canonical DB write — laptop only
#   training/sync_overnight_to_teacher.py  teacher dataset — laptop only
#   training/extend_teacher_dataset.py     teacher dataset — laptop only
#   training/self_play_overnight.py        superseded by oracle_game_factory
#   training/pool_state_io.py              pool state — laptop only
#   *.db  *.parquet  *.pt  *.bin  *.npy   data/weights — never ship in bundle
#   *.log  *.lock    *.pyc  *.pyd         noise

$ErrorActionPreference = "Stop"
$repo = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Push-Location $repo
try {
    $gitSha = if ($Version) { $Version } else { (git rev-parse --short=12 HEAD 2>$null) }
    if (-not $gitSha) { $gitSha = "nogit" }
    $stageRoot = Join-Path $repo "dist\oracle-worker-stage"
    $stage = Join-Path $stageRoot "oracle-worker"
    if (Test-Path $stageRoot) { Remove-Item $stageRoot -Recurse -Force }
    New-Item -ItemType Directory -Force $stage | Out-Null

    # ── 1. Oracle game factory (Python worker + shell scripts + systemd) ──────
    $gfSrc = Join-Path $repo "training\oracle_game_factory"
    $gfDst = Join-Path $stage "training\oracle_game_factory"
    New-Item -ItemType Directory -Force $gfDst | Out-Null
    robocopy $gfSrc $gfDst /E /XD __pycache__ .pytest_cache /XF *.pyc *.pyd *.db *.log | Out-Null
    if ($LASTEXITCODE -gt 7) { throw "robocopy oracle_game_factory failed ($LASTEXITCODE)" }
    $global:LASTEXITCODE = 0

    $systemdSrc = Join-Path $gfSrc "systemd"
    $systemdDst = Join-Path $stage "systemd"
    if (Test-Path $systemdSrc) {
        robocopy $systemdSrc $systemdDst /E /XF *.pyc *.pyd *.db *.log | Out-Null
        if ($LASTEXITCODE -gt 7) { throw "robocopy systemd failed ($LASTEXITCODE)" }
        $global:LASTEXITCODE = 0
    }

    # ── 2. Engine source (no .git, no worktrees, no target*) ─────────────────
    $engSrc = Join-Path $repo "engine"
    $engDst = Join-Path $stage "engine"
    New-Item -ItemType Directory -Force $engDst | Out-Null

    # Copy only the directories needed for a native cargo build --release
    foreach ($sub in @("src", "build", "benches")) {
        $s = Join-Path $engSrc $sub
        $d = Join-Path $engDst $sub
        robocopy $s $d /E /XD __pycache__ /XF *.pyc *.pyd *.db *.npy *.log | Out-Null
        if ($LASTEXITCODE -gt 7) { throw "robocopy engine/$sub failed ($LASTEXITCODE)" }
        $global:LASTEXITCODE = 0
    }

    # Compile-time embedded weights (include_bytes!); runtime self-play uses pushed generation files.
    foreach ($wf in @(
        "src\titanium\net_weights.bin",
        "src\titanium\net_weights_frozen.bin",
        "src\ace\net_weights.bin"
    )) {
        $src = Join-Path $engSrc $wf
        if (Test-Path $src) {
            $dst = Join-Path $engDst $wf
            New-Item -ItemType Directory -Force (Split-Path $dst -Parent) | Out-Null
            Copy-Item $src $dst -Force
        }
    }

    # Copy root engine files needed by Cargo
    foreach ($f in @("Cargo.toml", "Cargo.lock", "build.rs")) {
        $src = Join-Path $engSrc $f
        if (Test-Path $src) {
            Copy-Item $src (Join-Path $engDst $f) -Force
        }
    }

    # ── 3. Top-level docs ─────────────────────────────────────────────────────
    $setupDoc = Join-Path $repo "ORACLE_SETUP.md"
    if (Test-Path $setupDoc) {
        Copy-Item $setupDoc (Join-Path $stage "ORACLE_SETUP.md") -Force
    }

    # ── 4. Safety check: abort if any data files leaked in ───────────────────
    $forbidden = Get-ChildItem -Path $stage -Recurse -File |
        Where-Object {
            ($_.Extension -in @(".db",".parquet",".pt",".npy") -and $_.FullName -notmatch "net_weights") -or
            $_.Name -in @("api_token","api_token.txt")
        }
    if ($forbidden) {
        $forbidden | ForEach-Object { Write-Error "LEAK: $($_.FullName)" }
        throw "Forbidden data files found in bundle staging area. Aborting."
    }

    # ── 5. Build manifest ─────────────────────────────────────────────────────
    $engineSourceHash = (Get-ChildItem -Path (Join-Path $repo "engine\src") -Recurse -File |
        Sort-Object FullName |
        ForEach-Object { (Get-FileHash $_.FullName -Algorithm SHA256).Hash }) -join "" |
        ForEach-Object {
            $sha = [System.Security.Cryptography.SHA256]::Create()
            try { [System.BitConverter]::ToString($sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($_))).Replace("-","").ToLowerInvariant() }
            finally { $sha.Dispose() }
        }

    $manifest = [ordered]@{
        repository_commit = (git rev-parse HEAD 2>$null)
        engine_source_hash = $engineSourceHash
        worker_protocol_version = "titanium-oracle-game-factory/1"
        game_record_schema_version = "titanium-oracle-game/1"
        required_weight_schema = "halfpw-sparse-route5-catheat-ws20-cat-v2"
        build_timestamp = (Get-Date).ToUniversalTime().ToString("o")
        supported_architecture = "linux-x86_64"
        bundle_type = "oracle-game-generator-slim"
        excluded = @("engine/.git","engine/worktrees","engine/target*",
                     "training/titanium_training","training/tools","training/data",
                     "weight-files (pushed separately as a generation)")
        expected_runtime_configuration = [ordered]@{
            workers = 13
            move_time = 2.0
            bind = "127.0.0.1:8765"
            install_dir = "/opt/titanium-game-factory"
            data_dir = "/var/lib/titanium-game-factory"
        }
    }
    $manifest | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 (Join-Path $stage "BUILD_MANIFEST.json")

    # ── 6. Checksums ──────────────────────────────────────────────────────────
    $checksumLines = Get-ChildItem -Path $stage -Recurse -File |
        Where-Object { $_.Name -ne "checksums.sha256" } |
        Sort-Object FullName |
        ForEach-Object {
            $rel = $_.FullName.Substring($stage.Length + 1).Replace("\", "/")
            "$((Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant())  $rel"
        }
    $checksumLines | Set-Content -Encoding ASCII (Join-Path $stage "checksums.sha256")

    # ── 7. Report pre-archive stats ──────────────────────────────────────────
    $allFiles = Get-ChildItem -Path $stage -Recurse -File
    $totalMB = [math]::Round(($allFiles | Measure-Object Length -Sum).Sum / 1MB, 2)
    Write-Host "Staging: $($allFiles.Count) files, ${totalMB} MB uncompressed"

    # ── 8. Pack ───────────────────────────────────────────────────────────────
    New-Item -ItemType Directory -Force (Join-Path $repo $OutputDir) | Out-Null
    $archive = Join-Path $repo "$OutputDir\titanium-oracle-worker-$gitSha.tar.zst"
    if (Test-Path $archive) { Remove-Item $archive -Force }
    tar --zstd -cf $archive -C $stageRoot oracle-worker
    $sha = (Get-FileHash $archive -Algorithm SHA256).Hash.ToLowerInvariant()
    "$sha  $(Split-Path $archive -Leaf)" | Set-Content -Encoding ASCII "$archive.sha256"

    $sizeMB = [math]::Round((Get-Item $archive).Length / 1MB, 2)
    Write-Host ""
    Write-Host "Archive : $archive"
    Write-Host "Size    : ${sizeMB} MB"
    Write-Host "SHA256  : $sha"
} finally {
    Pop-Location
}
