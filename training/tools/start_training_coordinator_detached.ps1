$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$LogDir = Join-Path $Repo "training\data\overnight_logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$OutLog = Join-Path $LogDir "training_coordinator_stdout.log"
$ErrLog = Join-Path $LogDir "training_coordinator_stderr.log"
$PidFile = Join-Path $LogDir "training_coordinator.pid"

if (Test-Path $PidFile) {
    $raw = (Get-Content $PidFile -Raw).Trim()
    $existingPid = 0
    [void][int]::TryParse($raw, [ref]$existingPid)
    $existingProc = Get-CimInstance Win32_Process -Filter "ProcessId=$existingPid" -EA SilentlyContinue
    if ($existingProc -and $existingProc.CommandLine -like "*training_coordinator.py*") {
        Write-Host "training_coordinator already running pid=$existingPid - skipping launch"
        exit 0
    }
}

$env:TRAINING_PREP_ONLY = "1"
$env:PYTHONPATH = Join-Path $Repo "training"
$env:PYTHONUNBUFFERED = "1"
$env:RUSTFLAGS = "-C target-cpu=native"
$env:NNUE_DEPLOY_EVERY = "0"
$env:ORACLE_PUSH_EACH_EPOCH = "0"
$env:TITANIUM_GENERATION_ENGINE = "titanium-v17"
$env:STREAM_PRIOR_EPOCH_FRACTION = "0.30"
$env:STREAM_RETIRED_REPLAY_FRACTION = "0.05"
# 2026-07-10: fresh h=32 lineage restart (ACE v13 weights transplanted + kept
# cat_heat, see runs/v16 epoch_0000 provenance). The normal queue sampler
# prioritizes low-training_visits positions and would starve openings (they
# recur in nearly every game, so their visit counts are already huge from the
# old h=96 lineage) -- full-active mode draws every epoch from the whole
# shuffled active pool instead, so nothing is starved by stale visit counts.
# Revert to unset (queue-sampler default) once this lineage is past its
# initial catch-up pass over the backlog.
# Explicitly keep bootstrap bounded to each claimed batch. A full-active pass
# over the multi-million-row corpus is a later deliberate consolidation step,
# not an implicit first-cycle canary.
$env:STREAM_FULL_ACTIVE_EPOCH = "0"
$env:STREAM_BOOTSTRAP_MIN_PENDING = "100000"
$env:STREAM_TRIGGER_THRESHOLD = "100000"
# 2026-07-10: featurization-rejection guard (streaming_db_loader.py's
# FeaturizationRejectionRateExceeded), added alongside the storage_kind bug
# fix that unlocked 1.4M previously-dead teacher: positions. Per-category
# thresholds, NOT one blended number -- a single generous global threshold
# would hide a brand-new corruption bug as long as it stayed under it, the
# same masking that let the storage_kind bug go unnoticed. Only overriding
# the known, already-characterized missing_from_db baseline here (observed
# ~28-38% live, residue from the previously diagnosed-unrecoverable 4.29M
# orphan incident); corruption/packed_validation/semantic stay at their
# near-zero-tolerance code defaults, so anything resembling the storage_kind
# bug still aborts immediately. Lower this back once the position_usage
# orphan residue is actually cleaned up.
$env:STREAM_MISSING_FROM_DB_BASELINE_FRACTION = "0.30"
# Widened-net (net2net h48) bootstrapping: the strength gate compares each
# candidate against its own immediate parent in the chain, which right now
# IS epoch_9 (the widened net) -- a brand-new, not-yet-trained architecture
# change is not expected to already match a fully-matured net's strength on
# cycle one. 0.45 (normal steady-state bar) would reject ordinary noisy
# progress and force every cycle to retry from scratch, so learning could
# never compound. 0.20 turned out too loose -- a candidate that's genuinely
# worse everywhere except the handful of opening lines opening_sanity checks
# could still clear it and get promoted. 0.35 let candidates through that only
# barely edged out their parent (e.g. 0.38) -- raised to 0.40 (2026-07-07) so a
# narrow pass forces another training cycle on more accumulated data instead of
# promoting a marginal, noise-level win; raise back to 0.45 once the widened
# net stabilizes.
$env:STREAM_PRIOR_EPOCH_MIN_SCORE = "0.45"
$env:ORACLE_HOST = "92.5.77.92"
$env:STREAM_PRIOR_EPOCH_MIN_SCORE = "0.45"
$env:STREAM_REPAIR_MODE = "1"
$env:STREAM_TRAIN_LR = "0.0002"
$env:STREAM_TRAIN_FAILED_BACKOFF_SEC = "90"
$env:ORACLE_USER = "ubuntu"
$env:ORACLE_KEY_PATH = "$env:USERPROFILE\.ssh\oracle_titanium.key"
$env:ORACLE_URL = "http://127.0.0.1:8765"
$env:ORACLE_TOKEN_FILE = "$env:LOCALAPPDATA\titanium-oracle-api-token"
$env:ORACLE_MOVE_TIME_SEC = "5.0"
$env:ORACLE_NODE_BUDGET = "200000"

$py = (Get-Command py).Source
$script = Join-Path $Repo "training\training_coordinator.py"

$p = Start-Process -FilePath $py `
    -ArgumentList "-3.12 -u `"$script`" --poll-sec 30 --epoch-size 100000 --batch 512 --featurize-chunk 4096" `
    -WorkingDirectory $Repo `
    -RedirectStandardOutput $OutLog `
    -RedirectStandardError $ErrLog `
    -PassThru `
    -WindowStyle Hidden

$p.Id | Set-Content -Encoding ascii $PidFile
Write-Host "Detached training_coordinator pid=$($p.Id)"
