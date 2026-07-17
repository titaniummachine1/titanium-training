# Start continuous Ka-AB teacher labeling for NNUE value targets.
$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
Set-Location $Repo

$env:KA_ACE = if ($env:KA_ACE) { $env:KA_ACE } else { "C:\Users\Terminatort8000\Downloads\ace.html" }
$LogDir = Join-Path $Repo "training\data\overnight_logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Out = Join-Path $Repo "training\data\ka_teacher_quarantine\ka_ab_labels.jsonl"
$RunnerLog = Join-Path $LogDir "ka_ab_labeling_runner.log"

Write-Host "Ka-AB labeling -> $Out"
Write-Host "Ace bundle: $env:KA_ACE"
Write-Host "Runner log: $RunnerLog"

$argsList = @(
    "training/tools/ka_teacher/ka_ab_collect_labels.py",
    "--continuous",
    "--batch-size", "48",
    "--nodes", "32768",
    "--sync-labels-db",
    "--out", $Out
)

$RunnerErr = Join-Path $LogDir "ka_ab_labeling_runner.err.log"

Start-Process -FilePath "python" -ArgumentList $argsList -WorkingDirectory $Repo `
    -RedirectStandardOutput $RunnerLog -RedirectStandardError $RunnerErr -WindowStyle Hidden

Write-Host "Started background Ka-AB labeler (PID in Task Manager: python ka_ab_collect_labels.py)"
