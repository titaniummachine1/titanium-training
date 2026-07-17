$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$LogDir = Join-Path $Repo "training\data\overnight_logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$OutLog = Join-Path $LogDir "rebuild_heartbeat.log"
$ErrLog = Join-Path $LogDir "rebuild_heartbeat_err.log"
$PidFile = Join-Path $LogDir "rebuild_heartbeat.pid"

$env:PYTHONPATH = Join-Path $Repo "training"
$env:PYTHONUNBUFFERED = "1"

$py = (Get-Command python).Source
$script = Join-Path $Repo "training\tools\rebuild_progress_heartbeat.py"

$p = Start-Process -FilePath $py `
    -ArgumentList "-u `"$script`" --interval-sec 60" `
    -WorkingDirectory $Repo `
    -RedirectStandardOutput $OutLog `
    -RedirectStandardError $ErrLog `
    -PassThru `
    -WindowStyle Hidden

$p.Id | Set-Content -Encoding ascii $PidFile
Write-Host "Detached rebuild heartbeat pid=$($p.Id)"
Write-Host "jsonl: $(Join-Path $LogDir 'rebuild_progress.jsonl')"
