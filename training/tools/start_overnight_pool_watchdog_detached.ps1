$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$LogDir = Join-Path $Repo "training\data\overnight_logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$OutLog = Join-Path $LogDir "overnight_pool_watchdog.log"
$ErrLog = Join-Path $LogDir "overnight_pool_watchdog_err.log"
$PidFile = Join-Path $LogDir "overnight_pool_watchdog.pid"

$env:PYTHONPATH = Join-Path $Repo "training"
$env:PYTHONUNBUFFERED = "1"

$py = (Get-Command python).Source
$script = Join-Path $Repo "training\tools\overnight_pool_watchdog.py"

$p = Start-Process -FilePath $py `
    -ArgumentList "-u `"$script`" --poll-sec 120 --pool-stall-sec 600" `
    -WorkingDirectory $Repo `
    -RedirectStandardOutput $OutLog `
    -RedirectStandardError $ErrLog `
    -PassThru `
    -WindowStyle Hidden

$p.Id | Set-Content -Encoding ascii $PidFile
Write-Host "Detached pool watchdog pid=$($p.Id)"
