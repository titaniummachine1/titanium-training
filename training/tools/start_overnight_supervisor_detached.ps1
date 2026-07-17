param(
    [int]$RebuildPid = 8068,
    [int]$WatcherPid = 2424,
    [int]$PoolPid = 4812
)
$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$LogDir = Join-Path $Repo "training\data\overnight_logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$OutLog = Join-Path $LogDir "overnight_supervisor.log"
$ErrLog = Join-Path $LogDir "overnight_supervisor_err.log"
$PidFile = Join-Path $LogDir "overnight_supervisor.pid"

$env:PYTHONPATH = Join-Path $Repo "training"
$env:PYTHONUNBUFFERED = "1"

$py = (Get-Command python).Source
$script = Join-Path $Repo "training\tools\overnight_supervisor.py"
$argList = "-u `"$script`" --rebuild-pid $RebuildPid --watcher-pid $WatcherPid --pool-pid $PoolPid --poll-sec 60"

$p = Start-Process -FilePath $py `
    -ArgumentList $argList `
    -WorkingDirectory $Repo `
    -RedirectStandardOutput $OutLog `
    -RedirectStandardError $ErrLog `
    -PassThru `
    -WindowStyle Hidden

$p.Id | Set-Content -Encoding ascii $PidFile
