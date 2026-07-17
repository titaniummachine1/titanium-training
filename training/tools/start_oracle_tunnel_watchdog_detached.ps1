$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$LogDir = Join-Path $Repo "training\data\overnight_logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$OutLog = Join-Path $LogDir "oracle_tunnel_watchdog_stdout.log"
$ErrLog = Join-Path $LogDir "oracle_tunnel_watchdog_stderr.log"
$PidFile = Join-Path $LogDir "oracle_tunnel_watchdog.pid"

if (Test-Path $PidFile) {
    $existingPid = [int](Get-Content $PidFile -Raw).Trim()
    $existingProc = Get-CimInstance Win32_Process -Filter "ProcessId=$existingPid" -EA SilentlyContinue
    if ($existingProc -and $existingProc.CommandLine -like "*oracle_tunnel_watchdog.py*") {
        Write-Host "oracle_tunnel_watchdog already running pid=$existingPid - skipping launch"
        exit 0
    }
}

$env:PYTHONPATH = Join-Path $Repo "training"
$env:PYTHONUNBUFFERED = "1"
$env:ORACLE_HOST = "92.5.77.92"
$env:ORACLE_USER = "ubuntu"
$env:ORACLE_KEY_PATH = "$env:USERPROFILE\.ssh\oracle_titanium.key"

$py = (Get-Command python).Source
$script = Join-Path $Repo "training\tools\oracle_tunnel_watchdog.py"

$p = Start-Process -FilePath $py `
    -ArgumentList "-u `"$script`" --poll-sec 30" `
    -WorkingDirectory $Repo `
    -RedirectStandardOutput $OutLog `
    -RedirectStandardError $ErrLog `
    -PassThru `
    -WindowStyle Hidden

$p.Id | Set-Content -Encoding ascii $PidFile
Write-Host "Detached oracle tunnel watchdog pid=$($p.Id)"
