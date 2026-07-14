$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$LogDir = Join-Path $Repo "training\data\overnight_logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$OutLog = Join-Path $LogDir "oracle_importer.log"
$ErrLog = Join-Path $LogDir "oracle_importer_err.log"
$PidFile = Join-Path $LogDir "oracle_importer.pid"
$TokenFile = Join-Path $env:LOCALAPPDATA "titanium-oracle-api-token"

if (-not (Test-Path $TokenFile)) {
    throw "Oracle token missing: $TokenFile"
}
$token = (Get-Content $TokenFile -Raw).Trim()

if (Test-Path $PidFile) {
    $existingPid = [int](Get-Content $PidFile -Raw).Trim()
    $existingProc = Get-CimInstance Win32_Process -Filter "ProcessId=$existingPid" -EA SilentlyContinue
    if ($existingProc -and $existingProc.CommandLine -like "*oracle_importer.py*") {
        Write-Host "oracle_importer already running pid=$existingPid - skipping launch"
        exit 0
    }
}

$env:TRAINING_PREP_ONLY = "1"
$env:PYTHONPATH = Join-Path $Repo "training"
$env:PYTHONUNBUFFERED = "1"

$py = (Get-Command python).Source
$script = Join-Path $Repo "training\oracle_importer.py"
$argList = @(
    "-u `"$script`"",
    "--url http://127.0.0.1:8765",
    "--token $token",
    "--poll-sec 30"
) -join " "

$p = Start-Process -FilePath $py `
    -ArgumentList $argList `
    -WorkingDirectory $Repo `
    -RedirectStandardOutput $OutLog `
    -RedirectStandardError $ErrLog `
    -PassThru `
    -WindowStyle Hidden

$p.Id | Set-Content -Encoding ascii $PidFile
Write-Host "Detached oracle_importer pid=$($p.Id)"
