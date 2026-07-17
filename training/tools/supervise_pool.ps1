# Unattended training supervisor — logs snapshot every 5 min, restarts pool if dead.
param(
    [int]$IntervalSec = 300,
    [string]$Repo = "c:\gitProjects\Quoridor best AI"
)
$ErrorActionPreference = "SilentlyContinue"
$log = Join-Path $Repo "training\data\overnight_logs\supervisor.log"
$env:PYTHONPATH = Join-Path $Repo "training"
$env:RUSTFLAGS = "-C target-cpu=native"

function Write-SupervisorLog([string]$Msg) {
    $line = "$(Get-Date -Format 'yyyy-MM-ddTHH:mm:ss')  $Msg"
    Add-Content -Path $log -Value $line -Encoding UTF8
}

Write-SupervisorLog "=== supervisor started interval=${IntervalSec}s ==="

while ($true) {
    $status = & python (Join-Path $Repo "training\tools\pool_status.py") 2>&1 | Out-String
    Write-SupervisorLog $status.Trim()

    $lock = Join-Path $Repo "training\data\overnight_logs\continuous_pool.lock.json"
    $poolUp = $false
    if (Test-Path $lock) {
        $pid = (Get-Content $lock | ConvertFrom-Json).pid
        if ($pid -and (Get-Process -Id $pid -ErrorAction SilentlyContinue)) { $poolUp = $true }
    }
    if (-not $poolUp) {
        Write-SupervisorLog "POOL DOWN — launching start_overnight_pool.bat"
        Start-Process -FilePath "cmd.exe" -ArgumentList "/c", (Join-Path $Repo "start_overnight_pool.bat") `
            -WorkingDirectory $Repo -WindowStyle Hidden
    }
    Start-Sleep -Seconds $IntervalSec
}
