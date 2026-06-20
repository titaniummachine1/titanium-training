# Supervised overnight pool + 5-min health checks in ONE console.
# Close this window or Ctrl+C -> kills pool, supervisor, node workers, all titanium.exe.
#
#   training/run_supervised_session.cmd   (opens this window)
#   powershell -ExecutionPolicy Bypass -File training/run_supervised_session.ps1

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root

$script:SupervisorProc = $null
$script:PollRunspace = $null
$script:PollHandle = $null
$script:CleaningUp = $false

function Stop-SupervisedSession {
    param([switch]$Quiet)
    if ($script:CleaningUp) { return }
    $script:CleaningUp = $true

    if (-not $Quiet) {
        Write-Host ""
        Write-Host "=== STOPPING SESSION ===" -ForegroundColor Yellow
    }

    if ($script:PollRunspace) {
        try { $script:PollRunspace.Stop() } catch {}
        try { $script:PollRunspace.Dispose() } catch {}
        $script:PollRunspace = $null
        $script:PollHandle = $null
    }

    if ($script:SupervisorProc -and -not $script:SupervisorProc.HasExited) {
        try { $script:SupervisorProc.Kill() } catch {}
        try { $script:SupervisorProc.WaitForExit(3000) } catch {}
    }
    $script:SupervisorProc = $null

    Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object {
            $_.CommandLine -and (
                $_.CommandLine -match 'run_swiss_overnight|supervise\.py|overnight_batch|remote_game_worker|run_nnue_cycle|coordinator\.py|ka_teacher_worker'
            )
        } |
        ForEach-Object {
            Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
        }

    cmd /c "taskkill /F /IM titanium.exe /T >nul 2>nul"
    Remove-Item -Force "$Root/training/data/eval_batch.lock" -ErrorAction SilentlyContinue

    if (-not $Quiet) {
        Write-Host "=== All workers stopped ===" -ForegroundColor Green
    }
}

# Trap Ctrl+C and normal PowerShell exit.
Register-EngineEvent -SourceIdentifier PowerShell.Exiting -Action { Stop-SupervisedSession -Quiet } | Out-Null
trap {
    Stop-SupervisedSession
    break
}

# Trap console window close (X button) on Windows.
if ($IsWindows -or $env:OS -match "Windows") {
    Add-Type @"
using System;
using System.Runtime.InteropServices;
public class ConsoleCloseHook {
    public delegate bool Handler(int sig);
    [DllImport("Kernel32")]
    public static extern bool SetConsoleCtrlHandler(Handler h, bool add);
}
"@ -ErrorAction SilentlyContinue | Out-Null
    if ([ConsoleCloseHook]) {
        $script:ConsoleHandler = [ConsoleCloseHook+Handler]{
            param([int]$sig)
            Stop-SupervisedSession -Quiet
            return $false
        }
        [void][ConsoleCloseHook]::SetConsoleCtrlHandler($script:ConsoleHandler, $true)
    }
}

$SessionLog = "$Root/training/data/session_build.log"
New-Item -ItemType Directory -Force -Path "$Root/training/data" | Out-Null
"=== session $(Get-Date -Format o) ===" | Out-File -FilePath $SessionLog -Encoding utf8

Write-Host ""
Write-Host "  QUORIDOR SUPERVISED TRAINING" -ForegroundColor Cyan
Write-Host "  Build log: training/data/session_build.log" -ForegroundColor DarkGray
Write-Host "  Ctrl+C or close window -> stops pool + supervisor + titanium" -ForegroundColor DarkGray
Write-Host ""

# Initial orphan cleanup (must not abort on empty taskkill).
$ErrorActionPreference = "SilentlyContinue"
cmd /c "taskkill /F /IM titanium.exe /T >nul 2>nul"
$ErrorActionPreference = "Stop"
Stop-SupervisedSession -Quiet

Write-Host "[1/6] Native rebuild..." -ForegroundColor Gray -NoNewline
$env:RUSTFLAGS = "-C target-cpu=native"
Push-Location "$Root/engine"
$ErrorActionPreference = "Continue"
$cargoOut = & cargo build --release -p titanium 2>&1
$cargoRc = $LASTEXITCODE
$cargoOut | Out-File -Append -FilePath $SessionLog -Encoding utf8
$ErrorActionPreference = "Stop"
Pop-Location
if ($cargoRc -ne 0) {
    Write-Host " FAILED (see session_build.log)" -ForegroundColor Red
    throw "cargo build failed (exit $cargoRc)"
}
Write-Host " OK" -ForegroundColor Green

Write-Host "[2/6] Pool preflight..." -ForegroundColor Gray -NoNewline
& python "$Root/training/pool_preflight.py" 2>&1 | Out-File -Append -FilePath $SessionLog -Encoding utf8
if ($LASTEXITCODE -ne 0) {
    Write-Host " FAILED (see session_build.log)" -ForegroundColor Red
    throw "pool_preflight failed - fix errors before starting pool"
}
Write-Host " OK" -ForegroundColor Green

Write-Host "[3/6] Engine stamp + parity..." -ForegroundColor Gray -NoNewline
& python "$Root/training/titanium_training/validation/engine_identity.py" --write 2>&1 | Out-File -Append -FilePath $SessionLog -Encoding utf8
if ($LASTEXITCODE -ne 0) { Write-Host " FAILED" -ForegroundColor Red; throw "engine_identity failed" }
& python "$Root/training/titanium_training/validation/parity_check.py" 2>&1 | Out-File -Append -FilePath $SessionLog -Encoding utf8
if ($LASTEXITCODE -ne 0) { Write-Host " FAILED" -ForegroundColor Red; throw "parity_check failed" }
Write-Host " 6/6 OK" -ForegroundColor Green

Write-Host "[4/5] Catch-up micro-trains..." -ForegroundColor Gray -NoNewline
& python "$Root/training/run_nnue_cycle.py" --catch-up 2>&1 | Out-File -Append -FilePath $SessionLog -Encoding utf8
Write-Host " done" -ForegroundColor Green

Write-Host "[5/5] Starting pool UI..." -ForegroundColor Gray

# Supervisor: 15s log watch + 5min health; alerts -> training/data/supervisor_alert.json (pool UI flash)
$supervisorPsi = New-Object System.Diagnostics.ProcessStartInfo
$supervisorPsi.FileName = "python"
$supervisorPsi.Arguments = "-u `"$Root/training/tools/operations/supervise.py`" --start-pool --interval 300 --watch-interval 15 --parity-every 3 --grace-sec 120 --remediate"
$supervisorPsi.WorkingDirectory = $Root
$supervisorPsi.UseShellExecute = $false
$supervisorPsi.CreateNoWindow = $true
$supervisorPsi.EnvironmentVariables["POOL_UI"] = "1"
$script:SupervisorProc = [System.Diagnostics.Process]::Start($supervisorPsi)
"Supervisor pid $($script:SupervisorProc.Id) watch=15s check=300s alerts=supervisor_alert.json" | Out-File -Append -FilePath $SessionLog -Encoding utf8

# Live pool UI owns the console - startup logs go to training/data/pool_startup.log
Clear-Host
$env:POOL_UI = "1"

try {
    & python -u "$Root/training/run_swiss_overnight.py"
}
finally {
    Stop-SupervisedSession
}
