# Database-first runtime launcher.
# Starts only: Oracle importer, local game pool, training coordinator.
param(
    [string]$OracleHost = "92.5.77.92",
    [string]$KeyPath = "$env:USERPROFILE\.ssh\oracle_titanium.key"
)

$ErrorActionPreference = "Stop"
$Repo = "c:\gitProjects\Quoridor best AI"
$TokenFile = "$env:LOCALAPPDATA\titanium-oracle-api-token"

Set-Location $Repo

function Get-Token {
    if (Test-Path $TokenFile) {
        return (Get-Content $TokenFile -Raw).Trim()
    }
    $token = ssh -i $KeyPath -o ConnectTimeout=15 -o BatchMode=yes `
        "ubuntu@$OracleHost" "sudo cat /var/lib/titanium-game-factory/api_token"
    if (-not $token) { throw "Could not read oracle API token" }
    Set-Content -Path $TokenFile -Value $token.Trim() -NoNewline -Encoding ascii
    return $token.Trim()
}

function Test-Tunnel {
    param([string]$Token)
    try {
        # /health is the ONLY fast probe endpoint; /status hangs and must never
        # be used for liveness checks (it stalls the 5s timeout every launch).
        $null = Invoke-RestMethod -Uri "http://127.0.0.1:8765/health" `
            -Headers @{ Authorization = "Bearer $Token" } -TimeoutSec 5
        return $true
    } catch {
        return $false
    }
}

function Start-TunnelIfNeeded {
    param([string]$Token)
    if (Test-Tunnel -Token $Token) {
        Write-Host "Oracle tunnel already up"
        return
    }
    Write-Host "Starting SSH tunnel to $OracleHost..."
    Start-Process -FilePath "ssh" -ArgumentList @(
        "-i", $KeyPath, "-N", "-L", "8765:127.0.0.1:8765",
        "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=6",
        "-o", "ConnectTimeout=15", "-o", "BatchMode=yes",
        "ubuntu@$OracleHost"
    ) -WindowStyle Hidden
    Start-Sleep -Seconds 6
    if (-not (Test-Tunnel -Token $Token)) {
        throw "Oracle tunnel started but API is not reachable"
    }
}

$token = Get-Token
Start-TunnelIfNeeded -Token $token

powershell -NoProfile -ExecutionPolicy Bypass -File "$Repo\training\tools\start_oracle_importer_detached.ps1"
powershell -NoProfile -ExecutionPolicy Bypass -File "$Repo\training\tools\start_local_game_pool_detached.ps1"
powershell -NoProfile -ExecutionPolicy Bypass -File "$Repo\training\tools\start_training_coordinator_detached.ps1"

Write-Host "Database-first runtime started. Logs: training\data\overnight_logs"
