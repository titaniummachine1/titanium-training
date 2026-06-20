# Live scoreboard viewer — attach without restarting the pool.
# Double-click training/pool_watch.cmd or run while pool is up elsewhere.

$ErrorActionPreference = "SilentlyContinue"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$url = if ($env:COORDINATOR_URL) { $env:COORDINATOR_URL.TrimEnd('/') } else { "http://127.0.0.1:8765" }

while ($true) {
    Clear-Host
    Write-Host ""
    Write-Host "  QUORIDOR POOL WATCH  ($url)" -ForegroundColor Cyan
    Write-Host "  Close this window anytime — pool keeps running in the training window." -ForegroundColor DarkGray
    Write-Host ""

    try {
        $sb = Invoke-RestMethod -Uri "$url/api/scoreboard?compact=1" -TimeoutSec 5
        if ($sb.text) { Write-Host $sb.text }
        $st = Invoke-RestMethod -Uri "$url/api/pool-status" -TimeoutSec 5
        $ka = $st.slot_counts.ka
        $fr = $st.slot_counts.frozen
        Write-Host ""
        Write-Host "  Active slots: $($st.slots_in_flight)/7" -ForegroundColor Yellow
        Write-Host ("  Ka short={0} medium={1} long={2}  |  JS={3}  |  frozen 5s={4} 10s={5}" -f `
            $ka.short, $ka.medium, $ka.long, $st.slot_counts.js, $fr.'5s', $fr.'10s')
    }
    catch {
        Write-Host "  Coordinator not reachable at $url" -ForegroundColor Red
        Write-Host "  Start training: training\run_supervised_session.cmd" -ForegroundColor Yellow
    }

    Write-Host ""
    Write-Host "  refresh in 5s..." -ForegroundColor DarkGray
    Start-Sleep -Seconds 5
}
