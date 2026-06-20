# 10-minute overnight training watchdog — logs to training/data/overnight_watch.log
$Root = "c:\gitProjects\Quoridor best AI"
$Log = Join-Path $Root "training\data\overnight_watch.log"
$Py = "python"

function Write-Watch($msg) {
    $ts = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    $line = "$ts $msg"
    Add-Content -Path $Log -Value $line -Encoding utf8
    Write-Output $line
}

Write-Watch "watchdog start interval=600s"

while ($true) {
    Start-Sleep -Seconds 600
    $ts = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    Write-Watch "=== tick ==="

    $pool = Get-CimInstance Win32_Process -Filter "name='python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -match 'run_swiss_overnight' }
    if ($pool) {
        Write-Watch "pool: RUNNING pid=$($pool.ProcessId -join ',')"
    }
    else {
        Write-Watch "pool: STOPPED"
    }

    & $Py "$Root\training\supervise.py" --once 2>&1 | ForEach-Object { Write-Watch "supervise: $_" }

    & $Py "$Root\training\verify_db_games.py" 2>&1 | ForEach-Object { Write-Watch "db: $_" }

    if (Test-Path "$Root\training\data\nnue_guard_state.json") {
        $g = Get-Content "$Root\training\data\nnue_guard_state.json" -Raw | ConvertFrom-Json
        Write-Watch ("guard: trained_id={0} runs={1} since_deploy={2}" -f $g.last_trained_game_id, $g.train_runs, $g.games_since_deploy)
    }

    if (Test-Path "$Root\training\data\STATUS.txt") {
        Get-Content "$Root\training\data\STATUS.txt" -TotalCount 25 | ForEach-Object { Write-Watch "status: $_" }
    }

    Write-Watch 'AGENT_LOOP_TICK_overnight {"prompt":"Check overnight_watch.log and supervisor.log; report progress or alert if pool stopped or health FAIL"}'
}
