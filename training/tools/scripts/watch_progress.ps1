# Live training dashboard — read-only; does not start/stop the pool.
# Use while run_supervised_session.cmd runs elsewhere, or alone to tail logs.

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$SupLog = Join-Path $Root "training/data/supervisor.log"
$TrainLog = Join-Path $Root "training/data/nnue_train.log"
$Db = Join-Path $Root "training/data/all_games.db"

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  QUORIDOR TRAINING — live progress (read-only)" -ForegroundColor Cyan
Write-Host "  supervisor.log + game count + scoreboard every 60s" -ForegroundColor Cyan
Write-Host "  Close this window anytime — pool keeps running" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

$lastSup = ""
$lastTrain = ""
$tick = 0

while ($true) {
    $tick++
    if (Test-Path $Db) {
        $n = python -c "import sqlite3; print(sqlite3.connect(r'$Db').execute('select count(*) from games').fetchone()[0])" 2>$null
        if ($n) { Write-Host "[$(Get-Date -Format 'HH:mm:ss')] games in DB: $n" -ForegroundColor Green }
    }
    if (Test-Path $SupLog) {
        $line = Get-Content $SupLog -Tail 1 -ErrorAction SilentlyContinue
        if ($line -and $line -ne $lastSup) {
            $lastSup = $line
            Write-Host "[supervisor] $line" -ForegroundColor Yellow
        }
    }
    if (Test-Path $TrainLog) {
        $tline = Get-Content $TrainLog -Tail 1 -ErrorAction SilentlyContinue
        if ($tline -and $tline -ne $lastTrain) {
            $lastTrain = $tline
            Write-Host "[train]      $tline" -ForegroundColor Gray
        }
    }
    if ($tick % 12 -eq 1) {
        Write-Host ""
        Write-Host "--- scoreboard ---" -ForegroundColor Cyan
        python "$Root/training/run_swiss_overnight.py" --scoreboard 2>$null | Select-Object -First 14
        Write-Host ""
    }
    Start-Sleep -Seconds 5
}
