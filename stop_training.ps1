# Stop all Quoridor/Titanium training processes (safe to run after a runaway pool).
$repo = Split-Path -Parent $MyInvocation.MyCommand.Path
$patterns = @('continuous_pool', 'trainer.py', 'self_play_overnight', 'build_feature_cache', 'overnight_loop')

Write-Host "Stopping training-related processes..."
Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
Where-Object { $cmd = $_.CommandLine; $patterns | Where-Object { $cmd -like "*$_*" } } |
ForEach-Object {
    Write-Host "  kill python PID $($_.ProcessId): $($_.CommandLine.Substring(0, [Math]::Min(120, $_.CommandLine.Length)))"
    Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
}

Get-Process -Name titanium -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Host "  kill titanium PID $($_.Id)"
    Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
}

Write-Host "Done. Remaining python:"
Get-Process python* -ErrorAction SilentlyContinue | Select-Object Id, CPU, @{N = 'MB'; E = { [math]::Round($_.WS / 1MB) } }
