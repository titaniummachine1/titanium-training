param(
    [string]$OracleHost = "92.5.77.92",
    [string]$KeyPath = "$env:USERPROFILE\.ssh\oracle_titanium.key",
    [string]$User = "ubuntu",
    [int]$Workers = 13,
    [double]$MoveTime = 5.0,
    [int]$NodeBudget = 200000
)
$ErrorActionPreference = "Stop"
$unit = "/etc/systemd/system/titanium-game-factory.service"
$exec = @(
    "/usr/bin/python3", "-m", "training.oracle_game_factory.server",
    "--host", "127.0.0.1", "--port", "8765",
    "--data-dir", "/var/lib/titanium-game-factory",
    "--engine-bin", "/opt/titanium-game-factory/engine/target/release/titanium",
    "--workers", "$Workers",
    "--move-time", "$MoveTime",
    "--node-budget", "$NodeBudget"
) -join " "
$remote = @"
set -euo pipefail
sudo sed -i 's|^ExecStart=.*|ExecStart=$exec|' $unit
sudo systemctl daemon-reload
sudo systemctl restart titanium-game-factory.service
sleep 2
systemctl is-active titanium-game-factory.service
grep '^ExecStart=' $unit
"@
ssh -i $KeyPath "${User}@${OracleHost}" $remote
Write-Host "Oracle search: workers=$Workers move-time=${MoveTime}s node-budget=$NodeBudget"
