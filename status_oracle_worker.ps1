param(
    [Parameter(Mandatory=$true)][string]$Host,
    [string]$KeyPath = "$env:USERPROFILE\.ssh\oracle_titanium.key",
    [string]$User = "ubuntu"
)
ssh -i $KeyPath "${User}@${Host}" "bash /opt/titanium-game-factory/training/oracle_game_factory/status.sh"

