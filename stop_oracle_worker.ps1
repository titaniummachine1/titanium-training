param(
    [Parameter(Mandatory=$true)][string]$Host,
    [string]$KeyPath = "$env:USERPROFILE\.ssh\oracle_titanium.key",
    [string]$User = "ubuntu"
)
ssh -i $KeyPath "${User}@${Host}" "sudo systemctl stop titanium-game-factory"

