param(
    [Parameter(Mandatory = $true)][string]$Host,
    [Parameter(Mandatory = $true)][string]$KeyPath,
    [string]$User = "ubuntu"
)
ssh -i $KeyPath "${User}@${Host}" "sudo systemctl start titanium-game-factory && sudo systemctl status --no-pager titanium-game-factory"

