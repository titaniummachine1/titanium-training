param(
    [Parameter(Mandatory=$true)][string]$Host,
    [string]$KeyPath = "$env:USERPROFILE\.ssh\oracle_titanium.key",
    [string]$User = "ubuntu",
    [int]$LocalPort = 8765
)
ssh -i $KeyPath -N -L ${LocalPort}:127.0.0.1:8765 "${User}@${Host}"

