param(
    [Parameter(Mandatory = $true)][string]$OracleHost,
    [string]$KeyPath = "$env:USERPROFILE\.ssh\oracle_titanium.key",
    [Parameter(Mandatory = $true)][string]$Token,
    [string]$User = "ubuntu",
    [string]$Url = "http://127.0.0.1:8765",
    [string]$Current = "training\runs\v16\net_weights_best.bin",
    [string]$Prior = "training\runs\v16\net_weights_previous.bin",
    [int]$Epoch = 0,
    [double]$MoveTimeSec = 5.0,
    [int]$NodeBudget = 200000
)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $MyInvocation.MyCommand.Path
python (Join-Path $repo "training\oracle_weight_sync.py") `
    --oracle-host $OracleHost `
    --key-path $KeyPath `
    --user $User `
    --token $Token `
    --url $Url `
    --current (Join-Path $repo $Current) `
    --previous (Join-Path $repo $Prior) `
    --epoch $Epoch `
    --move-time $MoveTimeSec `
    --node-budget $NodeBudget
