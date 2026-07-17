param(
    [Parameter(Mandatory=$true)][string]$Token,
    [string]$Url = "http://127.0.0.1:8765",
    [int]$Limit = 25
)
$env:PYTHONPATH = "$PWD\training"
python training\oracle_laptop_client.py --url $Url --token $Token --limit $Limit

