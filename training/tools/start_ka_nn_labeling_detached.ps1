param(
    [ValidateSet("directml", "cpu")]
    [string]$Backend = "directml",
    [int]$DeviceId = 1,
    [int]$BatchSize = 4096,
    [int]$ModelBatch = 64,
    [int]$LocalCpuWorkers = 3,
    [int]$OracleWorkers = 26,
    [string]$OracleHost = "92.5.77.92",
    [string]$OracleKey = "$env:USERPROFILE\.ssh\oracle_titanium.key"
)

$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$LogDir = Join-Path $Repo "training\data\overnight_logs"
$PidFile = Join-Path $LogDir "ka_nn_labeling.pid"
$OutLog = Join-Path $LogDir "ka_nn_labeling_stdout.log"
$ErrLog = Join-Path $LogDir "ka_nn_labeling_stderr.log"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

if (Test-Path $PidFile) {
    $existingPid = 0
    [void][int]::TryParse((Get-Content $PidFile -Raw).Trim(), [ref]$existingPid)
    $existing = Get-CimInstance Win32_Process -Filter "ProcessId=$existingPid" -ErrorAction SilentlyContinue
    if ($existing -and $existing.CommandLine -like "*ka_nn_collect_labels.py*") {
        Write-Host "Ka NN labeler already running pid=$existingPid"
        exit 0
    }
}

$py = (Get-Command py).Source
$script = Join-Path $Repo "training\tools\ka_teacher\ka_nn_collect_labels.py"
$arguments = (
    "-3.12 -u `"$script`" --continuous --backend $Backend " +
    "--device-id $DeviceId --limit $BatchSize --batch-max $ModelBatch " +
    "--model-batch $ModelBatch --threads 1 --local-cpu-workers $LocalCpuWorkers " +
    "--oracle-workers $OracleWorkers --oracle-host $OracleHost " +
    "--oracle-key `"$OracleKey`" --chunk-size $ModelBatch --sleep-sec 0"
)

$process = Start-Process -FilePath $py `
    -ArgumentList $arguments `
    -WorkingDirectory $Repo `
    -RedirectStandardOutput $OutLog `
    -RedirectStandardError $ErrLog `
    -PassThru `
    -WindowStyle Hidden

$process.Id | Set-Content -Encoding ascii $PidFile
Write-Host (
    "Detached Ka NN pool pid=$($process.Id) gpu=$Backend`:$DeviceId " +
    "local_cpu=$LocalCpuWorkers oracle_cpu=$OracleWorkers batch=$BatchSize"
)
