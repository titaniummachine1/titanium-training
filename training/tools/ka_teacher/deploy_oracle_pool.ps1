param(
    [string]$OracleHost = "92.5.77.92",
    [string]$OracleUser = "ubuntu",
    [string]$KeyPath = "$env:USERPROFILE\.ssh\oracle_titanium.key",
    [string]$RemoteRoot = "/home/ubuntu/ka-teacher"
)

$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$Remote = "${OracleUser}@${OracleHost}"
$WorkerDir = "$RemoteRoot/training/tools/ka_teacher"
$NativeDir = "$WorkerDir/native_runtime"
$AssetDir = "$RemoteRoot/reference/ka_weights_export"

& ssh -i $KeyPath -o BatchMode=yes $Remote "mkdir -p '$NativeDir' '$AssetDir'"
if ($LASTEXITCODE -ne 0) { throw "Could not create the Oracle teacher directories." }

& scp -i $KeyPath -o BatchMode=yes `
    (Join-Path $PSScriptRoot "ka_nn_batch_worker.mjs") `
    "${Remote}:$WorkerDir/"
if ($LASTEXITCODE -ne 0) { throw "Could not deploy the Ka worker." }

$NativeFiles = @(
    (Join-Path $PSScriptRoot "native_runtime\package.json"),
    (Join-Path $PSScriptRoot "native_runtime\package-lock.json"),
    (Join-Path $PSScriptRoot "native_runtime\ka_epoch15000_b64.onnx")
)
& scp -i $KeyPath -o BatchMode=yes @NativeFiles "${Remote}:$NativeDir/"
if ($LASTEXITCODE -ne 0) { throw "Could not deploy the native teacher runtime." }

$Assets = @(
    (Join-Path $Repo "reference\ka_weights_export\engine-core.js"),
    (Join-Path $Repo "reference\ka_weights_export\ka-encoder.js"),
    (Join-Path $Repo "reference\ka_weights_export\ka-weights.json")
)
& scp -i $KeyPath -o BatchMode=yes @Assets "${Remote}:$AssetDir/"
if ($LASTEXITCODE -ne 0) { throw "Could not deploy the Ka runtime assets." }

$Install = @"
set -eu
if ! command -v npm >/dev/null 2>&1; then
  sudo apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq npm
fi
cd '$NativeDir'
npm ci --omit=dev --no-audit --no-fund
test -f node_modules/onnxruntime-node/bin/napi-v6/linux/x64/onnxruntime_binding.node
"@
& ssh -i $KeyPath -o BatchMode=yes $Remote ($Install -replace "`r`n", "`n")
if ($LASTEXITCODE -ne 0) { throw "Oracle ONNX Runtime installation failed." }

$Smoke = '{"id":"deploy-smoke","positions":[{"id":"start","moves":[]}]}'
$Smoke | & ssh -i $KeyPath -o BatchMode=yes $Remote `
    nice -n 5 node "$WorkerDir/ka_nn_batch_worker.mjs" `
    --backend cpu --batch-max 64 --model-batch 64 --threads 1
if ($LASTEXITCODE -ne 0) { throw "Oracle teacher smoke test failed." }

Write-Host "Oracle Ka teacher deployed and verified at $RemoteRoot"
