param(
    [Parameter(Mandatory = $true)][string]$OracleHost,
    [string]$KeyPath = "$env:USERPROFILE\.ssh\oracle_titanium.key",
    [string]$User = "ubuntu",
    [int]$Workers = 13,
    [double]$MoveTime = 5.0,
    [int]$NodeBudget = 200000,
    [switch]$DryRun
)
$ErrorActionPreference = "Stop"
$archive = Get-ChildItem dist\titanium-oracle-worker-*.tar.zst |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
if (-not $archive) {
    throw "No dist\titanium-oracle-worker-*.tar.zst found. Run training\tools\build_oracle_bundle.ps1 first."
}

function Invoke-NativeChecked {
    param(
        [Parameter(Mandatory = $true)][string]$Command,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )
    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Command failed with exit code $LASTEXITCODE."
    }
}

$remoteScript = @'
set -euo pipefail
archive_name="$1"
workers="$2"
move_time="$3"
node_budget="$4"
install_dir=/opt/titanium-game-factory
remote_archive="/tmp/${archive_name}"
stage_dir="/tmp/oracle-worker.$$"
weights_backup="/tmp/titanium-oracle-weights.$$"
weights_restored=0

cleanup() {
    rm -rf "$stage_dir" "$remote_archive"
    if [[ "$weights_restored" -eq 1 ]]; then
        sudo rm -rf "$weights_backup"
    elif sudo test -d "$weights_backup"; then
        echo "Deployment failed; restoring the previous weights." >&2
        sudo mkdir -p "$install_dir"
        sudo rm -rf "$install_dir/weights"
        sudo cp -a "$weights_backup" "$install_dir/weights"
        echo "Weights backup retained at $weights_backup for inspection." >&2
    else
        echo "No previous weights backup was created." >&2
    fi
}
trap cleanup EXIT

rm -rf "$stage_dir" "$weights_backup"
mkdir -p "$stage_dir"
test -s "$remote_archive"
tar --zstd -xf "$remote_archive" -C "$stage_dir"
bundle_dir="$stage_dir/oracle-worker"
test -d "$bundle_dir"
test -s "$bundle_dir/BUILD_MANIFEST.json"
test -s "$bundle_dir/checksums.sha256"
test -f "$bundle_dir/training/oracle_game_factory/install.sh"
(cd "$bundle_dir" && sha256sum -c checksums.sha256)

if sudo test -d "$install_dir/weights"; then
    sudo cp -a "$install_dir/weights" "$weights_backup"
fi

sudo rm -rf "$install_dir"
sudo mkdir -p "$install_dir"
sudo cp -a "$bundle_dir/." "$install_dir/"
sudo test -s "$install_dir/BUILD_MANIFEST.json"
sudo test -f "$install_dir/training/oracle_game_factory/install.sh"
sudo env WORKERS="$workers" MOVE_TIME="$move_time" NODE_BUDGET="$node_budget" \
    bash "$install_dir/training/oracle_game_factory/install.sh"
sudo test -x "$install_dir/engine/target/release/titanium"

if sudo test -d "$weights_backup"; then
    sudo rm -rf "$install_dir/weights"
    sudo cp -a "$weights_backup" "$install_dir/weights"
fi
sudo test -d "$install_dir/weights"
sudo systemctl restart titanium-game-factory.service
sudo systemctl is-active --quiet titanium-game-factory.service
weights_restored=1
'@

function ConvertTo-SshSingleQuoted {
    param([Parameter(Mandatory = $true)][string]$Value)
    $quote = [char]39
    $escapedQuote = [string]($quote + [char]92 + $quote + $quote)
    return $quote + $Value.Replace([string]$quote, $escapedQuote) + $quote
}

$remoteArgs = @(
    (ConvertTo-SshSingleQuoted $archive.Name),
    (ConvertTo-SshSingleQuoted $Workers.ToString([Globalization.CultureInfo]::InvariantCulture)),
    (ConvertTo-SshSingleQuoted $MoveTime.ToString([Globalization.CultureInfo]::InvariantCulture)),
    (ConvertTo-SshSingleQuoted $NodeBudget.ToString([Globalization.CultureInfo]::InvariantCulture))
)
# PowerShell's native-process pipeline writes CRLF even when the here-string is
# normalized.  Feeding it directly to Bash leaves a trailing `\r` on every
# source line, which can make shell arithmetic fail after an otherwise
# successful install.  Transfer the script as base64 so Bash receives the
# intended LF-only bytes.
$remoteScriptUtf8 = [Text.Encoding]::UTF8.GetBytes(($remoteScript -replace "`r`n", "`n"))
$remoteScriptBase64 = [Convert]::ToBase64String($remoteScriptUtf8)
$remoteCommand = "printf '%s' $remoteScriptBase64 | base64 -d | bash -s -- " + ($remoteArgs -join " ")

if ($DryRun) {
    Write-Host "DryRun: archive=$($archive.FullName)"
    Write-Host "DryRun: ssh $User@$OracleHost $remoteCommand"
    Write-Host "DryRun: remote script validated for fail-closed execution."
    return
}

$remote = "${User}@${OracleHost}"
Invoke-NativeChecked -Command "scp" -Arguments @(
    "-i", $KeyPath, $archive.FullName, "${remote}:/tmp/$($archive.Name)"
)
& ssh -i $KeyPath $remote $remoteCommand
if ($LASTEXITCODE -ne 0) {
    throw "ssh deployment failed with exit code $LASTEXITCODE."
}
