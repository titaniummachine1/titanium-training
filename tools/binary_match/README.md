# Remote Oracle shard launcher

## Reusable Oracle compute controller

`tools/oracle_compute.ps1` is the supported lifecycle controller for the
deployed Oracle machine. It manages the existing `/opt/titanium-game-factory`
installation and never uploads raw engine source or starts local workers.
Local four-worker runs remain a separate responsibility.

The controller defaults to `ubuntu@92.5.77.92`, requires an SSH key, and uses
batch SSH with `accept-new`, a 15-second connect timeout, and server-alive
probes. It supports `deploy`, `status`, `match-start`, `match-status`,
`match-pull`, `match-stop`, `factory-pause`, and `factory-resume`.

Deploy reuses the existing pipeline exactly:

```powershell
.\tools\oracle_compute.ps1 -Mode deploy -SshKeyPath C:\keys\oracle.key
```

That runs `training/tools/build_oracle_bundle.ps1`, then
`deploy_oracle_worker.ps1`; the controller verifies that
`titanium-game-factory` is active and prints the SHA-256 of the native
`/opt/titanium-game-factory/engine/target/release/titanium`.

Start one Oracle coordinator for remote slots 4..16:

```powershell
.\tools\oracle_compute.ps1 -Mode match-start `
  -RunId oracle_20260711_220000_demo `
  -SshKeyPath C:\keys\oracle.key `
  -EngineA titanium-v17-lmp-ace -EngineB titanium-v17 `
  -Games 200 -ClockSec 60
```

`match-start` pauses the factory, verifies the deployed binary, uploads only
the match harness, opening book, and its small Python import closure to
`/var/lib/titanium-match-jobs/<RunId>`, and launches exactly one coordinator
with `--workers 13 --shard-count 17 --shard-offset 4 --shard-span 13`.
It records `coordinator.pid`, `command.json`, `coordinator.log`,
`status.json`, and `binary.sha256`. A failed start automatically resumes the
factory. Duplicate active (and previously recorded) run IDs are rejected.

```powershell
.\tools\oracle_compute.ps1 -Mode match-status -RunId oracle_20260711_220000_demo -SshKeyPath C:\keys\oracle.key
.\tools\oracle_compute.ps1 -Mode match-pull -RunId oracle_20260711_220000_demo -SshKeyPath C:\keys\oracle.key
.\tools\oracle_compute.ps1 -Mode match-stop -RunId oracle_20260711_220000_demo -SshKeyPath C:\keys\oracle.key
```

Pull refuses to resume or interfere with a live coordinator. Stop creates the
stop file, waits a bounded interval, and kills only the recorded coordinator
PID if it remains alive; both stop and completed pull resume the factory by
default. Use `-NoFactoryResume` only when deliberately keeping the factory
paused. `-DryRun` prints SSH/SCP/deploy actions without connecting.

The same deployed game-factory protocol is the extension point for future
remote training or deep-label jobs: add a narrowly scoped controller mode that
submits a typed job manifest to the existing factory/job directory and tracks
its recorded PID/status. Do not add arbitrary shell execution or source
archives to this controller.

There are two launchers:

* `launch_oracle_shards.ps1` is the legacy source-packaging launcher. Its
  existing behavior is unchanged: it uploads an engine source archive and
  builds remotely.
* `launch_oracle_deployed.ps1` is the deployed-binary launcher described
  below. Use it when `/opt/titanium-game-factory` is already installed.

## Deployed-binary launch

`launch_oracle_deployed.ps1` never archives, uploads, or builds engine source.
It verifies the deployed native executable at
`/opt/titanium-game-factory/engine/target/release/titanium`, then uploads only
`tools/binary_match/parallel_engine_match.py` and
`training/data/opening_book/non_titanium_10ply.json`. The remote script imports
its Python support files from `/opt/titanium-game-factory/training` and accepts
labels such as `titanium-v17-lmp-ace`; labels select the deployed binary's
engine mode, not a second executable.

From the repository root in PowerShell:

```powershell
.\tools\binary_match\launch_oracle_deployed.ps1 `
  -SshKeyPath C:\keys\oracle.key `
  -EngineA titanium-v17-lmp-ace -EngineB titanium-v17 `
  -Games 200 -ClockSec 60 -OpenPlies 4 -Seed 1337
```

The default host is `ubuntu@92.5.77.92`. Override `-OracleHost`,
`-OracleUser`, `-DeployedRoot`, `-RemoteRoot`, `-RunId`, or `-LocalRunDir` as
needed. The launcher starts shard offsets `4..16` with one worker each and
writes the manifest under `tools/binary_match/runs/<run-id>/`.

Each upload uses a unique `.uploading.<guid>` remote name, then checks remote
SHA-256 and byte count before an atomic `mv`. Failed uploads retry at most
`-UploadAttempts` times (default 3); a failed verification is never renamed
into place. No raw engine archive is copied.

### Prerequisite and deployment integration

The deployed root must already contain the native binary and the Python
closure: `training/engine_session.py`,
`training/self_play_overnight.py`, and
`training/titanium_training/paths.py`. If it is not installed, run the
existing deployment script from the repository root; this launcher does not
modify or invoke it automatically:

```powershell
.\deploy_oracle_worker.ps1 `
  -OracleHost 92.5.77.92 `
  -KeyPath C:\keys\oracle.key `
  -User ubuntu
```

That command deploys the latest `dist\titanium-oracle-worker-*.tar.zst` and
restarts `titanium-game-factory`. Build that bundle first with the command
required by the existing deployment workflow if no matching archive exists:
`training\tools\build_oracle_bundle.ps1`.

### Status, pull, and stop

```powershell
.\tools\binary_match\launch_oracle_deployed.ps1 `
  -Mode status -RunId oracle_20260711_200000_ab12cd34 `
  -SshKeyPath C:\keys\oracle.key

.\tools\binary_match\launch_oracle_deployed.ps1 `
  -Mode pull -RunId oracle_20260711_200000_ab12cd34 `
  -SshKeyPath C:\keys\oracle.key
```

`pull` stores shard status, JSONL results, and logs in
`tools/binary_match/runs/<run-id>/`. To stop without killing in-flight games:

```powershell
ssh -i C:\keys\oracle.key ubuntu@92.5.77.92 `
  "touch /tmp/quoridor-oracle-deployed/<run-id>/STOP"
```

### Dry-run and syntax validation

`-DryRun` prints every SSH/SCP command and never invokes either executable.
It also avoids creating local run output. Validate both PowerShell launchers
without connecting:

```powershell
$errors = $null
foreach ($path in @(
  ".\tools\binary_match\launch_oracle_shards.ps1",
  ".\tools\binary_match\launch_oracle_deployed.ps1"
)) {
  [System.Management.Automation.Language.Parser]::ParseFile(
    (Resolve-Path $path), [ref]$null, [ref]$errors
  )
}
if ($errors.Count) { $errors | Format-List; exit 1 }
python -m py_compile .\tools\binary_match\parallel_engine_match.py
```

The dry-run still requires an existing `-SshKeyPath` because the parameter
validation is intentionally the same as a real launch.

## Legacy source-packaging launcher

`launch_oracle_shards.ps1` packages the exact local Titanium engine source
checkout needed to build the engine (Cargo manifest/lockfile, build helpers,
and `src/`; never `target/`), plus the match harness, its small Python
dependency closure, and the audited opening-book JSON. Oracle builds a native
Linux `titanium` binary with `RUSTFLAGS='-C target-cpu=native'`, verifies that
the artifact is executable, and records source and artifact SHA-256 metadata.
It then starts exactly 13 single-worker processes for shard offsets `4..16`,
with `--shard-count 17` and `--shard-span 1`. No local worker is started.

The package is intentionally minimal: `parallel_engine_match.py` imports
`engine_session.py`, `self_play_overnight.py`, `db_import.py`, and the
`titanium_training` path/state/config modules. The runner also needs
`training/data/opening_book/non_titanium_10ply.json`; no training databases,
weights, unrelated datasets, or service files are copied.

## Launch

From the repository root in PowerShell:

```powershell
.\tools\binary_match\launch_oracle_shards.ps1 `
  -SshKeyPath C:\keys\oracle.key `
  -EngineA titanium-v17 -EngineB titanium-v16 `
  -Games 200 -ClockSec 60 -OpenPlies 4 -Seed 1337
```

The default host is `ubuntu@92.5.77.92`. Override `-OracleUser`,
`-OracleHost`, `-RemoteRoot`, or `-RunId` when needed. Each run writes its
manifest and pulled artifacts under
`tools/binary_match/runs/<run-id>/`. The manifest records hashes for every
packaged engine source file and Cargo manifest; the remote
`engine_artifact.sha256` records the native Linux executable hash. The local
`engine-source.tar` archive is retained in that run directory for
reproducibility; an existing archive is never overwritten.

## Status and pull

Use the same SSH key, host, and run ID. No candidate binary is needed:

```powershell
.\tools\binary_match\launch_oracle_shards.ps1 `
  -Mode status -RunId oracle_20260711_200000_ab12cd34 `
  -SshKeyPath C:\keys\oracle.key

.\tools\binary_match\launch_oracle_shards.ps1 `
  -Mode pull -RunId oracle_20260711_200000_ab12cd34 `
  -SshKeyPath C:\keys\oracle.key
```

`pull` copies each shard's `status_shard_<offset>.json`,
`results_shard_<offset>_1.jsonl`, and log into the matching local run
directory. The launcher never removes remote files; stop a run by creating
the remote stop file:

```powershell
ssh -i C:\keys\oracle.key ubuntu@92.5.77.92 `
  "touch /tmp/quoridor-oracle/<run-id>/STOP"
```

## Dry run and validation

Add `-DryRun` to print all SSH/SCP and launch commands without connecting,
copying, or creating local output. SSH uses batch mode, `accept-new`
host-key handling, a 15-second connect timeout, and server-alive probes.
Validate PowerShell syntax without deploying:

```powershell
$errors = $null
[System.Management.Automation.Language.Parser]::ParseFile(
  (Resolve-Path .\tools\binary_match\launch_oracle_shards.ps1),
  [ref]$null, [ref]$errors
)
if ($errors.Count) { $errors | Format-List; exit 1 }
```
