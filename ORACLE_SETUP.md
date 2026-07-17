# Titanium Oracle Game Factory Setup

This guide provisions a **game generator only**. The laptop remains the only
training authority: `games.db`, active teacher dataset, feature cache, trainer,
optimizer, validation, promotion gate, deployed-weight decision, and epoch state
stay local.

Do not expose port `8765` publicly. The API is bound to `127.0.0.1` on Oracle
and reached from Windows through SSH local forwarding:

```powershell
ssh -i <key-file> -N -L 8765:127.0.0.1:8765 ubuntu@<oracle-ip>
```

## OCI Console Checklist

1. Open **Oracle Cloud Console** -> **Compute** -> **Instances** -> **Create instance**.
2. Name: `titanium-game-factory`.
3. Image: click **Edit** -> **Canonical Ubuntu** -> **Ubuntu 24.04** -> x86-64 platform image.
4. Shape: click **Change shape** -> **Virtual machine** -> **AMD** -> `VM.Standard.E6.Flex`.
5. Shape settings:
   - OCPUs: `16`
   - Memory: `32 GB`
   - No GPU
6. Boot volume: expand **Show advanced options** if needed:
   - Size: `50 GB`
   - Performance: balanced/default
   - No additional block volume initially
7. Networking:
   - Public subnet with public IPv4
   - Prefer **Reserved public IPv4** if convenient
   - Normal outbound internet access
8. Security list / NSG ingress:
   - TCP `22` only
   - Source: your current public IPv4 as `/32`
   - Do **not** expose HTTP, DB, Parquet, or worker ports
9. SSH keys:
   - Choose **Paste public keys** if you already have one, or **Generate key pair**.
   - If generating, download the private key and store it somewhere like:
     `C:\Users\<you>\.ssh\oracle_titanium.key`
   - Lock it down from PowerShell if needed:
     ```powershell
     icacls C:\Users\<you>\.ssh\oracle_titanium.key /inheritance:r
     icacls C:\Users\<you>\.ssh\oracle_titanium.key /grant:r "$env:USERNAME:R"
     ```
10. Click **Create** and wait for the instance to reach **Running**.
11. Copy the instance **Public IPv4 address** from the instance details page.

If `VM.Standard.E6.Flex` is unavailable, do not choose ARM. Check **AMD x86**
flexible alternatives in the same shape selector, such as the closest available
`VM.Standard.E5.Flex` or `VM.Standard.E4.Flex`, and tell the user exactly which
AMD x86 shapes are available in that region.

## First Connect From Windows

```powershell
ssh -i C:\Users\<you>\.ssh\oracle_titanium.key ubuntu@<oracle-ip>
```

Verify the VM:

```bash
nproc                 # expected approximately 32
free -h               # expected about 32 GB
df -h /               # expected about 50 GB boot volume
uname -m              # x86_64
```

## Build the Bundle on the Laptop

```powershell
cd "c:\gitProjects\Quoridor best AI"
.\training\tools\build_oracle_bundle.ps1
```

Expected output:

```text
dist/titanium-oracle-worker-<git-sha>.tar.zst
dist/titanium-oracle-worker-<git-sha>.tar.zst.sha256
```

## Upload and Install

```powershell
.\deploy_oracle_worker.ps1 -Host <oracle-ip> -KeyPath C:\Users\<you>\.ssh\oracle_titanium.key -User ubuntu -Workers 32 -MoveTime 2.0
```

The install script:

- creates unprivileged user `titanium`;
- installs runtime dependencies;
- builds Linux Titanium with `RUSTFLAGS="-C target-cpu=native" cargo build --release`;
- installs files under `/opt/titanium-game-factory`;
- stores mutable data under `/var/lib/titanium-game-factory`;
- installs `titanium-game-factory.service`.

## Start, Stop, Update, Status

```powershell
.\start_oracle_worker.ps1  -Host <oracle-ip> -KeyPath <key-file>
.\status_oracle_worker.ps1 -Host <oracle-ip> -KeyPath <key-file>
.\stop_oracle_worker.ps1   -Host <oracle-ip> -KeyPath <key-file>
```

On Oracle directly:

```bash
sudo systemctl start titanium-game-factory
sudo systemctl stop titanium-game-factory
sudo systemctl status --no-pager titanium-game-factory
journalctl -u titanium-game-factory -f
bash /opt/titanium-game-factory/training/oracle_game_factory/status.sh
```

Update after uploading a new bundle:

```bash
cd /opt/titanium-game-factory
sudo bash training/oracle_game_factory/update.sh
```

## Open the SSH Tunnel

Keep this PowerShell open:

```powershell
.\open_oracle_tunnel.ps1 -Host <oracle-ip> -KeyPath <key-file> -User ubuntu
```

The laptop talks to:

```text
http://127.0.0.1:8765
```

Read the API token:

```powershell
ssh -i <key-file> ubuntu@<oracle-ip> "sudo cat /var/lib/titanium-game-factory/api_token"
```

## Push Promoted Weights

Only push a **promoted** generation:

```powershell
.\push_oracle_generation.ps1 -Host <oracle-ip> -KeyPath <key-file> -Token <api-token> -Epoch <epoch>
```

This creates:

```text
generation/
  current.bin
  prior.bin
  generation.json
  checksums.sha256
```

The Oracle service stages the generation, verifies hashes, then atomically
activates it. In-flight games may finish on the old generation; every game
records the generation and side hashes actually used.

## Pull Results Manually

```powershell
.\pull_oracle_results.ps1 -Token <api-token> -Limit 25
```

The continuous pool can also run a background importer after a controlled
restart:

```powershell
python -u training\continuous_pool.py --threads 8 --time 2.0 --batch-games 1024 `
  --no-parity --saturate-grace-epochs 10 --no-initial-epoch `
  --oracle-url http://127.0.0.1:8765 --oracle-token <api-token>
```

## Smoke Mode

Use this before production:

```powershell
.\deploy_oracle_worker.ps1 -Host <oracle-ip> -KeyPath <key-file> -Workers 2 -MoveTime 0.1
.\open_oracle_tunnel.ps1 -Host <oracle-ip> -KeyPath <key-file>
.\push_oracle_generation.ps1 -Host <oracle-ip> -KeyPath <key-file> -Token <api-token> -Epoch 0
.\pull_oracle_results.ps1 -Token <api-token> -Limit 10
```

Smoke success means ten games were generated, downloaded, imported through
canonical SQLite, synchronized into the active teacher dataset, and acknowledged.

## Production Start

```powershell
.\deploy_oracle_worker.ps1 -Host <oracle-ip> -KeyPath <key-file> -Workers 32 -MoveTime 2.0
.\open_oracle_tunnel.ps1 -Host <oracle-ip> -KeyPath <key-file>
.\push_oracle_generation.ps1 -Host <oracle-ip> -KeyPath <key-file> -Token <api-token> -Epoch <promoted-epoch>
```

Then restart the local pool only during a controlled window with
`--oracle-url` and `--oracle-token`.

## Emergency Stop

Oracle:

```powershell
.\stop_oracle_worker.ps1 -Host <oracle-ip> -KeyPath <key-file>
```

Laptop:

```powershell
.\stop_training.ps1
```

## Acknowledgement and Retry Recovery

- Oracle writes completed games to `spool/ready/<game-id>.json.gz`.
- Laptop downloads one game, validates checksums/schema/hashes, imports through
  `db_import.write_batch`, then calls `sync_single_game` for teacher parquet.
- The laptop sends `/ack` only after both stages succeed.
- Repeating `/ack` is idempotent.
- If DB import succeeds but teacher sync fails, retry sees the DB row and
  completes teacher sync before ack.
- Unacknowledged games remain in Oracle `ready/` across service restarts.

## Promotion and Regression Arena

Oracle games are training data only. They are not the formal promotion test.
After each local training epoch, run the local candidate-vs-current regression
arena outside `games.db`, teacher data, and feature cache. Promotion rotates:

- `training_head`: latest finite checkpoint for optimizer continuation;
- `current_deployed`: last promoted weights sent to Oracle;
- `prior_deployed`: previous distinct promoted weights.

Held candidates are never sent to Oracle. Held promotion continues the next
training epoch. Clear large regression or repeated failures should restore the
last known-good training checkpoint according to the configured gate.
