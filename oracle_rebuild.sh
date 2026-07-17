#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.cargo/bin:$PATH"
cd /opt/titanium-game-factory/engine
RUSTFLAGS='-C target-cpu=native' cargo build --release
sudo systemctl restart titanium-game-factory.service
sleep 3
systemctl is-active titanium-game-factory.service
grep '^ExecStart=' /etc/systemd/system/titanium-game-factory.service
