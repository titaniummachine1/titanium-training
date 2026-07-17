#!/usr/bin/env bash
set -euo pipefail
INSTALL_DIR="${INSTALL_DIR:-/opt/titanium-game-factory}"
sudo chown -R titanium:titanium "${INSTALL_DIR}"
cd "${INSTALL_DIR}/engine"
sudo -u titanium bash -lc 'source ~/.cargo/env && RUSTFLAGS="-C target-cpu=native" cargo build --release'
sudo systemctl restart titanium-game-factory
sleep 3
systemctl is-active titanium-game-factory
sudo journalctl -u titanium-game-factory -n 5 --no-pager
