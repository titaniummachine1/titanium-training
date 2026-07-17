#!/usr/bin/env bash
set -euo pipefail
INSTALL_DIR="${INSTALL_DIR:-/opt/titanium-game-factory}"
sudo systemctl stop titanium-game-factory.service || true
if [[ -d "${INSTALL_DIR}/engine" ]]; then
  pushd "${INSTALL_DIR}/engine" >/dev/null
  sudo -u titanium bash -lc "source ~/.cargo/env && RUSTFLAGS='-C target-cpu=native' cargo build --release"
  popd >/dev/null
fi
sudo systemctl start titanium-game-factory.service

