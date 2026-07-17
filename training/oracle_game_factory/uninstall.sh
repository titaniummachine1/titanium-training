#!/usr/bin/env bash
set -euo pipefail
sudo systemctl stop titanium-game-factory.service || true
sudo systemctl disable titanium-game-factory.service || true
sudo rm -f /etc/systemd/system/titanium-game-factory.service
sudo systemctl daemon-reload
echo "Mutable data remains under /var/lib/titanium-game-factory"

