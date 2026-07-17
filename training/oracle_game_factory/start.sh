#!/usr/bin/env bash
set -euo pipefail
sudo systemctl start titanium-game-factory.service
sudo systemctl status --no-pager titanium-game-factory.service

