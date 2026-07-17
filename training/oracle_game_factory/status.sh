#!/usr/bin/env bash
set -euo pipefail
systemctl status --no-pager titanium-game-factory.service || true
echo
echo "nproc: $(nproc)"
free -h
df -h / /var/lib/titanium-game-factory 2>/dev/null || true
journalctl -u titanium-game-factory.service -n 80 --no-pager || true

