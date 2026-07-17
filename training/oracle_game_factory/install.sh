#!/usr/bin/env bash
set -euo pipefail

INSTALL_DIR="${INSTALL_DIR:-/opt/titanium-game-factory}"
DATA_DIR="${DATA_DIR:-/var/lib/titanium-game-factory}"
LOG_DIR="${LOG_DIR:-/var/log/titanium-game-factory}"
WORKERS="${WORKERS:-13}"
MOVE_TIME="${MOVE_TIME:-5.0}"
NODE_BUDGET="${NODE_BUDGET:-200000}"

if [[ "${EUID}" -ne 0 ]]; then
  echo "install.sh must run with sudo/root" >&2
  exit 1
fi

useradd --system --home "${DATA_DIR}" --shell /usr/sbin/nologin titanium 2>/dev/null || true
mkdir -p "${INSTALL_DIR}" "${DATA_DIR}" "${LOG_DIR}" "${INSTALL_DIR}/weights"
chown -R titanium:titanium "${DATA_DIR}" "${LOG_DIR}" "${INSTALL_DIR}/weights"

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential curl git python3 python3-venv pkg-config libssl-dev

if ! command -v cargo >/dev/null 2>&1; then
  sudo -u titanium bash -lc 'curl https://sh.rustup.rs -sSf | sh -s -- -y'
fi

if [[ -d "${INSTALL_DIR}/engine" ]]; then
  chown -R titanium:titanium "${INSTALL_DIR}"
  pushd "${INSTALL_DIR}/engine" >/dev/null
  sudo -u titanium bash -lc "source ~/.cargo/env && RUSTFLAGS='-C target-cpu=native' cargo build --release"
  popd >/dev/null
fi

install -m 0644 "${INSTALL_DIR}/systemd/titanium-game-factory.service" /etc/systemd/system/titanium-game-factory.service
sed -i "s|__INSTALL_DIR__|${INSTALL_DIR}|g; s|__DATA_DIR__|${DATA_DIR}|g; s|__WORKERS__|${WORKERS}|g; s|__MOVE_TIME__|${MOVE_TIME}|g; s|__NODE_BUDGET__|${NODE_BUDGET}|g" /etc/systemd/system/titanium-game-factory.service
systemctl daemon-reload
systemctl enable titanium-game-factory.service

echo "Installed. Start with: sudo systemctl start titanium-game-factory"
echo "API token: ${DATA_DIR}/api_token (created on first service start)"

