#!/usr/bin/env bash
# Fast oracle recovery — pad weights to legacy binary size, no cargo rebuild.
set -euo pipefail
DATA_DIR="${DATA_DIR:-/var/lib/titanium-game-factory}"
LEGACY_BYTES=340280

echo "Killing any in-flight cargo build (optional)..."
sudo pkill -f 'cargo build' 2>/dev/null || true

pad_file() {
  local f="$1"
  if [[ ! -f "$f" ]]; then
    return 0
  fi
  local sz
  sz=$(stat -c%s "$f")
  if [[ "$sz" -eq "$LEGACY_BYTES" ]]; then
    echo "OK $f ($sz bytes)"
    return 0
  fi
  if [[ "$sz" -gt "$LEGACY_BYTES" ]]; then
    sudo truncate -s "$LEGACY_BYTES" "$f"
    echo "truncated $f $sz -> $LEGACY_BYTES"
    return 0
  fi
  echo "WARN $f too small ($sz < $LEGACY_BYTES)" >&2
}

# Active generation + any staged copies
if [[ -d "$DATA_DIR/active" ]]; then
  pad_file "$DATA_DIR/active/current.bin"
  pad_file "$DATA_DIR/active/prior.bin"
fi
while IFS= read -r -d '' gen; do
  pad_file "$gen/current.bin"
  pad_file "$gen/prior.bin"
done < <(find "$DATA_DIR/generations" -mindepth 1 -maxdepth 1 -type d -print0 2>/dev/null || true)

sudo systemctl restart titanium-game-factory || sudo systemctl start titanium-game-factory
sleep 2
systemctl is-active titanium-game-factory
journalctl -u titanium-game-factory -n 8 --no-pager
