#!/usr/bin/env bash
# Pull the latest code and restart everything. Safe to run while serving:
# the engine binary is replaced atomically and only restarts when you say so.
#
#   cd /opt/forge && ./update.sh          # code + gateway restart (engines keep running)
#   cd /opt/forge && ./update.sh engines  # also restart every running model engine
set -euo pipefail
cd "$(dirname "$0")"
[ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"
export PATH=$PATH:/usr/local/go/bin

echo "== pull"
git pull --ff-only

echo "== engine"
(cd engine && cargo build --release -q && mv -f target/release/forge-engine forge-engine)

echo "== gateway"
(cd gateway && go build -o forge-gateway .)

echo "== admin"
(cd admin && npm ci --silent && npm run build --silent)

if command -v systemctl >/dev/null && [ -d /run/systemd/system ]; then
  sudo systemctl restart forge-gateway
  echo "== gateway restarted"
else
  echo "== no systemd: restart ./start.sh yourself"
fi

if [ "${1:-}" = "engines" ]; then
  pkill -f forge-engine || true
  echo "== engines stopped — press 'start engine' in the admin (models load from the weight cache)"
fi
echo "done"
