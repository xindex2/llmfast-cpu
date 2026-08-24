#!/usr/bin/env bash
# Pull the latest code and restart everything. Safe to run while serving:
# the engine binary is replaced atomically and only restarts when you say so.
#
#   cd /opt/llmfast && ./update.sh          # code + gateway restart (engines keep running)
#   cd /opt/llmfast && ./update.sh engines  # also restart every running model engine
set -euo pipefail
cd "$(dirname "$0")"
[ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"
export PATH=$PATH:/usr/local/go/bin

echo "== pull"
git pull --ff-only

echo "== engine"
# Copy via a temp name, then rename over the old binary. A bare `mv` fails with "are the same
# file" when engine/llmfast-engine is a symlink into target/release, and renaming (rather than
# copying onto) the destination avoids "Text file busy" while an engine is running.
(cd engine && cargo build --release -q \
  && cp -f target/release/llmfast-engine llmfast-engine.new \
  && chmod +x llmfast-engine.new \
  && mv -f llmfast-engine.new llmfast-engine)

echo "== gateway"
(cd gateway && go build -o llmfast-gateway .)

echo "== admin"
(cd admin && npm ci --silent && npm run build --silent)

if command -v systemctl >/dev/null && [ -d /run/systemd/system ]; then
  # (re)install the unit pointing at this checkout, and retire the pre-rename one if present
  if [ -f /etc/systemd/system/forge-gateway.service ]; then
    sudo systemctl disable --now forge-gateway || true
    sudo rm -f /etc/systemd/system/forge-gateway.service
    echo "== migrated forge-gateway.service -> llmfast-gateway.service"
  fi
  HERE=$(pwd)
  sudo cp deploy/llmfast-gateway.service /etc/systemd/system/llmfast-gateway.service
  sudo sed -i "s#/opt/llmfast#$HERE#g; s#^User=llmfast#User=$(id -un)#" /etc/systemd/system/llmfast-gateway.service
  sudo systemctl daemon-reload
  sudo systemctl enable llmfast-gateway >/dev/null 2>&1 || true
  sudo systemctl restart llmfast-gateway
  echo "== gateway restarted"
else
  echo "== no systemd: restart ./start.sh yourself"
fi

if [ "${1:-}" = "engines" ]; then
  pkill -f llmfast-engine || true
  echo "== engines stopped — press 'start engine' in the admin (models load from the weight cache)"
fi
echo "done"
