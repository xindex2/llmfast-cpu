#!/usr/bin/env bash
# One-shot installer for a fresh Ubuntu 22.04/24.04 box (CPU VPS or GPU server).
# Builds engine + gateway + admin, installs to /opt/llmfast, and (if systemd is available) starts the gateway.
#   curl -fsSL https://raw.githubusercontent.com/xindex2/llmfast-cpu/main/install.sh | bash
set -euo pipefail
REPO=${REPO:-https://github.com/xindex2/llmfast-cpu.git}
DEST=${DEST:-/opt/llmfast}
SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"

echo "== packages"
$SUDO apt-get update -qq
$SUDO apt-get install -y -qq build-essential curl git pkg-config ca-certificates libvulkan1 >/dev/null   # (GPU boxes: see docs/install-gpu-runpod.md for the vendor Vulkan ICD)

echo "== rust"
if ! command -v cargo >/dev/null; then curl -sSf https://sh.rustup.rs | sh -s -- -y -q; fi
source "$HOME/.cargo/env"

echo "== go"
if ! command -v go >/dev/null; then
  GOV=1.24.5; curl -sL "https://go.dev/dl/go${GOV}.linux-amd64.tar.gz" | $SUDO tar -C /usr/local -xz
fi
export PATH=$PATH:/usr/local/go/bin

echo "== node"
if ! command -v node >/dev/null; then curl -fsSL https://deb.nodesource.com/setup_22.x | ${SUDO:+$SUDO -E} bash - >/dev/null && $SUDO apt-get install -y -qq nodejs >/dev/null; fi

echo "== source → $DEST"
if [ -d "$DEST/.git" ]; then git -C "$DEST" pull -q; else $SUDO mkdir -p "$DEST" && $SUDO chown "$(id -u)" "$DEST" && git clone -q "$REPO" "$DEST"; fi
cd "$DEST"

echo "== build engine (this is the slow part, ~2-5 min)"
(cd engine && cargo build --release 2>&1 | tail -1)
install -m 755 engine/target/release/llmfast-engine engine/llmfast-engine.new && mv -f engine/llmfast-engine.new engine/llmfast-engine   # atomic replace, works while the old binary runs
echo "== build gateway"
(cd gateway && go build -o llmfast-gateway .)
echo "== build admin"
(cd admin && npm ci --silent && npm run build --silent)

mkdir -p models
if [ ! -f gateway.env ]; then
  cp deploy/gateway.env.example gateway.env
  sed -i "s#/opt/llmfast#$DEST#g; s#change-me-long-random#$(head -c 24 /dev/urandom | base64 | tr -d '/+=')#" gateway.env
  echo "== wrote $DEST/gateway.env (ADMIN_TOKEN generated — see inside)"
fi

if command -v systemctl >/dev/null && [ -d /run/systemd/system ]; then
  $SUDO cp deploy/llmfast-gateway.service /etc/systemd/system/llmfast-gateway.service
  $SUDO sed -i "s#/opt/llmfast#$DEST#g; s#^User=llmfast#User=$(id -un)#" /etc/systemd/system/llmfast-gateway.service
  $SUDO systemctl daemon-reload && $SUDO systemctl enable --now llmfast-gateway
  echo "== gateway running as a service:  systemctl status llmfast-gateway"
else
  echo "== no systemd (container?): start with   ./start.sh"
fi
echo
echo "Admin UI:  http://<server-ip>:8080/admin/ui      token: $(grep ADMIN_TOKEN gateway.env | cut -d= -f2)"
echo "API:       http://<server-ip>:8080/v1/chat/completions"
