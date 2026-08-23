# Deploying Forge at api.llmfa.st

One Linux box (CPU or GPU). Everything runs as systemd services; Caddy does TLS.

```bash
# 0. DNS: A record  api.llmfa.st → server IP   (and AAAA for IPv6 if you have it)

# 1. build (on the server, or cross-compile and scp)
sudo apt install -y build-essential curl git
curl https://sh.rustup.rs -sSf | sh -s -- -y && source ~/.cargo/env
# Go 1.24+: https://go.dev/dl/   Node 22: https://nodejs.org
git clone <your repo> /opt/forge && cd /opt/forge
(cd engine  && cargo build --release)                 # → engine/target/release/forge-engine
(cd gateway && go build -o forge-gateway .)
(cd admin   && npm ci && npm run build)               # → admin/dist
sudo useradd -r -s /usr/sbin/nologin forge; sudo chown -R forge /opt/forge
sudo cp engine/target/release/forge-engine /opt/forge/engine/forge-engine

# 2. configure
sudo cp deploy/gateway.env.example /opt/forge/gateway.env && sudo nano /opt/forge/gateway.env
sudo cp deploy/forge-gateway.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now forge-gateway

# 3. TLS + domain (Caddy: automatic certificates)
sudo apt install -y caddy && sudo cp deploy/Caddyfile /etc/caddy/Caddyfile && sudo systemctl reload caddy

# 4. models: open https://api.llmfa.st/admin/ui → Settings (base URL https://api.llmfa.st, admin token)
#    → Models → add "Qwen/Qwen3-30B-A3B" (quant q4 on CPU, q8 on a 24 GB GPU) → start engine.
```

Endpoints OpenRouter needs:
- `https://api.llmfa.st/models` — provider document (schema 2.4), auto-generated from the Models page
- `https://api.llmfa.st/v1/chat/completions` — OpenAI-compatible, streaming, `stop`, `cached_tokens`

Engines are started per model by the gateway and auto-pick **GPU** (wgpu: Vulkan on Linux/NVIDIA) when the
weights fit `GPU_MEM_MB`, otherwise **CPU** (AVX2/AVX kernels, NUMA-aware). Force with the device selector
on the Models page. Engine logs: `<models dir>/<model>/engine.log`.

Checklist before applying to OpenRouter: a privacy policy and terms page on llmfa.st, uptime monitoring
(the gateway's `/health`), and a benchmark run on the real box (Benchmarks page) to set `CAP_*` honestly.
