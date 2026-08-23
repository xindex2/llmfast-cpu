# Install on a CPU VPS (DigitalOcean, Hetzner, OVH, any Ubuntu box)

Forge runs entirely on CPU: our own Rust engine (AVX2/AVX kernels, int8/int4, batching, prefix
cache, speculation), a Go gateway (OpenAI-compatible API + OpenRouter provider document), and a
React admin. No CUDA, no Python, no third-party inference libraries.

## 1. Pick a droplet

| model you want to serve | RAM needed (int8 / int4) | DigitalOcean size that works | expected single-stream speed* |
|---|---|---|---|
| Qwen3-0.6B (testing) | 1 GB / 0.5 GB | Basic 2 vCPU / 4 GB ($24/mo) | 10–20 tok/s |
| Qwen3-1.7B | 2.5 GB / 1.5 GB | 4 vCPU / 8 GB | 8–15 tok/s |
| Qwen3-4B | 5 GB / 3 GB | CPU-Optimized 8 vCPU / 16 GB | 8–12 tok/s |
| **Qwen3-30B-A3B (MoE)** | 32 GB / 19 GB | **CPU-Optimized 16 vCPU / 32 GB** (int4) or 32 vCPU / 64 GB (int8) | 15–30 tok/s |

\* decode speed is memory-bandwidth-bound: bytes-per-token ÷ RAM bandwidth. Cloud VPS RAM is
usually 20–40 GB/s; a dedicated 8–12 channel DDR4/DDR5 server is 3–10× faster for the same money.
Dense 27B–32B models need ~15 GB/token at int4 → ~1–3 tok/s on a VPS: not worth serving.
MoE models (30B-A3B, 80B-A3B) are the CPU sweet spot because only ~3B params are touched per token.

Choose **Ubuntu 24.04 x64**. For AMD EPYC / Intel Xeon (AVX2) droplets the fast kernels are
selected automatically; on very old CPUs the engine falls back to AVX or scalar.

## 2. Install (one command)

```bash
ssh root@<droplet-ip>
adduser forge && usermod -aG sudo forge && su - forge
curl -fsSL https://raw.githubusercontent.com/xindex2/llmfast-cpu/main/install.sh | bash
```

This builds everything into `/opt/forge`, generates an admin token in `/opt/forge/gateway.env`,
and starts `forge-gateway` as a systemd service on port 8080.

Manual equivalent:

```bash
sudo apt install -y build-essential curl git
curl -sSf https://sh.rustup.rs | sh -s -- -y && source ~/.cargo/env
# Go 1.24 and Node 22 as in install.sh
git clone https://github.com/xindex2/llmfast-cpu /opt/forge && cd /opt/forge
(cd engine && cargo build --release && cp target/release/forge-engine forge-engine)
(cd gateway && go build -o forge-gateway .) && (cd admin && npm ci && npm run build)
cp deploy/gateway.env.example gateway.env   # edit ADMIN_TOKEN, paths
./start.sh                                  # or the systemd unit in deploy/
```

## 3. Add a model from the admin

Open `http://<droplet-ip>:8080/admin/ui` → **Settings**: base URL `http://<droplet-ip>:8080`, paste
the admin token → **Models** → paste `Qwen/Qwen3-0.6B` (or `Qwen/Qwen3-30B-A3B`), choose
quant `q4` for big models on CPU, set prices → **add & download** → **start engine**.

The first start quantizes the checkpoint (≈1 min per 10 GB). Then test in **Playground** and run
**Benchmarks** — the "aggregate tok/s" column is what sizes your fleet, and the earnings columns
use the prices you set.

## 4. CPU tuning (`Models` → engine env, or `gateway.env`)

| variable | default | when to change |
|---|---|---|
| `THREADS` | all logical CPUs | try physical-core count; hyperthreads rarely help bandwidth-bound decode |
| `QUANT` | q8 | `q4` halves bytes/token (and RAM) — use for ≥7B models; `q8` for small models where quality matters |
| `MAX_CONTEXT` | 4096 | raise for long prompts; KV cache is ~0.23 MB/token for 0.6B, scales with model |
| `NGRAM` | on | prompt-lookup speculation; 2–3× on copy-heavy output, neutral otherwise |
| `DRAFT_MODEL` | — | dir of a small same-family model for speculative decoding (big dense targets) |
| `PIN`, `STATIC` | on (Linux) | NUMA-aware pinning / owner-first scheduling; set `PIN=0` on tiny shared VPSes |

Diagnostics: `engine/target/release/forge-engine --bench` (kernel GFLOPS & GB/s),
`--prompt "hi"` (CLI generation), `PROFILE=1` (per-step timing in the engine log).

## 5. Domain + TLS + OpenRouter

Point `api.yourdomain` at the droplet, install Caddy with `deploy/Caddyfile` (automatic HTTPS),
and your provider document is at `https://api.yourdomain/models`. Set `PROVIDER_SLUG`,
`DATACENTER_COUNTRY`, `CAP_*` in `gateway.env` from real benchmark numbers before applying.
Firewall: `ufw allow 22,80,443/tcp` and keep 8080 internal (Caddy proxies to it).
