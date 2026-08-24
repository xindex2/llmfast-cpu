# Deploying llmfa.st

One Linux box runs everything: the inference engines, the API gateway, the admin/customer
dashboard and the marketing site. **Caddy** terminates TLS (automatic Let's Encrypt, no certbot
cron to maintain) and **systemd** supervises the gateway (already on the box, restarts on crash,
survives reboots — pm2 is a Node process manager and we ship Go and Rust binaries).

## 1. First install

```bash
ssh root@<server-ip>
adduser forge && usermod -aG sudo forge && su - forge
curl -fsSL https://raw.githubusercontent.com/xindex2/llmfast-cpu/main/install.sh | bash
```

Builds into `/opt/forge`, writes `/opt/forge/gateway.env` with a generated `ADMIN_TOKEN`, and
starts `forge-gateway` on port 8080 under systemd.

Put the tools on your PATH permanently (the installer's shells are throwaway):

```bash
echo 'source ~/.cargo/env' >> ~/.bashrc
echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc
```

## 2. Domain and TLS

DNS — both records point at the same server:

| type | name | value |
|---|---|---|
| A | `llmfa.st` | server IP |
| A | `api` | server IP |
| AAAA | same two | IPv6, if you have one |

```bash
sudo apt install -y caddy
sudo cp /opt/forge/deploy/Caddyfile /etc/caddy/Caddyfile
sudo cp -r /Users/you/llmfast/site /opt/forge/site      # the marketing site
sudo systemctl reload caddy                              # certificates are issued on first request
sudo ufw allow 22,80,443/tcp && sudo ufw enable          # keep 8080 internal
```

What each URL serves:

| URL | serves |
|---|---|
| `https://llmfa.st` | marketing site (`/opt/forge/site`) |
| `https://llmfa.st/v1/chat/completions` | the API — a friendly base URL for customers |
| `https://llmfa.st/app/admin/ui` | admin + customer dashboard |
| `https://api.llmfa.st` | the same API on its own host — **give OpenRouter this one**, so the backend can move hardware without touching the site |
| `https://api.llmfa.st/models` | the OpenRouter provider document |

## 3. Updating

```bash
cd /opt/forge && ./update.sh           # pull, rebuild, restart gateway (engines keep serving)
cd /opt/forge && ./update.sh engines   # also stop engines, then press "start engine" in the admin
```

Engine restarts are cheap after the first load: quantized weights are cached next to the
checkpoint (`.forge-cache-<quant>.bin`), so a 27B starts in ~20 s instead of ~4 minutes.

## 4. Day-to-day

- **Models** page: paste a Hugging Face id, set prices, start/stop engines, pick quant and device.
- **Benchmarks** page: run real load at a chosen concurrency; p50/p95 TTFT and per-stream tok/s
  are the numbers OpenRouter publishes, aggregate tok/s sizes the fleet.
- **Launch** page: live checklist of every OpenRouter provider requirement.
- **Dashboard**: today / 7d / custom period, uptime, failures, earnings.
- Logs: `journalctl -u forge-gateway -f` and `<models dir>/<model>/engine.log`.

## 5. Useful settings (`/opt/forge/gateway.env`)

| variable | why |
|---|---|
| `MAX_INFLIGHT` | requests above this get an immediate 429 — never queue, OpenRouter counts queueing as slowness |
| `CAP_PROMPT_TPM`, `CAP_COMPLETION_TPM`, `CAP_RPM` | declared capacity; set from a real benchmark |
| `PROVIDER_SLUG`, `DATACENTER_COUNTRY`, `ZDR` | provider document fields |
| `GPU_MEM_MB` | on a GPU box, so `DEVICE=auto` knows what fits |
| `HF_TOKEN` | only for gated Hugging Face repos |

Engine tuning lives per model in the admin (quant, context, device, draft model) or as env in
`models.go`'s launcher: `THREADS`, `NGRAM`, `WCACHE`, `STATIC`, `PIN`, `FORCE_SIMD`.
