# Deploying llmfa.st

One Linux box runs everything: the inference engines, the API gateway, the admin/customer
dashboard and the marketing site. **Caddy** terminates TLS (automatic Let's Encrypt, no certbot
cron to maintain) and **systemd** supervises the gateway (already on the box, restarts on crash,
survives reboots — pm2 is a Node process manager and we ship Go and Rust binaries).

## 1. First install

```bash
ssh root@<server-ip>
adduser llmfast && usermod -aG sudo llmfast && su - llmfast
curl -fsSL https://raw.githubusercontent.com/xindex2/llmfast-cpu/main/install.sh | bash
```

Builds into `/opt/llmfast`, writes `/opt/llmfast/gateway.env` with a generated `ADMIN_TOKEN`, and
starts `llmfast-gateway` on port 8080 under systemd.

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
sudo cp /opt/llmfast/deploy/Caddyfile /etc/caddy/Caddyfile
sudo cp -r /Users/you/llmfast/site /opt/llmfast/site      # the marketing site
sudo systemctl reload caddy                              # certificates are issued on first request
sudo ufw allow 22,80,443/tcp && sudo ufw enable          # keep 8080 internal
```

What each URL serves:

| URL | serves |
|---|---|
| `https://llmfa.st` | marketing site (`/opt/llmfast/site`) |
| `https://llmfa.st/v1/chat/completions` | the API — a friendly base URL for customers |
| `https://llmfa.st/app/admin/ui` | admin + customer dashboard |
| `https://api.llmfa.st` | the same API on its own host — **give OpenRouter this one**, so the backend can move hardware without touching the site |
| `https://api.llmfa.st/models` | the OpenRouter provider document |

## 3. Renaming an existing install (forge → llmfa.st)

Servers set up before the rename keep working: `./update.sh` retires the old
`forge-gateway.service`, installs `llmfast-gateway.service` pointing at wherever the checkout
lives, and the engine still reads `.forge-cache-*.bin` files so big models are not re-quantized.
The directory can stay `/opt/forge`; rename it only if you want to:

```bash
cd /opt/forge && ./update.sh          # migrates the service in place
# optional, and only if nothing else references the old path:
sudo systemctl stop llmfast-gateway
sudo mv /opt/forge /opt/llmfast && cd /opt/llmfast && ./update.sh
```

## 4. Updating

```bash
cd /opt/llmfast && ./update.sh           # pull, rebuild, restart gateway (engines keep serving)
cd /opt/llmfast && ./update.sh engines   # also stop engines, then press "start engine" in the admin
```

Engine restarts are cheap after the first load: quantized weights are cached next to the
checkpoint (`.llmfast-cache-<quant>.bin`), so a 27B starts in ~20 s instead of ~4 minutes.

## 5. Day-to-day

- **Models** page: paste a Hugging Face id, set prices, start/stop engines, pick quant and device.
- **Benchmarks** page: run real load at a chosen concurrency; p50/p95 TTFT and per-stream tok/s
  are the numbers OpenRouter publishes, aggregate tok/s sizes the fleet.
- **Launch** page: live checklist of every OpenRouter provider requirement.
- **Dashboard**: today / 7d / custom period, uptime, failures, earnings.
- Logs: `journalctl -u llmfast-gateway -f` and `<models dir>/<model>/engine.log`.

## 6. Useful settings (`/opt/llmfast/gateway.env`)

| variable | why |
|---|---|
| `MAX_INFLIGHT` | requests above this get an immediate 429 — never queue, OpenRouter counts queueing as slowness |
| `CAP_PROMPT_TPM`, `CAP_COMPLETION_TPM`, `CAP_RPM` | declared capacity; set from a real benchmark |
| `PROVIDER_SLUG`, `DATACENTER_COUNTRY`, `ZDR` | provider document fields |
| `GPU_MEM_MB` | on a GPU box, so `DEVICE=auto` knows what fits |
| `HF_TOKEN` | only for gated Hugging Face repos |

Engine tuning lives per model in the admin (quant, context, device, draft model) or as env in
`models.go`'s launcher: `THREADS`, `NGRAM`, `WCACHE`, `STATIC`, `PIN`, `FORCE_SIMD`.
