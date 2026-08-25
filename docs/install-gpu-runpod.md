# Install on a GPU server (RunPod, Vast, Lambda, any NVIDIA/AMD Linux box)

llmfa.st's GPU backend is our own: compute shaders (`engine/src/shaders.wgsl`) driven through `wgpu`,
which talks to **Vulkan** on Linux (NVIDIA and AMD). There is no CUDA dependency, so it runs
on any card with a Vulkan driver. The same binary auto-detects the GPU; if the model doesn't
fit (or no GPU is found) it falls back to the CPU engine.

> Status: the GPU path is validated bit-exact against the CPU engine (Metal, AMD), dense AND
> mixture-of-experts. NVIDIA/Vulkan is the same code through a different driver — run
> `--gpu-check` first (step 3). If anything fails, `DEVICE=cpu` keeps you serving while you
> report the issue.

## 0. What a card can actually deliver (read this before renting)

Decode streams the model's active bytes once per token, so tok/s ≈ card bandwidth ÷ GB/token.
No software change beats this — pick the model/card pair from the arithmetic:

| model | GB/token (q8) | A40 (696 GB/s) | 4090 (1008) | A100-80G (2039) |
|---|---|---|---|---|
| Qwen3-30B-A3B (MoE, 3B active) | ~3.3 | **~150 ceiling** | ~220 | ~440 |
| Qwen3-14B dense | ~15 | ~45 | ~65 | ~135 |
| Qwen3.8-27B (DeltaNet hybrid) | 28.85 | — CPU only, see below | — | — |

- **Qwen3-30B-A3B on an A40 is the 100-tok/s configuration.** MoE routing runs on the GPU and
  only the routed experts' weights are streamed. Weights are ~32 GB q8 — fits 48 GB VRAM.
- **Qwen3.8-27B does not run on the GPU backend** (its Gated-DeltaNet layers, attention gate
  and partial rotary are CPU-only), and even if it did, 28.85 GB/token puts 100 tok/s beyond
  every card short of an H100. Serve it on CPU or not at all.
- Ceilings are upper bounds; real decode lands beneath them (norms, attention, dispatch).
  Measure with `--bench-model` (step 3) before quoting numbers to OpenRouter.

## 1. Pick a pod

| card | VRAM | fits (int8) | fits (int4) | rough $/hr |
|---|---|---|---|---|
| RTX 4000 Ada / A4500 | 20 GB | up to ~14B | 30B-A3B, 27B dense (tight) | $0.25–0.30 |
| RTX 3090 / 4090 / A5000 | 24 GB | up to ~20B | 30B-A3B, 32B dense | $0.30–0.50 |
| A40 / A6000 / L40 | 48 GB | 30B-A3B, 32B dense | 70B dense | $0.45–0.60 |

Weights per model ≈ params × 1.1 bytes (int8) or × 0.63 (int4), plus KV cache (~1 GB per
concurrent 4k-context sequence on a 30B model).

**RunPod template**: "RunPod Pytorch" or any Ubuntu 22.04 CUDA image is fine (we only need the
driver). Expose **HTTP port 8080**. Use an **on-demand (non-interruptible)** pod: OpenRouter
deprioritizes providers below 95% uptime, and spot pods get killed.

## 2. Install

```bash
# in the pod's terminal (root) — safe to re-run after every container reset
curl -fsSL https://raw.githubusercontent.com/xindex2/llmfast-cpu/main/deploy/runpod-setup.sh | bash
```

The script installs Vulkan, writes the ICD manifest RunPod omits, pulls any driver userspace
libraries the mount is missing (matched to the host's exact driver version, cached on
/workspace), installs llmfast, points MODELS_DIR at /workspace/models, and starts the gateway.
It prints the admin token at the end. Manual equivalent, if you prefer:

```bash
apt-get update && apt-get install -y libvulkan1 vulkan-tools   # Vulkan loader + vulkaninfo
vulkaninfo --summary | grep -E "deviceName|driverName"          # must list your GPU
curl -fsSL https://raw.githubusercontent.com/xindex2/llmfast-cpu/main/install.sh | bash
```

If `vulkaninfo` errors with `ERROR_INCOMPATIBLE_DRIVER` / "Found no drivers", fix it in this
order (learned the hard way on a real A40 pod):

1. **Env var** — pod env `NVIDIA_DRIVER_CAPABILITIES=all` (RunPod → Edit Pod → Environment
   Variables; this resets the container). Default pods mount only the compute slice of the
   driver — no Vulkan libraries at all.
2. **The missing manifest** — with the env var set, RunPod mounts the libraries
   (`ls /usr/lib/x86_64-linux-gnu/ | grep glcore` shows them) but often NOT the loader
   manifest, and Vulkan still fails. Write it yourself:

   ```bash
   mkdir -p /usr/share/vulkan/icd.d
   printf '{ "file_format_version": "1.0.0", "ICD": { "library_path": "libGLX_nvidia.so.0", "api_version": "1.3.277" } }\n' > /usr/share/vulkan/icd.d/nvidia_icd.json
   ```

Do NOT `apt-get install libnvidia-gl-*` in a RunPod container: the repo's point release will
not match the host kernel module (userspace and kernel must match exactly), and dpkg fails
with "Invalid cross-device link" on the bind-mounted driver files anyway.

**Container resets**: every Edit Pod wipes everything outside `/workspace`. Put models on the
volume (`MODELS_DIR=/workspace/models` in gateway.env) so a reset costs a 2-minute reinstall,
not a re-download.

Containers have no systemd, so start the gateway with:

```bash
cd /opt/llmfast
echo "GPU_MEM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)" >> gateway.env
nohup ./start.sh > gateway.log 2>&1 &
```

`GPU_MEM_MB` tells `DEVICE=auto` how big a model the card can take (weights must fit in half of it).

## 3. Verify the GPU path before serving

```bash
cd /opt/llmfast/engine
MODEL=../models/qwen3-0.6b ./target/release/llmfast-engine --gpu-bench    # kernel vs CPU, GB/s
MODEL=../models/qwen3-0.6b ./target/release/llmfast-engine --gpu-check    # full forward: GPU logits == CPU logits
QUANT=q8 DEVICE=gpu GPU_MEM_MB=49152 ./llmfast-engine --bench-model /opt/llmfast/models/qwen3-30b-a3b
```

`--bench-model` prints GB streamed/token, measured tok/s and achieved GB/s against the card —
that line, not the ceiling table, is what goes in the OpenRouter application.

Expect `max abs err 0.0000` and matching argmax. On an NVIDIA card the decode number should be
several times the CPU number (the design is bandwidth-bound at ~15 µs per dispatch on Vulkan).

## 4. Add models

Admin UI: `https://<pod-id>-8080.proxy.runpod.net/admin/ui` → Settings (base URL = that host,
admin token from `gateway.env`) → Models → e.g. `Qwen/Qwen3-30B-A3B`, quant `q8` (24 GB+ cards)
or `q4`, device `auto` → start engine. The Dashboard shows which device each engine landed on.

Current GPU limitations (CPU fallback is automatic): **GPU context is capped at 2048 tokens**
(the attention shader's shared-memory tile — fine for benchmarking and short-context serving,
not yet for long-context customers), and very old GPU drivers may be dispatch-overhead-bound.
Loading stages the q8 model in CPU RAM before upload, so the pod needs RAM ≥ model size + a
few GB (a 50 GB pod fits the 32 GB 30B-A3B).

## 5. Going live

RunPod's HTTP proxy terminates TLS for you, so `https://<pod-id>-8080.proxy.runpod.net/models`
already works as an OpenRouter provider document URL. For a custom domain (`api.llmfa.st`),
point DNS at a small VPS running Caddy (`deploy/Caddyfile`) that proxies to the pod, or run
the gateway on the VPS and point `ENGINE_URL` at the pod's engine port.
