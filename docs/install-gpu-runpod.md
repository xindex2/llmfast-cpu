# Install on a GPU server (RunPod, Vast, Lambda, any NVIDIA/AMD Linux box)

Forge's GPU backend is our own: compute shaders (`engine/src/shaders.wgsl`) driven through `wgpu`,
which talks to **Vulkan** on Linux (NVIDIA and AMD). There is no CUDA dependency, so it runs
on any card with a Vulkan driver. The same binary auto-detects the GPU; if the model doesn't
fit (or no GPU is found) it falls back to the CPU engine.

> Status: the GPU path is validated bit-exact against the CPU engine (Metal, AMD). NVIDIA/Vulkan
> is the same code through a different driver — run `--gpu-check` first (step 3). If anything
> fails, `DEVICE=cpu` keeps you serving while you report the issue.

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
# in the pod's terminal (root)
apt-get update && apt-get install -y libvulkan1 vulkan-tools   # Vulkan loader + vulkaninfo
vulkaninfo --summary | grep -E "deviceName|driverName"          # must list your GPU
curl -fsSL https://raw.githubusercontent.com/xindex2/llmfast-cpu/main/install.sh | bash
```

If `vulkaninfo` shows no device, the container is missing the NVIDIA Vulkan ICD. Fix: start the
pod with env `NVIDIA_DRIVER_CAPABILITIES=all` (RunPod → Edit Pod → Environment Variables), or
`apt-get install -y libnvidia-gl-<driver-major>` matching `nvidia-smi`'s driver version.

Containers have no systemd, so start the gateway with:

```bash
cd /opt/forge
echo "GPU_MEM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)" >> gateway.env
nohup ./start.sh > gateway.log 2>&1 &
```

`GPU_MEM_MB` tells `DEVICE=auto` how big a model the card can take (weights must fit in half of it).

## 3. Verify the GPU path before serving

```bash
cd /opt/forge/engine
MODEL=../models/qwen3-0.6b ./target/release/forge-engine --gpu-bench    # kernel vs CPU, GB/s
MODEL=../models/qwen3-0.6b ./target/release/forge-engine --gpu-check    # full forward: GPU logits == CPU logits
```

Expect `max abs err 0.0000` and matching argmax. On an NVIDIA card the decode number should be
several times the CPU number (the design is bandwidth-bound at ~15 µs per dispatch on Vulkan).

## 4. Add models

Admin UI: `https://<pod-id>-8080.proxy.runpod.net/admin/ui` → Settings (base URL = that host,
admin token from `gateway.env`) → Models → e.g. `Qwen/Qwen3-30B-A3B`, quant `q8` (24 GB+ cards)
or `q4`, device `auto` → start engine. The Dashboard shows which device each engine landed on.

Current GPU limitations (CPU fallback is automatic): MoE models run on CPU (GPU MoE is next),
GPU context is capped at 2048 tokens, and very old GPU drivers may be dispatch-overhead-bound.

## 5. Going live

RunPod's HTTP proxy terminates TLS for you, so `https://<pod-id>-8080.proxy.runpod.net/models`
already works as an OpenRouter provider document URL. For a custom domain (`api.llmfa.st`),
point DNS at a small VPS running Caddy (`deploy/Caddyfile`) that proxies to the pod, or run
the gateway on the VPS and point `ENGINE_URL` at the pod's engine port.
