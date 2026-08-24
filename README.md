# llmfa.st (llmfast) — your own LLM inference provider, CPU or GPU

**Install guides:** [CPU VPS (DigitalOcean etc.)](docs/install-cpu-vps.md) · [GPU server (RunPod etc.)](docs/install-gpu-runpod.md) · [domain + TLS](deploy/README.md)

One-liner on a fresh Ubuntu box: `curl -fsSL https://raw.githubusercontent.com/xindex2/llmfast-cpu/main/install.sh | bash`

Own-built inference provider: run big open MoE models (Qwen3-30B-A3B, Qwen3-235B-A22B)
on commodity CPU servers and sell tokens through OpenRouter.

```
llmfa.st/
  engine/    Rust   — inference engine: CPU (AVX2/AVX kernels) and GPU (our WGSL shaders via wgpu), MoE, batching, speculation
  gateway/   Go     — OpenAI-compatible API, API keys, routing, metering, admin API
  admin/     React  — admin dashboard, playground, benchmarks, earnings calculator
```

## Scale math (the numbers everything is designed around)

Per-token cost on CPU = bytes moved through RAM. Dense 27B BF16 = 54 GB/token (2–5 tok/s).
MoE 30B-A3B int4 ≈ 1.8 GB/token → 30–50 tok/s single stream, ~300 tok/s batched per server.

| Target         | tok/s sustained | servers (MoE int4, batched) |
|----------------|-----------------|-----------------------------|
| 100M tok/day   | ~1,160          | 4–6                         |
| 1B tok/day     | ~11,600         | 40–60                       |

Design rule: everything is horizontal — a server is one `engine` process, the gateway
load-balances across N of them, stats are aggregated centrally.

## Milestones

1. **Gateway + admin running with a mock engine** (this scaffold) — playground streams, stats flow.
2. ~~**Engine M1 – correctness**~~ done: safetensors loader, byte-level BPE (matches HF exactly), Qwen3 dense+MoE forward pass, **Qwen3.5 hybrid** (gated DeltaNet linear attention + gated full attention, validated against the HF reference via `--fixture` + `scripts/make_q35_fixture.py`). Qwen3.5 runs on CPU; speculation/prefix-rollback auto-disabled (recurrent state).
3. **Engine M2 – speed**: ~~thread pool, AVX2 kernels, batched prefill, Q8~~ done; next: int4, AVX-512/AMX, KV-cache int8, speculative decoding.
4. **Engine M3 – throughput**: ~~continuous batching~~ done; next: chunked prefill, prefix cache, paged KV. Benchmark vs llama.cpp.
5. **Provider launch**: ~~OpenRouter `/models` document (schema 2.4), early 429s, cached-token billing, SSE keep-alives, stop sequences~~ done; next: domain + TLS, uptime monitoring, tool calling, apply.

## Run locally

```bash
# engine (needs a checkpoint dir with config.json, model.safetensors, tokenizer.json)
cd engine && cargo build --release
MODEL=../models/qwen3-0.6b ./target/release/llmfast-engine                    # serves :9000
MODEL=../models/qwen3-0.6b ./target/release/llmfast-engine --prompt "hello"   # one-shot CLI
MODEL=../models/qwen3-0.6b ./target/release/llmfast-engine --bench            # kernel GFLOPS/GB-s

# gateway (OpenAI API + admin API + provider document)
cd gateway && go build -o llmfast-gateway . && ./llmfast-gateway                # :8080

# admin UI
cd admin && npm install && npm run dev                                      # :5173

curl -N localhost:8080/v1/chat/completions -H 'Authorization: Bearer dev-key' \
  -d '{"model":"qwen3-0.6b","stream":true,"messages":[{"role":"user","content":"hi"}]}'
```

## Deploy on a server (llmfa.st)

Full walkthrough — DNS, TLS, updates, day-to-day — in [deploy/README.md](deploy/README.md).
Short version: **Caddy** for TLS (automatic certificates), **systemd** for supervision.

```bash
# first install on fresh Ubuntu
curl -fsSL https://raw.githubusercontent.com/xindex2/llmfast-cpu/main/install.sh | bash
sudo cp /opt/llmfast/deploy/Caddyfile /etc/caddy/Caddyfile && sudo systemctl reload caddy

# every update after that
cd /opt/llmfast && ./update.sh            # pull + rebuild + restart gateway
cd /opt/llmfast && ./update.sh engines    # also restart the model engines
```

| URL | serves |
|---|---|
| `llmfa.st` | marketing site |
| `llmfa.st/v1/chat/completions` | API (customer-facing base URL) |
| `llmfa.st/app/admin/ui` | admin + customer dashboard |
| `api.llmfa.st` + `/models` | API and provider document — point OpenRouter here |

Speculative decoding: `MTP_K` (self-speculation with the checkpoint's multi-token-prediction
head, default 1 when present, 0 disables), `NGRAM`/`NGRAM_N`/`NGRAM_K` (prompt lookup),
`DRAFT_MODEL`+`SPEC_K` (separate draft model).

Engine env: `DEVICE=auto|cpu|gpu`, `GPU_MEM_MB`, `MAX_CONTEXT`, `QUANT=q8|q4|bf16`, `THREADS`,
`NGRAM`, `DRAFT_MODEL`+`SPEC_K`, `WCACHE=0` (disable the quantized weight cache), `STATIC`, `PIN`,
`FORCE_SIMD=0|1|2`, `PROFILE=1`, `LAYER_DEBUG=1`.
Gateway env: `ENGINE_URL`, `MODELS`, `MAX_INFLIGHT`, `ADMIN_TOKEN`, `STORE`, `ADMIN_DIR`,
`PROVIDER_SLUG`, `QUANTIZATION`, `DATACENTER_COUNTRY/REGION`, `CAP_PROMPT_TPM`,
`CAP_COMPLETION_TPM`, `CAP_RPM`, `ZDR`, `IS_READY`, `HF_TOKEN`.

## What the admin does

- **Dashboard** — tokens, earnings, uptime, TTFT p50/p95, failures; today / yesterday / 7d / 30d / custom range.
- **Playground** — streaming chat with live TTFT and tok/s; reasoning shown separately.
- **Models** — add by Hugging Face id, price it, choose quant/context/device/draft model, start and stop engines.
- **Benchmarks** — real load at a chosen concurrency: p50/p95 TTFT, per-stream and aggregate tok/s, servers needed, revenue and profit.
- **Earnings** — unit economics: prices, OpenRouter cut, server cost, utilization → profit per month.
- **Customers** — accounts, prepaid credit, usage, top-ups.
- **Account** — customer sign-in, self-serve API keys, balance, quickstart.
- **Launch** — live checklist of every OpenRouter provider requirement.

## GPU backend

`--gpu-bench` (kernel check), `--gpu-check` (full forward: GPU logits vs CPU logits must match). Tested on a Radeon R9 M370X (Metal):
bit-exact with the CPU path. On an old Metal driver each dispatch costs ~0.2 ms so it is overhead-bound (~340 dispatches/token);
on NVIDIA/Vulkan the same code is expected to be bandwidth-bound. MoE on GPU: not yet (falls back to CPU).

See `deploy/` for api.llmfa.st (systemd + Caddy TLS).

## Deploying on a multi-socket Xeon (e.g. dual E5-2660 v2, DDR3, AVX only)

- The engine auto-selects the AVX path on Ivy/Sandy Bridge (no AVX2/FMA) — ~90% of AVX2 speed.
- Threads are pinned and weights are quantized by the thread that will stream them, so each socket reads its own memory (first-touch NUMA).
- Try `THREADS=<physical cores>` (hyperthreads rarely help a bandwidth-bound decode) and compare.
- Expected: Qwen3-30B-A3B at Q4 ≈ 35–40 tok/s single stream, 150–250 tok/s aggregate.
