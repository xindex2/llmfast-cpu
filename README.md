# Forge (llmfast) — your own LLM inference provider, CPU or GPU

**Install guides:** [CPU VPS (DigitalOcean etc.)](docs/install-cpu-vps.md) · [GPU server (RunPod etc.)](docs/install-gpu-runpod.md) · [domain + TLS](deploy/README.md)

One-liner on a fresh Ubuntu box: `curl -fsSL https://raw.githubusercontent.com/xindex2/llmfast-cpu/main/install.sh | bash`

Own-built inference provider: run big open MoE models (Qwen3-30B-A3B, Qwen3-235B-A22B)
on commodity CPU servers and sell tokens through OpenRouter.

```
forge/
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

## Run

```bash
# 1. engine (our Rust inference engine) — needs a checkpoint dir with config.json, model.safetensors, tokenizer.json
cd engine && cargo build --release
MODEL=../models/qwen3-0.6b ./target/release/forge-engine                    # serves :9000
MODEL=../models/qwen3-0.6b ./target/release/forge-engine --prompt "hello"   # quick CLI test
MODEL=../models/qwen3-0.6b ./target/release/forge-engine --tokenize "text"  # inspect tokenization

# 2. gateway — talks to one or more engines (ENGINE_URL=http://a:9000,http://b:9000)
cd gateway && go build -o forge-gateway . && ./forge-gateway                 # :8080

# 3. admin UI
cd admin && npm install && npm run dev                                       # http://localhost:5173

# test the API
curl -N localhost:8080/v1/chat/completions -H 'Authorization: Bearer dev-key' \
  -d '{"model":"qwen3-0.6b","stream":true,"messages":[{"role":"user","content":"hi"}]}'
```

Engine env: `DEVICE=auto|cpu|gpu`, `GPU_MEM_MB`, `MAX_CONTEXT` (default 4096; GPU path caps at 2048 for now), `QUANT=q8|bf16` (default q8), `THREADS`, `THINK=1` (Qwen3 thinking mode), `MODEL_NAME`, `PROFILE=1` (per-step timing).
Engine flags: `--prompt "..."` one-shot CLI, `--tokenize "..."`, `--bench` kernel GFLOPS/GB/s.
Gateway env: `ENGINE_URL`, `MODELS=id:ctx:prompt$/M:completion$/M[:cached$/M],...`, `MAX_INFLIGHT` (429 above), `ADMIN_TOKEN`, `STORE`,
provider doc: `QUANTIZATION`, `DATACENTER_COUNTRY/REGION`, `PROVIDER_SLUG`, `CAP_PROMPT_TPM`, `CAP_COMPLETION_TPM`, `CAP_RPM`, `ZDR`, `IS_READY`.
Speculative decoding: n-gram prompt-lookup is on by default (`NGRAM=0` off, `NGRAM_N`, `NGRAM_K`); draft model: `DRAFT_MODEL=../models/qwen3-0.6b SPEC_K=4`.
CPU tuning: `THREADS`, `FORCE_SIMD=0|1|2` (scalar / AVX / AVX2+FMA), `STATIC=0|1` (owner-first NUMA scheduling, default on Linux), `PIN=0` (disable CPU pinning on Linux).

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
