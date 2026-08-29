# Submitting llmfa.st to OpenRouter — the walkthrough

The admin **Launch** page is the source of truth for readiness: it live-checks every
requirement against the running deployment and shows N/M passing. This doc is the order of
operations around it.

## 1. Serve something worth routing to, and measure it honestly

OpenRouter routes real traffic and measures uptime and latency. Do not apply with a model
doing 2 tok/s.

- Pick a model whose `--bench-model` decode speed is respectable for its class
  (`./engine/llmfast-engine --bench-model <dir>` — the same number OpenRouter's users will see).
- **Do not list a reranker or embedding model** (e.g. Qwen3-Reranker-*): OpenRouter provider
  routing is chat completions; a reranker answers "yes"/"no" to a relevance template and will
  read as broken to every user who hits it. Good cheap starters on CPU: Qwen3-4B (~30 tok/s
  on the reference box at q4) or Qwen3-8B (~17 tok/s).
- Set real prices on the Models page. Undercut the cheapest existing provider for the same
  model slightly; you can raise later.

## 2. Fill gateway.env with measured numbers, not placeholders

```
PROVIDER_SLUG=llmfast
DATACENTER_COUNTRY=US
QUANTIZATION=int8            # or int4 — match what the model actually serves
CAP_PROMPT_TPM=...           # from a Benchmarks-page run: prefill tok/s × 60 × safety 0.7
CAP_COMPLETION_TPM=...       # decode tok/s × 60 × MAX_INFLIGHT × 0.7
CAP_RPM=...                  # what a sustained load test survives, not a hope
ZDR=true                     # only if request logging stays off (we store key prefixes only)
```

Restart the gateway; the Launch page should now show every automatic check green.

## 3. The three URLs the application asks for

- Provider document: `https://api.llmfa.st/models` (schema 2.4 — the Launch page validates it)
- Privacy policy: `https://llmfa.st/privacy.html`
- Terms of service: `https://llmfa.st/terms.html`

## 4. Apply

Follow the current instructions at **openrouter.ai/docs — "For Providers"** (the process and
form change; the doc there is authoritative). You will need: the three URLs above, contact
email, payout details (so OpenRouter can pay for inference), and the capacity numbers from
step 2. Expect them to probe the endpoint before and after listing.

## 5. After listing

- **Uptime is the ranking.** Below ~95% success OpenRouter deprioritizes you. The dashboard's
  Uptime card tracks the same statistic — watch it daily at first.
- Keep TTFT p95 sane (dashboard card, server-side). Slow providers get routed around.
- Earnings page tracks revenue per model at your listed prices; compare against the server's
  monthly cost before scaling to more/bigger models.
- Change prices or add models any time — the provider document updates live from the Models
  page; OpenRouter re-reads it.
