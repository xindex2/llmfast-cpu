package main

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"net/url"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// Models we advertise. Prices are what we charge per 1M tokens (what OpenRouter pays us, before their cut).
// MODELS env overrides: "id:ctx:prompt$/M:completion$/M,..."
var models = []ModelInfo{
	{ID: "qwen3-0.6b", Object: "model", OwnedBy: "llmfast", ContextLength: 4096, PromptPrice: 0.02, OutputPrice: 0.05, CachedPrice: 0.005},
}

func loadModels() {
	spec := os.Getenv("MODELS")
	if spec == "" {
		return
	}
	models = nil
	for _, m := range strings.Split(spec, ",") {
		f := strings.Split(m, ":")
		if len(f) < 4 || len(f) > 5 {
			log.Fatalf("bad MODELS entry %q (want id:ctx:prompt:completion[:cached])", m)
		}
		ctx, _ := strconv.Atoi(f[1])
		pp, _ := strconv.ParseFloat(f[2], 64)
		cp, _ := strconv.ParseFloat(f[3], 64)
		cache := pp / 4
		if len(f) == 5 {
			cache, _ = strconv.ParseFloat(f[4], 64)
		}
		models = append(models, ModelInfo{ID: f[0], Object: "model", OwnedBy: "llmfast", ContextLength: ctx, PromptPrice: pp, OutputPrice: cp, CachedPrice: cache})
	}
}

func modelByID(id string) *ModelInfo {
	for i := range models {
		if models[i].ID == id {
			return &models[i]
		}
	}
	return nil
}

type Server struct {
	store       *Store
	engines     []Engine
	next        atomic.Uint64
	inflight    atomic.Int64
	maxInflight int64 // above this we return 429 immediately: OpenRouter measures queueing as slowness
	adminToken  string
	engMu       sync.RWMutex
	registry    *Registry
}

func (s *Server) addEngine(e Engine) {
	s.engMu.Lock()
	s.engines = append(s.engines, e)
	s.engMu.Unlock()
}

func (s *Server) removeEngineByURL(url string) {
	s.engMu.Lock()
	kept := s.engines[:0]
	for _, x := range s.engines {
		if x.Name() != url {
			kept = append(kept, x)
		}
	}
	s.engines = kept
	s.engMu.Unlock()
}

func (s *Server) removeEngine(e Engine) {
	s.engMu.Lock()
	kept := s.engines[:0]
	for _, x := range s.engines {
		if x != e {
			kept = append(kept, x)
		}
	}
	s.engines = kept
	s.engMu.Unlock()
}

// pick does round-robin across healthy engines serving `model` (engines with unknown model are
// accepted as a fallback). Replace with least-loaded once engines report queue depth.
func (s *Server) pick(model string) (Engine, error) {
	s.engMu.RLock()
	engines := append([]Engine{}, s.engines...)
	s.engMu.RUnlock()
	n := len(engines)
	var fallback Engine
	for i := 0; i < n; i++ {
		e := engines[int(s.next.Add(1))%n]
		if !e.Healthy() {
			continue
		}
		if e.Model() == model {
			return e, nil
		}
		if e.Model() == "" && fallback == nil {
			fallback = e
		}
	}
	if fallback != nil {
		return fallback, nil
	}
	return nil, fmt.Errorf("no running engine for model %s", model)
}

func newID(prefix string) string {
	b := make([]byte, 8)
	rand.Read(b)
	return prefix + "-" + hex.EncodeToString(b)
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(v)
}

func apiError(w http.ResponseWriter, code int, msg string) {
	writeJSON(w, code, map[string]any{"error": map[string]any{"message": msg, "type": "invalid_request_error"}})
}

func bearer(r *http.Request) string {
	return strings.TrimSpace(strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer"))
}

// ---------- /v1 (what OpenRouter and the playground call) ----------

func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, 200, map[string]any{"object": "list", "data": models})
}

func (s *Server) handleChat(w http.ResponseWriter, r *http.Request) {
	key := bearer(r)
	userID, ok, why := s.store.AuthKey(key)
	// The signed-in playground has no raw key to send: keys are only ever shown once, so the
	// browser cannot store one. Fall back to the session cookie, which is HttpOnly+SameSite=Lax,
	// so a cross-site POST cannot spend someone's credit on their behalf.
	if !ok && key == "" {
		if u := s.currentUser(r); u != nil {
			if u.Disabled {
				apiError(w, 403, "account disabled")
				return
			}
			if u.CreditUSD <= 0 {
				writeJSON(w, 402, map[string]any{"error": map[string]any{"message": "insufficient credit, top up to continue", "type": "insufficient_quota"}})
				return
			}
			userID, ok, key = u.ID, true, "session"
		}
	}
	if !ok {
		if strings.HasPrefix(why, "insufficient credit") {
			// 402 is a payment problem, not a provider outage
			writeJSON(w, 402, map[string]any{"error": map[string]any{"message": why, "type": "insufficient_quota"}})
			return
		}
		apiError(w, 401, why)
		return
	}
	var req ChatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		apiError(w, 400, "bad json: "+err.Error())
		return
	}
	m := modelByID(req.Model)
	if m == nil {
		apiError(w, 404, "unknown model "+req.Model)
		return
	}
	if len(req.Messages) == 0 {
		apiError(w, 400, "messages required")
		return
	}
	eng, err := s.pick(req.Model)
	if err != nil {
		apiError(w, 503, err.Error())
		return
	}
	if s.inflight.Load() >= s.maxInflight {
		w.Header().Set("Retry-After", "1")
		writeJSON(w, 429, map[string]any{"error": map[string]any{"message": "at capacity, retry shortly", "type": "rate_limit_error"}})
		return
	}
	s.inflight.Add(1)
	defer s.inflight.Add(-1)

	id := newID("chatcmpl")
	start := time.Now()
	var firstTok time.Time
	rec := RequestRecord{ID: id, At: start, Key: keyPrefix(key), UserID: userID, Model: req.Model, Engine: eng.Name(), StatusCode: 200}
	var full, reasoning strings.Builder

	if req.Stream {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("X-Accel-Buffering", "no")
		flusher, _ := w.(http.Flusher)
		send := func(c ChatChunk) {
			b, _ := json.Marshal(c)
			fmt.Fprintf(w, "data: %s\n\n", b)
			if flusher != nil {
				flusher.Flush()
			}
		}
		send(ChatChunk{ID: id, Object: "chat.completion.chunk", Created: start.Unix(), Model: req.Model,
			Choices: []ChunkChoice{{Delta: Delta{Role: "assistant"}}}})
		// SSE comment keep-alives while a long prompt is prefilling, so the client doesn't time out.
		var mu sync.Mutex
		stopKA := make(chan struct{})
		go func() {
			t := time.NewTicker(5 * time.Second)
			defer t.Stop()
			for {
				select {
				case <-stopKA:
					return
				case <-t.C:
					mu.Lock()
					fmt.Fprint(w, ": keep-alive\n\n")
					if flusher != nil {
						flusher.Flush()
					}
					mu.Unlock()
				}
			}
		}()
		usage, gerr := eng.Generate(r.Context(), &req, func(tok string) {
			if firstTok.IsZero() {
				firstTok = time.Now()
			}
			full.WriteString(tok)
			mu.Lock()
			send(ChatChunk{ID: id, Object: "chat.completion.chunk", Created: start.Unix(), Model: req.Model,
				Choices: []ChunkChoice{{Delta: Delta{Content: tok}}}})
			mu.Unlock()
		}, func(text string) {
			if firstTok.IsZero() {
				firstTok = time.Now() // reasoning counts as the first token for TTFT
			}
			reasoning.WriteString(text)
			mu.Lock()
			send(ChatChunk{ID: id, Object: "chat.completion.chunk", Created: start.Unix(), Model: req.Model,
				Choices: []ChunkChoice{{Delta: Delta{Reasoning: text}}}})
			mu.Unlock()
		})
		close(stopKA)
		mu.Lock()
		defer mu.Unlock()
		s.finish(&rec, m, usage, start, firstTok, gerr)
		stop := "stop"
		if gerr != nil {
			stop = "error"
		}
		finalizeUsage(&usage)
		if he, ok := eng.(*HTTPEngine); ok {
			if tc := he.LastToolCalls(); len(tc) > 0 {
				send(ChatChunk{ID: id, Object: "chat.completion.chunk", Created: start.Unix(), Model: req.Model,
					Choices: []ChunkChoice{{Delta: Delta{ToolCalls: tc}}}})
				stop = "tool_calls"
			}
		}
		send(ChatChunk{ID: id, Object: "chat.completion.chunk", Created: start.Unix(), Model: req.Model,
			Choices: []ChunkChoice{{Delta: Delta{}, FinishReason: &stop}}, Usage: &usage})
		fmt.Fprint(w, "data: [DONE]\n\n")
		if flusher != nil {
			flusher.Flush()
		}
		return
	}

	usage, gerr := eng.Generate(r.Context(), &req, func(tok string) {
		if firstTok.IsZero() {
			firstTok = time.Now()
		}
		full.WriteString(tok)
	}, func(text string) {
		if firstTok.IsZero() {
			firstTok = time.Now()
		}
		reasoning.WriteString(text)
	})
	s.finish(&rec, m, usage, start, firstTok, gerr)
	if gerr != nil {
		apiError(w, 502, gerr.Error())
		return
	}
	finalizeUsage(&usage)
	writeJSON(w, 200, ChatResponse{ID: id, Object: "chat.completion", Created: start.Unix(), Model: req.Model,
		Choices: []Choice{{Message: msgOut(eng, full.String(), reasoning.String()), FinishReason: finishOf(eng)}},
		Usage:   usage})
}

func (s *Server) finish(rec *RequestRecord, m *ModelInfo, u Usage, start, first time.Time, err error) {
	end := time.Now()
	rec.PromptTokens, rec.CompletionTokens = u.PromptTokens, u.CompletionTokens
	rec.DurationMs = float64(end.Sub(start).Microseconds()) / 1000
	if !first.IsZero() {
		rec.TTFTms = float64(first.Sub(start).Microseconds()) / 1000
		if gen := end.Sub(first).Seconds(); gen > 0 {
			rec.TokPerSec = float64(u.CompletionTokens) / gen
		}
	}
	fresh := u.PromptTokens - u.CachedTokens
	rec.CachedTokens = u.CachedTokens
	// A client that hangs up mid-stream is not a provider failure; recording it as one would
	// tank the uptime number OpenRouter routes on.
	if err != nil && (errors.Is(err, context.Canceled) || strings.Contains(err.Error(), "context canceled")) {
		rec.Canceled = true
		err = nil
	}
	rec.EarningsUSD = float64(fresh)/1e6*m.PromptPrice + float64(u.CachedTokens)/1e6*m.CachedPrice + float64(u.CompletionTokens)/1e6*m.OutputPrice
	if err != nil {
		rec.Error = err.Error()
		rec.StatusCode = 502
	}
	s.store.Record(*rec)
	s.store.Charge(rec.UserID, rec.EarningsUSD)
}

// ---------- /admin ----------

func (s *Server) adminAuth(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if bearer(r) != s.adminToken {
			apiError(w, 401, "admin token required")
			return
		}
		next(w, r)
	}
}

func (s *Server) handleStats(w http.ResponseWriter, r *http.Request) {
	from, to, label := periodRange(r.URL.Query())
	sum := s.store.SummarizeRange(from, to, label)
	writeJSON(w, 200, map[string]any{
		"summary":  sum,
		"inflight": s.inflight.Load(),
		"engines":  s.engineStatus(),
		"models":   models,
	})
}

// handleAccountUsage is handleStats scoped to the signed-in customer: same shape, so the
// customer dashboard and the admin dashboard can share rendering code.
func (s *Server) handleAccountUsage(w http.ResponseWriter, r *http.Request) {
	u := s.currentUser(r)
	if u == nil {
		apiError(w, 401, "not signed in")
		return
	}
	from, to, label := periodRange(r.URL.Query())
	writeJSON(w, 200, map[string]any{
		"summary": s.store.SummarizeUser(u.ID, from, to, label),
		"user":    u,
	})
}

// periodRange turns ?period=today|yesterday|7d|30d|custom (or ?hours=N) into a time window.
func periodRange(q url.Values) (time.Time, time.Time, string) {
	now := time.Now()
	from, to, label := now.Add(-24*time.Hour), now, "last 24h"
	switch q.Get("period") {
	case "today":
		from, label = time.Date(now.Year(), now.Month(), now.Day(), 0, 0, 0, 0, now.Location()), "today"
	case "yesterday":
		end := time.Date(now.Year(), now.Month(), now.Day(), 0, 0, 0, 0, now.Location())
		from, to, label = end.AddDate(0, 0, -1), end, "yesterday"
	case "7d":
		from, label = now.AddDate(0, 0, -7), "last 7 days"
	case "30d":
		from, label = now.AddDate(0, 0, -30), "last 30 days"
	case "custom":
		if f, err := time.Parse(time.RFC3339, q.Get("from")); err == nil {
			from = f
		}
		if t, err := time.Parse(time.RFC3339, q.Get("to")); err == nil {
			to = t
		}
		label = from.Format("2006-01-02") + " → " + to.Format("2006-01-02")
	default:
		if h, err := strconv.Atoi(q.Get("hours")); err == nil && h > 0 {
			from, label = now.Add(-time.Duration(h)*time.Hour), fmt.Sprintf("last %dh", h)
		}
	}
	return from, to, label
}

func (s *Server) engineStatus() []map[string]any {
	out := []map[string]any{}
	s.engMu.RLock()
	engines := append([]Engine{}, s.engines...)
	s.engMu.RUnlock()
	for _, e := range engines {
		entry := map[string]any{"name": e.Name(), "healthy": e.Healthy(), "model": e.Model()}
		if he, ok := e.(*HTTPEngine); ok {
			entry["device"] = he.Device()
			entry["loading"] = he.Loading()
			entry["progress"] = he.Progress()
		}
		out = append(out, entry)
	}
	return out
}

func (s *Server) handleKeys(w http.ResponseWriter, r *http.Request) {
	if r.Method == "POST" {
		k := "fk-" + newID("")[1:]
		s.store.AddKey(k)
		writeJSON(w, 200, map[string]string{"key": k})
		return
	}
	s.store.mu.Lock()
	keys := []string{}
	for k := range s.store.Keys {
		keys = append(keys, k)
	}
	s.store.mu.Unlock()
	writeJSON(w, 200, map[string]any{"keys": keys})
}

func (s *Server) handleBenchmarks(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		s.store.mu.Lock()
		b := s.store.Benchmarks
		s.store.mu.Unlock()
		writeJSON(w, 200, map[string]any{"benchmarks": b})
		return
	}
	var cfg struct {
		Model       string `json:"model"`
		Concurrency int    `json:"concurrency"`
		Requests    int    `json:"requests"`
		MaxTokens   int    `json:"max_tokens"`
		Prompt      string `json:"prompt"`
	}
	json.NewDecoder(r.Body).Decode(&cfg)
	if cfg.Concurrency <= 0 {
		cfg.Concurrency = 4
	}
	if cfg.Requests <= 0 {
		cfg.Requests = cfg.Concurrency * 2
	}
	if cfg.MaxTokens <= 0 {
		cfg.MaxTokens = 128
	}
	if cfg.Model == "" {
		cfg.Model = models[0].ID
	}
	if cfg.Prompt == "" {
		cfg.Prompt = "Write a detailed explanation of how transformers work."
	}
	eng, err := s.pick(cfg.Model)
	if err != nil {
		apiError(w, 503, err.Error())
		return
	}
	b := s.runBenchmark(r.Context(), eng, cfg.Model, cfg.Concurrency, cfg.Requests, cfg.MaxTokens, cfg.Prompt)
	s.store.AddBenchmark(b)
	writeJSON(w, 200, b)
}

func (s *Server) runBenchmark(ctx context.Context, eng Engine, model string, conc, n, maxTok int, prompt string) Benchmark {
	b := Benchmark{ID: newID("bench"), At: time.Now(), Engine: eng.Name(), Model: model, Concurrency: conc, Requests: n}
	var mu sync.Mutex
	var ttft, tps float64
	var ttfts []float64
	totalTok := 0
	sem := make(chan struct{}, conc)
	var wg sync.WaitGroup
	start := time.Now()
	for i := 0; i < n; i++ {
		wg.Add(1)
		sem <- struct{}{}
		go func() {
			defer wg.Done()
			defer func() { <-sem }()
			req := &ChatRequest{Model: model, MaxTokens: maxTok, Messages: []Message{{Role: "user", Content: jsonString(prompt)}}}
			t0 := time.Now()
			var first time.Time
			u, err := eng.Generate(ctx, req, func(string) {
				if first.IsZero() {
					first = time.Now()
				}
			}, func(string) {
				if first.IsZero() {
					first = time.Now()
				}
			})
			mu.Lock()
			defer mu.Unlock()
			if err != nil {
				b.Errors++
				b.LastError = err.Error()
				return
			}
			ms := float64(first.Sub(t0).Microseconds()) / 1000
			ttft += ms
			ttfts = append(ttfts, ms)
			if g := time.Since(first).Seconds(); g > 0 {
				tps += float64(u.CompletionTokens) / g
			}
			totalTok += u.CompletionTokens
		}()
	}
	wg.Wait()
	okN := n - b.Errors
	if okN > 0 {
		b.AvgTTFTms = ttft / float64(okN)
		b.AvgTokPerSec = tps / float64(okN)
	}
	if len(ttfts) > 0 {
		// p50/p95 under the requested concurrency: what OpenRouter's latency column shows
		sort.Float64s(ttfts)
		b.P50TTFTms = ttfts[len(ttfts)*50/100]
		b.P95TTFTms = ttfts[min(len(ttfts)-1, len(ttfts)*95/100)]
	}
	b.AggTokPerSec = float64(totalTok) / time.Since(start).Seconds()
	return b
}

// msgOut builds the assistant message, attaching tool calls the engine reported.
func msgOut(eng Engine, content, reasoning string) Message {
	m := Message{Role: "assistant", Content: jsonString(content), Reasoning: reasoning}
	if he, ok := eng.(*HTTPEngine); ok {
		if tc := he.LastToolCalls(); len(tc) > 0 {
			m.ToolCalls = tc
			m.Content = nil
		}
	}
	return m
}

func finishOf(eng Engine) string {
	if he, ok := eng.(*HTTPEngine); ok {
		if f := he.LastFinish(); f != "" {
			return f
		}
	}
	return "stop"
}

func finalizeUsage(u *Usage) {
	u.TotalTokens = u.PromptTokens + u.CompletionTokens
	if u.CachedTokens > 0 {
		u.PromptTokensDetails = &PromptTokensDetails{CachedTokens: u.CachedTokens}
	}
}

// ---------- OpenRouter provider document (schema 2.4) ----------
// https://openrouter.ai/docs/guides/community/for-providers — served at GET /models

func (s *Server) handleProviderModels(w http.ResponseWriter, r *http.Request) {
	quant := envOr("QUANTIZATION", "int8")
	country := envOr("DATACENTER_COUNTRY", "US")
	region := envOr("DATACENTER_REGION", "")
	hq := envOr("PROVIDER_SLUG", "llmfast")
	price := func(p float64) string {
		str := strconv.FormatFloat(p/1e6, 'f', 12, 64)
		str = strings.TrimRight(strings.TrimRight(str, "0"), ".")
		if str == "" {
			str = "0"
		}
		return str
	}
	data := []map[string]any{}
	for _, m := range models {
		dc := map[string]any{"country_code": country}
		if region != "" {
			dc["region"] = region
		}
		inputPricing := []map[string]any{{"type": "prompt", "unit": "token", "cost_usd": price(m.PromptPrice)}}
		if m.CachedPrice > 0 {
			inputPricing = append(inputPricing, map[string]any{"type": "cached_prompt", "unit": "token", "cost_usd": price(m.CachedPrice)})
		}
		data = append(data, map[string]any{
			"schema_version":  "2.4",
			"id":              m.ID,
			"name":            "llmfa.st: " + m.ID,
			"hugging_face_id": envOr("HF_ID_"+strings.ToUpper(strings.NewReplacer("-", "_", ".", "_", "/", "_").Replace(m.ID)), ""),
			"created":         1756000000,
			"quantization":    quant,
			"tokenizer":       "Qwen3",
			"description":     m.ID + " served by llmfa.st's CPU inference engine with prefix caching.",
			"input_modalities": []map[string]any{{
				"type":             "text",
				"supported_inputs": map[string]any{"max_context_length": map[string]any{"value": m.ContextLength, "unit": "token"}},
				"pricing":          inputPricing,
				"capacity":         []map[string]any{{"type": "prompt", "unit": "token", "per": "minute", "value": envInt("CAP_PROMPT_TPM", 600000)}},
			}},
			"output_modalities": []map[string]any{{
				"type":       "text",
				"max_length": map[string]any{"value": m.ContextLength, "unit": "token"},
				"streaming":  true,
				"supported_parameters": map[string]any{
					"temperature":     map[string]any{"type": "range", "min": 0, "max": 2},
					"top_p":           map[string]any{"type": "range", "min": 0, "max": 1},
					"max_tokens":      map[string]any{"type": "integer", "min": 1, "max": m.ContextLength, "unit": "token"},
					"stop":            map[string]any{"type": "array", "max_items": 4},
					"tools":           map[string]any{"type": "boolean"},
					"tool_choice":     map[string]any{"type": "enum", "values": []string{"auto", "none", "required"}},
					"response_format": map[string]any{"type": "enum", "values": []string{"text", "json_object"}},
					"reasoning":       map[string]any{"type": "boolean"},
				},
				"pricing": []map[string]any{
					{"type": "completion", "unit": "token", "cost_usd": price(m.OutputPrice)},
					// thinking models emit reasoning tokens; billed at the completion rate
					{"type": "internal_reasoning", "unit": "token", "cost_usd": price(m.OutputPrice)},
				},
				"capacity": []map[string]any{{"type": "completion", "unit": "token", "per": "minute", "value": envInt("CAP_COMPLETION_TPM", 60000)}},
			}},
			"capacity":    []map[string]any{{"type": "request", "unit": "request", "per": "minute", "value": envInt("CAP_RPM", 600)}, {"type": "concurrency", "unit": "request", "value": s.maxInflight}},
			"is_ready":    envOr("IS_READY", "true") == "true",
			"datacenters": []map[string]any{dc},
			"compliance":  map[string]any{"zdr": envOr("ZDR", "true") == "true", "hipaa": false},
			"openrouter":  map[string]any{"slug": hq + "/" + m.ID},
		})
	}
	writeJSON(w, 200, map[string]any{"data": data})
}

func envInt(k string, d int) int {
	if v, err := strconv.Atoi(os.Getenv(k)); err == nil {
		return v
	}
	return d
}

func cors(next http.Handler) http.Handler {
	// Credentialed requests (the session cookie) cannot use a wildcard origin, and echoing
	// any origin back would defeat the point. Only origins named in ALLOWED_ORIGINS get
	// credentials; everyone else keeps the wildcard, which is enough for bearer-key clients.
	allowed := map[string]bool{}
	for _, o := range strings.Split(os.Getenv("ALLOWED_ORIGINS"), ",") {
		if o = strings.TrimSpace(o); o != "" {
			allowed[o] = true
		}
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if o := r.Header.Get("Origin"); o != "" && allowed[o] {
			w.Header().Set("Access-Control-Allow-Origin", o)
			w.Header().Set("Access-Control-Allow-Credentials", "true")
			w.Header().Set("Vary", "Origin")
		} else {
			w.Header().Set("Access-Control-Allow-Origin", "*")
		}
		w.Header().Set("Access-Control-Allow-Headers", "Authorization, Content-Type")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
		if r.Method == "OPTIONS" {
			w.WriteHeader(204)
			return
		}
		next.ServeHTTP(w, r)
	})
}

func main() {
	loadModels()
	addr := envOr("ADDR", ":8080")
	s := &Server{store: LoadStore(envOr("STORE", "llmfast-store.json")), adminToken: envOr("ADMIN_TOKEN", "admin"), maxInflight: int64(envInt("MAX_INFLIGHT", 16))}
	if urls := os.Getenv("ENGINE_URL"); urls != "" {
		for _, u := range strings.Split(urls, ",") {
			s.engines = append(s.engines, NewHTTPEngine(strings.TrimSpace(u)))
		}
	} else if e := NewHTTPEngine("http://127.0.0.1:9000"); e.Healthy() {
		s.engines = append(s.engines, e) // a manually started engine on the default port
	}
	s.registry = NewRegistry(s)
	s.registry.syncPricing()
	// downloads can't survive a restart; running engines are re-adopted if still alive
	s.store.mu.Lock()
	for i := range s.store.Models {
		if s.store.Models[i].Status == "downloading" {
			s.store.Models[i].Status = "error"
			s.store.Models[i].Error = "download interrupted — press retry"
		}
	}
	s.store.mu.Unlock()
	s.registry.Adopt()

	mux := http.NewServeMux()
	serveUI := false
	mux.HandleFunc("GET /api", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, 200, map[string]any{"service": "llmfast-gateway", "endpoints": []string{"GET /health", "GET /models (OpenRouter provider doc)", "GET /v1/models", "POST /v1/chat/completions", "GET /admin/stats", "/admin/keys", "/admin/benchmarks"}, "admin_ui": envOr("ADMIN_URL", "http://localhost:5173")})
	})
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) { writeJSON(w, 200, map[string]string{"status": "ok"}) })
	mux.HandleFunc("GET /v1/models", s.handleModels)
	mux.HandleFunc("GET /models", s.handleProviderModels) // OpenRouter provider document
	mux.HandleFunc("POST /v1/chat/completions", s.handleChat)
	mux.HandleFunc("GET /admin/stats", s.adminAuth(s.handleStats))
	mux.HandleFunc("/admin/keys", s.adminAuth(s.handleKeys))
	mux.HandleFunc("/admin/benchmarks", s.adminAuth(s.handleBenchmarks))
	mux.HandleFunc("/admin/models", s.adminAuth(s.handleModelsAdmin))
	mux.HandleFunc("GET /admin/users", s.adminAuth(s.handleAdminUsers))
	mux.HandleFunc("POST /admin/topup", s.adminAuth(s.handleTopup))
	// customer-facing auth + self-serve keys
	mux.HandleFunc("POST /auth/signup", s.handleSignup)
	mux.HandleFunc("POST /auth/login", s.handleLogin)
	mux.HandleFunc("POST /auth/logout", s.handleLogout)
	mux.HandleFunc("GET /auth/me", s.handleMe)
	mux.HandleFunc("/account/keys", s.handleUserKeys)
	mux.HandleFunc("GET /account/usage", s.handleAccountUsage)
	mux.HandleFunc("POST /admin/models/{id}/{action}", s.adminAuth(s.handleModelAction))

	// Serve the built admin UI (admin/dist) at / when ADMIN_DIR is set, so one process = API + admin.
	if dir := os.Getenv("ADMIN_DIR"); dir != "" {
		// Vite emits content-hashed asset names, so assets can be cached forever, but index.html
		// must never be cached: a stale one points at a bundle that no longer exists (blank page
		// after a deploy, and new pages silently missing until a hard refresh).
		fs := http.FileServer(http.Dir(dir))
		mux.Handle("GET /assets/", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Cache-Control", "public, max-age=31536000, immutable")
			fs.ServeHTTP(w, r)
		}))
		page := func(file string) http.HandlerFunc {
			return func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Cache-Control", "no-store, must-revalidate")
				w.Header().Set("Pragma", "no-cache")
				http.ServeFile(w, r, dir+"/"+file)
			}
		}
		admin, app := page("index.html"), page("app.html")
		mux.HandleFunc("GET /admin/ui", admin)
		mux.HandleFunc("GET /admin/ui/", admin) // trailing slash and deep links
		// The customer console owns /. It routes on the hash, so one file covers every page.
		mux.HandleFunc("GET /{$}", app)
		mux.HandleFunc("GET /favicon.ico", func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(204) })
		serveUI = true
		log.Printf("customer console at /, admin UI at /admin/ui, from %s", dir)
	}

	if !serveUI {
		// No built UI on disk (API-only deploy): / stays the service document.
		mux.HandleFunc("GET /{$}", func(w http.ResponseWriter, r *http.Request) {
			http.Redirect(w, r, "/api", http.StatusTemporaryRedirect)
		})
	}

	handler := cors(mux)
	cert, key := os.Getenv("TLS_CERT"), os.Getenv("TLS_KEY")
	if cert != "" && key != "" {
		log.Printf("llmfa.st gateway listening on %s with TLS (engines: %d)", addr, len(s.engines))
		log.Fatal(http.ListenAndServeTLS(addr, cert, key, handler))
	}
	log.Printf("llmfa.st gateway listening on %s (engines: %d)", addr, len(s.engines))
	log.Fatal(http.ListenAndServe(addr, handler))
}

func envOr(k, d string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return d
}
