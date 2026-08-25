package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"
)

// Engine is one inference backend. The gateway routes requests across many of them.
type Engine interface {
	Name() string
	Model() string // model id this engine serves ("" = unknown/any)
	// Generate streams tokens for a request. It must call emit for every token and return
	// prompt/completion token counts. Cancelling ctx aborts generation.
	Generate(ctx context.Context, req *ChatRequest, emit func(token string), emitReasoning func(text string)) (Usage, error)
	// Tool calls and finish reason from the last Generate call (engine-reported).
	Healthy() bool
}

// ---------- http engine: forwards to any OpenAI-compatible engine (our Rust engine) ----------

type HTTPEngine struct {
	url        string
	client     *http.Client
	model      string
	device     string
	lastTools  []ToolCall
	lastFinish string
	loading    bool
	progress   float64
	loadErr    string
	kernel     Kernel
}

// LoadErr is the terminal failure the engine reported while loading ("" when none).
func (h *HTTPEngine) LoadErr() string { return h.loadErr }

func (h *HTTPEngine) Progress() float64 { return h.progress }
func (h *HTTPEngine) Loading() bool     { return h.loading }

func (h *HTTPEngine) LastToolCalls() []ToolCall { return h.lastTools }
func (h *HTTPEngine) LastFinish() string        { return h.lastFinish }

func NewHTTPEngine(url string) *HTTPEngine {
	return &HTTPEngine{url: strings.TrimRight(url, "/"), client: &http.Client{Timeout: 10 * time.Minute}}
}

func (h *HTTPEngine) Name() string  { return h.url }
func (h *HTTPEngine) Model() string { return h.model }
func (h *HTTPEngine) Device() string {
	if h.device == "" {
		h.model = "" // force a /health re-read
		h.Healthy()
	}
	return h.device
}

// Kernel is what the engine reports about the path it is actually executing. Every
// performance question in this project has eventually been "is the fast path even on", and
// answering it meant running a benchmark beside the server and inferring. Now it is on the
// dashboard next to the model it applies to.
type Kernel struct {
	Build      string  `json:"build"` // commit the engine binary was built from
	SimdLevel  int     `json:"simd_level"`
	Int8Decode bool    `json:"int8_decode"`
	Threads    int     `json:"threads"`
	Quant      string  `json:"quant"`
	WeightGB   float64 `json:"weight_gb"`
	KVInt8     bool    `json:"kv_int8"`
	MTPK       int     `json:"mtp_k"`
	Context    int     `json:"context"`
}

func (h *HTTPEngine) Kernel() Kernel { return h.kernel }

func (h *HTTPEngine) Healthy() bool {
	c := &http.Client{Timeout: 2 * time.Second}
	resp, err := c.Get(h.url + "/health")
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	var hs struct {
		Model    string  `json:"model"`
		Device   string  `json:"device"`
		Status   string  `json:"status"`
		Progress float64 `json:"progress"`
		Error    string  `json:"error"`
		Kernel
	}
	if json.NewDecoder(resp.Body).Decode(&hs) == nil {
		if h.model == "" {
			h.model = hs.Model
		}
		if hs.Device != "" {
			h.device = hs.Device
		}
		h.progress = hs.Progress
		h.loading = hs.Status == "loading"
		// A failed load is terminal: the process is alive but will never serve. Without this
		// the "starting" watcher polls a frozen progress bar for 30 minutes.
		if hs.Status == "error" && hs.Error != "" {
			h.loadErr = hs.Error
		}
		if hs.Status == "ok" {
			h.kernel = hs.Kernel
		}
	}
	// A loading engine is not healthy for routing, but it is alive and progressing.
	return resp.StatusCode < 500 && !h.loading
}

func (h *HTTPEngine) Generate(ctx context.Context, req *ChatRequest, emit func(string), emitReasoning func(string)) (Usage, error) {
	upstream := *req
	upstream.Stream = true
	upstream.StreamOptions = &StreamOptions{IncludeUsage: true}
	body, _ := json.Marshal(upstream)
	hr, err := http.NewRequestWithContext(ctx, "POST", h.url+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return Usage{}, err
	}
	hr.Header.Set("Content-Type", "application/json")
	resp, err := h.client.Do(hr)
	if err != nil {
		return Usage{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		return Usage{}, fmt.Errorf("engine returned %s", resp.Status)
	}
	var usage Usage
	completion := 0
	h.lastTools = nil
	h.lastFinish = ""
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			break
		}
		var chunk ChatChunk
		if json.Unmarshal([]byte(data), &chunk) != nil {
			continue
		}
		if chunk.Usage != nil {
			usage = *chunk.Usage
		}
		for _, ch := range chunk.Choices {
			if len(ch.Delta.ToolCalls) > 0 {
				h.lastTools = ch.Delta.ToolCalls
			}
			if ch.FinishReason != nil && *ch.FinishReason != "" {
				h.lastFinish = *ch.FinishReason
			}
			if ch.Delta.Reasoning != "" {
				emitReasoning(ch.Delta.Reasoning)
			}
			if ch.Delta.Content != "" {
				completion++
				emit(ch.Delta.Content)
			}
		}
	}
	if usage.CompletionTokens == 0 && usage.PromptTokens == 0 {
		usage = Usage{PromptTokens: estimateTokens(req), CompletionTokens: completion}
	}
	return usage, sc.Err()
}

// estimateTokens is a cheap ~4 chars/token guess used when the engine doesn't report usage.
func estimateTokens(req *ChatRequest) int {
	n := 0
	for _, m := range req.Messages {
		n += len(m.Content)/4 + 4
	}
	return n
}
