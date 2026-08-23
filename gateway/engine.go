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
	Generate(ctx context.Context, req *ChatRequest, emit func(token string)) (Usage, error)
	Healthy() bool
}

// ---------- http engine: forwards to any OpenAI-compatible engine (our Rust engine) ----------

type HTTPEngine struct {
	url    string
	client *http.Client
	model  string
	device string
}

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

func (h *HTTPEngine) Healthy() bool {
	c := &http.Client{Timeout: 2 * time.Second}
	resp, err := c.Get(h.url + "/health")
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	if h.model == "" {
		var hs struct {
			Model  string `json:"model"`
			Device string `json:"device"`
		}
		if json.NewDecoder(resp.Body).Decode(&hs) == nil {
			h.model = hs.Model
			h.device = hs.Device
		}
	}
	return resp.StatusCode < 500
}

func (h *HTTPEngine) Generate(ctx context.Context, req *ChatRequest, emit func(string)) (Usage, error) {
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
