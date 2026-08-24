package main

import (
	"bufio"
	"context"
	"encoding/json"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// slowEngine emits a first token after a known delay, so the server-measured TTFT has a
// value we can assert against rather than merely "non-zero".
type slowEngine struct{ ttft time.Duration }

func (e *slowEngine) Name() string  { return "fake" }
func (e *slowEngine) Model() string { return "qwen3-0.6b" }
func (e *slowEngine) Healthy() bool { return true }
func (e *slowEngine) Generate(ctx context.Context, req *ChatRequest, emit func(string), emitReasoning func(string)) (Usage, error) {
	time.Sleep(e.ttft)
	for i := 0; i < 4; i++ {
		emit("tok ")
		time.Sleep(5 * time.Millisecond)
	}
	return Usage{PromptTokens: 27, CompletionTokens: 4, TotalTokens: 31}, nil
}

func testServer(t *testing.T, ttft time.Duration) *Server {
	t.Helper()
	st := LoadStore(filepath.Join(t.TempDir(), "store.json"))
	st.Keys = map[string]bool{"k": true}
	return &Server{store: st, engines: []Engine{&slowEngine{ttft: ttft}}, maxInflight: 4}
}

func post(s *Server, body string) *httptest.ResponseRecorder {
	r := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
	r.Header.Set("Authorization", "Bearer k")
	w := httptest.NewRecorder()
	s.handleChat(w, r)
	return w
}

const oneMsg = `{"model":"qwen3-0.6b","messages":[{"role":"user","content":"hi"}]`

// The playground stopwatch and the dashboard disagreed by the network round trip. The fix is
// for the server to report what it measured, so a client can subtract its own latency.
func TestUsageCarriesServerTiming(t *testing.T) {
	s := testServer(t, 60*time.Millisecond)
	var u Usage
	if err := json.Unmarshal(post(s, oneMsg+"}").Body.Bytes(), &struct{ Usage *Usage }{&u}); err != nil {
		t.Fatal(err)
	}
	if u.Timing == nil {
		t.Fatal("no timing in usage")
	}
	if u.Timing.TTFTms < 50 || u.Timing.TTFTms > 400 {
		t.Errorf("ttft %.0f ms, want ~60", u.Timing.TTFTms)
	}
	if u.Timing.TokPerSec <= 0 || u.Timing.DurationMs < u.Timing.TTFTms {
		t.Errorf("bad timing %+v", *u.Timing)
	}
}

func TestStreamingFinalChunkCarriesTiming(t *testing.T) {
	s := testServer(t, 40*time.Millisecond)
	var last *Usage
	sc := bufio.NewScanner(post(s, oneMsg+`,"stream":true}`).Body)
	for sc.Scan() {
		data, ok := strings.CutPrefix(sc.Text(), "data: ")
		if !ok || data == "[DONE]" {
			continue
		}
		var c ChatChunk
		if json.Unmarshal([]byte(data), &c) == nil && c.Usage != nil {
			last = c.Usage
		}
	}
	if last == nil || last.Timing == nil {
		t.Fatal("streaming response carried no timing")
	}
	if last.Timing.TTFTms < 30 {
		t.Errorf("ttft %.0f ms, want >=40", last.Timing.TTFTms)
	}
}

// What the server reports and what the dashboard records must be the same measurement, or
// the two views drift apart again.
func TestRecordedTTFTMatchesReportedTiming(t *testing.T) {
	s := testServer(t, 50*time.Millisecond)
	var u Usage
	json.Unmarshal(post(s, oneMsg+"}").Body.Bytes(), &struct{ Usage *Usage }{&u})
	s.store.mu.Lock()
	defer s.store.mu.Unlock()
	if len(s.store.Records) != 1 {
		t.Fatalf("recorded %d requests, want 1", len(s.store.Records))
	}
	if got := s.store.Records[0].TTFTms; got != u.Timing.TTFTms {
		t.Errorf("dashboard records %.3f ms, response reports %.3f ms", got, u.Timing.TTFTms)
	}
}

func TestKeyIsNotPersistedInFull(t *testing.T) {
	s := testServer(t, time.Millisecond)
	r := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(oneMsg+"}"))
	r.Header.Set("Authorization", "Bearer fk-0123456789abcdefsecret")
	s.store.Keys["fk-0123456789abcdefsecret"] = true
	s.handleChat(httptest.NewRecorder(), r)
	s.store.mu.Lock()
	defer s.store.mu.Unlock()
	for _, rec := range s.store.Records {
		if strings.Contains(rec.Key, "secret") {
			t.Fatalf("full API key persisted in a request record: %q", rec.Key)
		}
	}
}
