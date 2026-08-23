package main

import (
	"encoding/json"
	"os"
	"sort"
	"sync"
	"time"
)

// RequestRecord is one metered request — the unit of billing and of every dashboard number.
type RequestRecord struct {
	ID               string    `json:"id"`
	At               time.Time `json:"at"`
	Key              string    `json:"key"`
	Model            string    `json:"model"`
	Engine           string    `json:"engine"`
	PromptTokens     int       `json:"prompt_tokens"`
	CachedTokens     int       `json:"cached_tokens"`
	CompletionTokens int       `json:"completion_tokens"`
	TTFTms           float64   `json:"ttft_ms"`     // time to first token
	DurationMs       float64   `json:"duration_ms"` // whole request
	TokPerSec        float64   `json:"tok_per_sec"` // completion tokens / generation time
	EarningsUSD      float64   `json:"earnings_usd"`
	Error            string    `json:"error,omitempty"`
}

type Benchmark struct {
	ID           string    `json:"id"`
	At           time.Time `json:"at"`
	Engine       string    `json:"engine"`
	Model        string    `json:"model"`
	Concurrency  int       `json:"concurrency"`
	Requests     int       `json:"requests"`
	AvgTTFTms    float64   `json:"avg_ttft_ms"`
	AvgTokPerSec float64   `json:"avg_tok_per_sec"` // per stream
	AggTokPerSec float64   `json:"agg_tok_per_sec"` // whole server, this is what sizes the fleet
	Errors       int       `json:"errors"`
	LastError    string    `json:"last_error,omitempty"`
}

type Store struct {
	mu         sync.Mutex
	path       string
	Records    []RequestRecord `json:"records"`
	Benchmarks []Benchmark     `json:"benchmarks"`
	Keys       map[string]bool `json:"keys"`
	Models     []ModelEntry    `json:"models"`
}

func LoadStore(path string) *Store {
	s := &Store{path: path, Keys: map[string]bool{}}
	if b, err := os.ReadFile(path); err == nil {
		_ = json.Unmarshal(b, s)
	}
	if len(s.Keys) == 0 {
		s.Keys["dev-key"] = true
	}
	return s
}

func (s *Store) save() {
	b, _ := json.MarshalIndent(s, "", " ")
	_ = os.WriteFile(s.path, b, 0o644)
}

func (s *Store) ValidKey(k string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.Keys[k]
}

func (s *Store) AddKey(k string) {
	s.mu.Lock()
	s.Keys[k] = true
	s.save()
	s.mu.Unlock()
}

func (s *Store) Record(r RequestRecord) {
	s.mu.Lock()
	s.Records = append(s.Records, r)
	if len(s.Records) > 100000 {
		s.Records = s.Records[len(s.Records)-100000:]
	}
	s.save()
	s.mu.Unlock()
}

func (s *Store) AddBenchmark(b Benchmark) {
	s.mu.Lock()
	s.Benchmarks = append(s.Benchmarks, b)
	s.save()
	s.mu.Unlock()
}

// Summary is what the admin dashboard renders.
type Summary struct {
	Window           string             `json:"window"`
	Requests         int                `json:"requests"`
	Errors           int                `json:"errors"`
	PromptTokens     int                `json:"prompt_tokens"`
	CompletionTokens int                `json:"completion_tokens"`
	EarningsUSD      float64            `json:"earnings_usd"`
	AvgTTFTms        float64            `json:"avg_ttft_ms"`
	AvgTokPerSec     float64            `json:"avg_tok_per_sec"`
	TokensPerDayRate float64            `json:"tokens_per_day_rate"` // extrapolated from the window
	ByModel          map[string]int     `json:"tokens_by_model"`
	ByKey            map[string]float64 `json:"earnings_by_key"`
	Hourly           []HourBucket       `json:"hourly"`
	Recent           []RequestRecord    `json:"recent"`
}

type HourBucket struct {
	Hour     string  `json:"hour"`
	Tokens   int     `json:"tokens"`
	Requests int     `json:"requests"`
	Earnings float64 `json:"earnings_usd"`
}

func (s *Store) Summarize(window time.Duration) Summary {
	s.mu.Lock()
	defer s.mu.Unlock()
	now := time.Now()
	since := now.Add(-window)
	sum := Summary{Window: window.String(), ByModel: map[string]int{}, ByKey: map[string]float64{}, Hourly: []HourBucket{}, Recent: []RequestRecord{}}
	buckets := map[string]*HourBucket{}
	var ttft, tps float64
	ok := 0
	for _, r := range s.Records {
		if r.At.Before(since) {
			continue
		}
		sum.Requests++
		if r.Error != "" {
			sum.Errors++
			continue
		}
		ok++
		sum.PromptTokens += r.PromptTokens
		sum.CompletionTokens += r.CompletionTokens
		sum.EarningsUSD += r.EarningsUSD
		ttft += r.TTFTms
		tps += r.TokPerSec
		sum.ByModel[r.Model] += r.PromptTokens + r.CompletionTokens
		sum.ByKey[r.Key] += r.EarningsUSD
		h := r.At.Truncate(time.Hour).Format("2006-01-02T15:00")
		b := buckets[h]
		if b == nil {
			b = &HourBucket{Hour: h}
			buckets[h] = b
		}
		b.Tokens += r.PromptTokens + r.CompletionTokens
		b.Requests++
		b.Earnings += r.EarningsUSD
	}
	if ok > 0 {
		sum.AvgTTFTms = ttft / float64(ok)
		sum.AvgTokPerSec = tps / float64(ok)
	}
	total := float64(sum.PromptTokens + sum.CompletionTokens)
	sum.TokensPerDayRate = total / window.Hours() * 24
	for _, b := range buckets {
		sum.Hourly = append(sum.Hourly, *b)
	}
	sort.Slice(sum.Hourly, func(i, j int) bool { return sum.Hourly[i].Hour < sum.Hourly[j].Hour })
	n := len(s.Records)
	start := n - 50
	if start < 0 {
		start = 0
	}
	for i := n - 1; i >= start; i-- {
		sum.Recent = append(sum.Recent, s.Records[i])
	}
	return sum
}
