package main

// Model registry: add a model by Hugging Face id, download it, price it, and run an engine
// process for it — all from the admin UI.

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

type ModelEntry struct {
	ID            string  `json:"id"`    // served id, e.g. "qwen3-0.6b"
	HFID          string  `json:"hf_id"` // "Qwen/Qwen3-0.6B"
	Dir           string  `json:"dir"`   // local checkpoint directory
	ContextLength int     `json:"context_length"`
	PromptPrice   float64 `json:"prompt_price_per_m"`
	OutputPrice   float64 `json:"completion_price_per_m"`
	CachedPrice   float64 `json:"cached_price_per_m"`
	Quant         string  `json:"quant"`         // q8 | q4 | bf16
	Draft         string  `json:"draft"`         // optional dir of a draft model for speculative decoding
	Device        string  `json:"device"`        // auto | cpu | gpu
	Status        string  `json:"status"`        // downloading | ready | error | running
	Progress      float64 `json:"progress"`      // 0..1 while downloading
	LoadProgress  float64 `json:"load_progress"` // 0..1 while the engine loads weights
	Downloaded    int64   `json:"downloaded"`    // bytes
	TotalBytes    int64   `json:"total_bytes"`   // bytes
	Error         string  `json:"error,omitempty"`
	Port          int     `json:"port"` // engine port when running
	Pid           int     `json:"pid"`
	Params        string  `json:"params,omitempty"`
}

type Registry struct {
	mu      sync.Mutex
	store   *Store
	procs   map[string]*exec.Cmd
	server  *Server
	dir     string
	engine  string
	nextPrt int
}

func NewRegistry(s *Server) *Registry {
	return &Registry{store: s.store, procs: map[string]*exec.Cmd{}, server: s,
		dir: envOr("MODELS_DIR", "../models"), engine: envOr("ENGINE_BIN", "../engine/target/release/llmfast-engine"), nextPrt: 9001}
}

func slugify(hf string) string {
	hf = strings.TrimPrefix(strings.TrimPrefix(hf, "https://huggingface.co/"), "http://huggingface.co/")
	hf = strings.Trim(hf, "/")
	if i := strings.Index(hf, "/tree/"); i > 0 {
		hf = hf[:i]
	}
	return hf
}

func (r *Registry) find(id string) *ModelEntry {
	for i := range r.store.Models {
		if r.store.Models[i].ID == id {
			return &r.store.Models[i]
		}
	}
	return nil
}

func (r *Registry) update(id string, f func(*ModelEntry)) {
	r.store.mu.Lock()
	if m := r.find(id); m != nil {
		f(m)
	}
	r.store.save()
	r.store.mu.Unlock()
	r.syncPricing()
}

// syncPricing rebuilds the served model list (pricing, context) from the registry.
func (r *Registry) syncPricing() {
	r.store.mu.Lock()
	defer r.store.mu.Unlock()
	var out []ModelInfo
	for _, m := range r.store.Models {
		out = append(out, ModelInfo{ID: m.ID, Object: "model", OwnedBy: "llmfast", ContextLength: m.ContextLength,
			PromptPrice: m.PromptPrice, OutputPrice: m.OutputPrice, CachedPrice: m.CachedPrice})
	}
	if len(out) > 0 {
		models = out
	}
}

// Add registers a model and starts downloading it from Hugging Face in the background.
func (r *Registry) Add(e ModelEntry) (*ModelEntry, error) {
	e.HFID = slugify(e.HFID)
	if e.HFID == "" {
		return nil, fmt.Errorf("hf_id required")
	}
	if e.ID == "" {
		e.ID = strings.ToLower(e.HFID[strings.LastIndex(e.HFID, "/")+1:])
	}
	if e.Quant == "" {
		e.Quant = "q8"
	}
	if e.ContextLength == 0 {
		e.ContextLength = 8192
	}
	if e.CachedPrice == 0 {
		e.CachedPrice = e.PromptPrice / 4
	}
	e.Dir = filepath.Join(r.dir, e.ID)
	e.Status = "downloading"
	r.store.mu.Lock()
	if r.find(e.ID) != nil {
		r.store.mu.Unlock()
		return nil, fmt.Errorf("model %s already exists", e.ID)
	}
	r.store.Models = append(r.store.Models, e)
	r.store.save()
	r.store.mu.Unlock()
	r.syncPricing()
	go r.download(e)
	return &e, nil
}

type hfFile struct {
	Path string `json:"path"`
	Size int64  `json:"size"`
	Type string `json:"type"`
}

func (r *Registry) download(e ModelEntry) {
	fail := func(err error) { r.update(e.ID, func(m *ModelEntry) { m.Status = "error"; m.Error = err.Error() }) }
	resp, err := http.Get("https://huggingface.co/api/models/" + e.HFID + "/tree/main")
	if err != nil {
		fail(err)
		return
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		fail(fmt.Errorf("hugging face: %s (gated or private models need HF_TOKEN)", resp.Status))
		return
	}
	var files []hfFile
	if err := json.NewDecoder(resp.Body).Decode(&files); err != nil {
		fail(err)
		return
	}
	want := []hfFile{}
	var total int64
	for _, f := range files {
		keep := f.Path == "config.json" || f.Path == "tokenizer.json" || f.Path == "tokenizer_config.json" ||
			f.Path == "generation_config.json" || f.Path == "model.safetensors.index.json" || strings.HasSuffix(f.Path, ".safetensors")
		if keep {
			want = append(want, f)
			total += f.Size
		}
	}
	if !hasFile(want, "config.json") || !hasFile(want, "tokenizer.json") {
		fail(fmt.Errorf("repo has no config.json/tokenizer.json — not a transformers checkpoint"))
		return
	}
	// Check the architecture BEFORE the multi-gigabyte shards: config.json is a few KB, and
	// learning "unsupported architecture" after a 60 GB download is how this check got written.
	if cr, err := http.Get("https://huggingface.co/" + e.HFID + "/raw/main/config.json"); err == nil {
		var cfg struct {
			Architectures []string `json:"architectures"`
		}
		if json.NewDecoder(cr.Body).Decode(&cfg) == nil && len(cfg.Architectures) > 0 && !strings.HasPrefix(cfg.Architectures[0], "Qwen3") {
			cr.Body.Close()
			fail(fmt.Errorf("unsupported architecture %s — this engine runs the Qwen3 family (dense, MoE, hybrid); nothing was downloaded", cfg.Architectures[0]))
			return
		}
		cr.Body.Close()
	}
	if err := os.MkdirAll(e.Dir, 0o755); err != nil {
		fail(err)
		return
	}
	r.update(e.ID, func(m *ModelEntry) { m.TotalBytes = total })
	var done int64
	for _, f := range want {
		dst := filepath.Join(e.Dir, f.Path)
		if st, err := os.Stat(dst); err == nil && st.Size() == f.Size {
			done += f.Size
			r.update(e.ID, func(m *ModelEntry) { m.Downloaded = done; m.Progress = float64(done) / float64(max64(total, 1)) })
			continue
		}
		req, _ := http.NewRequest("GET", "https://huggingface.co/"+e.HFID+"/resolve/main/"+f.Path, nil)
		if tok := os.Getenv("HF_TOKEN"); tok != "" {
			req.Header.Set("Authorization", "Bearer "+tok)
		}
		res, err := http.DefaultClient.Do(req)
		if err != nil {
			fail(err)
			return
		}
		if res.StatusCode != 200 {
			res.Body.Close()
			fail(fmt.Errorf("download %s: %s", f.Path, res.Status))
			return
		}
		out, err := os.Create(dst + ".part")
		if err != nil {
			res.Body.Close()
			fail(err)
			return
		}
		buf := make([]byte, 1<<20)
		last := time.Now()
		for {
			n, rerr := res.Body.Read(buf)
			if n > 0 {
				if _, werr := out.Write(buf[:n]); werr != nil {
					fail(werr)
					return
				}
				done += int64(n)
				if time.Since(last) > 500*time.Millisecond {
					last = time.Now()
					r.update(e.ID, func(m *ModelEntry) { m.Downloaded = done; m.Progress = float64(done) / float64(max64(total, 1)) })
				}
			}
			if rerr == io.EOF {
				break
			}
			if rerr != nil {
				fail(rerr)
				return
			}
		}
		out.Close()
		res.Body.Close()
		os.Rename(dst+".part", dst)
	}
	r.update(e.ID, func(m *ModelEntry) { m.Status = "ready"; m.Progress = 1; m.Downloaded = done; m.Error = "" })
}

func hasFile(fs []hfFile, name string) bool {
	for _, f := range fs {
		if f.Path == name {
			return true
		}
	}
	return false
}

func max64(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
}

// Start launches an engine process for the model and registers it for routing.
func (r *Registry) Start(id string) error {
	r.store.mu.Lock()
	m := r.find(id)
	if m == nil {
		r.store.mu.Unlock()
		return fmt.Errorf("unknown model")
	}
	e := *m
	r.store.mu.Unlock()
	if e.Status != "ready" && e.Status != "running" && e.Status != "starting" {
		return fmt.Errorf("model is %s", e.Status)
	}
	// "ready" is a stored status, not a live check: the files can be deleted out from under it
	// (disk reclaimed by hand, container reset on an ephemeral disk). Starting an engine on a
	// gutted directory exits 101 with its log unwritable — say what actually happened instead.
	if _, err := os.Stat(filepath.Join(e.Dir, "config.json")); err != nil {
		r.update(id, func(m *ModelEntry) {
			m.Status, m.Error = "error", "model files are missing on disk — remove this entry and download again"
		})
		return fmt.Errorf("model files missing at %s — remove and re-download", e.Dir)
	}
	// Refuse foreign architectures with words, before the engine panics with "exit status 101".
	// The engine speaks the Qwen3 family (dense, MoE, Next hybrids); anything else (gpt-oss,
	// Llama, Mistral, ...) parses wrong at best and panics at first missing tensor at worst.
	if raw, err := os.ReadFile(filepath.Join(e.Dir, "config.json")); err == nil {
		var cfg struct {
			Architectures []string `json:"architectures"`
		}
		if json.Unmarshal(raw, &cfg) == nil && len(cfg.Architectures) > 0 {
			arch := cfg.Architectures[0]
			if !strings.HasPrefix(arch, "Qwen3") {
				msg := fmt.Sprintf("unsupported architecture %s — this engine runs the Qwen3 family (dense, MoE, hybrid)", arch)
				r.update(id, func(m *ModelEntry) { m.Status, m.Error = "error", msg })
				return fmt.Errorf("%s", msg)
			}
		}
	}
	r.mu.Lock()
	if _, running := r.procs[id]; running {
		r.mu.Unlock()
		return fmt.Errorf("already running")
	}
	port := r.nextPrt
	r.nextPrt++
	r.mu.Unlock()

	cmd := exec.Command(r.engine)
	cmd.Env = append(os.Environ(), "MODEL="+e.Dir, "MODEL_NAME="+e.ID, fmt.Sprintf("ADDR=127.0.0.1:%d", port),
		"QUANT="+e.Quant, fmt.Sprintf("MAX_CONTEXT=%d", e.ContextLength))
	if e.Draft != "" {
		cmd.Env = append(cmd.Env, "DRAFT_MODEL="+e.Draft)
	}
	if e.Device != "" {
		cmd.Env = append(cmd.Env, "DEVICE="+e.Device)
	}
	if v := os.Getenv("GPU_MEM_MB"); v != "" {
		cmd.Env = append(cmd.Env, "GPU_MEM_MB="+v)
	}
	logf, _ := os.Create(filepath.Join(e.Dir, "engine.log"))
	cmd.Stdout, cmd.Stderr = logf, logf
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("start engine: %v (ENGINE_BIN=%s)", err, r.engine)
	}
	r.mu.Lock()
	r.procs[id] = cmd
	r.mu.Unlock()
	eng := NewHTTPEngine(fmt.Sprintf("http://127.0.0.1:%d", port))
	eng.model = id
	r.server.addEngine(eng)
	pid := cmd.Process.Pid
	r.update(id, func(m *ModelEntry) { m.Status = "starting"; m.Port = port; m.Pid = pid; m.Error = "" })
	// Big checkpoints quantize for minutes at startup: flip to "running" when /health answers.
	go func() {
		eng := NewHTTPEngine(fmt.Sprintf("http://127.0.0.1:%d", port))
		for i := 0; i < 900; i++ { // up to 30 min
			time.Sleep(2 * time.Second)
			eng.Healthy() // refreshes progress even before it is servable
			if msg := eng.LoadErr(); msg != "" {
				// The load died (bad quant/device combination, OOM, corrupt checkpoint):
				// stop the walking-dead process and surface the engine's own message.
				_ = cmd.Process.Kill()
				r.update(id, func(m *ModelEntry) {
					if m.Status == "starting" {
						m.Status, m.Error, m.Pid, m.Port = "error", msg, 0, 0
					}
				})
				return
			}
			lp := eng.Progress()
			r.update(id, func(m *ModelEntry) {
				if m.Status == "starting" {
					m.LoadProgress = lp
				}
			})
			r.store.mu.Lock()
			cur := ""
			if m := r.find(id); m != nil {
				cur = m.Status
			}
			r.store.mu.Unlock()
			if cur != "starting" {
				return // exited or was stopped
			}
			if eng.Healthy() {
				r.update(id, func(m *ModelEntry) {
					if m.Status == "starting" {
						m.Status = "running"
						m.LoadProgress = 1
					}
				})
				return
			}
		}
	}()
	go func() {
		err := cmd.Wait()
		r.mu.Lock()
		delete(r.procs, id)
		r.mu.Unlock()
		r.server.removeEngine(eng)
		msg := ""
		if err != nil {
			msg = "engine exited: " + err.Error() + " (see engine.log in the model dir)"
		}
		r.update(id, func(m *ModelEntry) {
			if m.Status == "running" || m.Status == "starting" {
				m.Status = "ready"
				m.Port = 0
				m.Error = msg
			}
		})
	}()
	return nil
}

func (r *Registry) Stop(id string) error {
	r.mu.Lock()
	cmd, ok := r.procs[id]
	r.mu.Unlock()
	if ok {
		return cmd.Process.Kill()
	}
	// adopted after a gateway restart: we only know the pid
	r.store.mu.Lock()
	m := r.find(id)
	pid := 0
	if m != nil {
		pid = m.Pid
	}
	r.store.mu.Unlock()
	if pid == 0 {
		return fmt.Errorf("not running")
	}
	if p, err := os.FindProcess(pid); err == nil {
		_ = p.Kill()
	}
	r.server.removeEngineByURL(fmt.Sprintf("http://127.0.0.1:%d", m.Port))
	r.update(id, func(m *ModelEntry) { m.Status = "ready"; m.Port = 0; m.Pid = 0 })
	return nil
}

// Adopt re-attaches engines that a previous gateway process started and that are still alive.
func (r *Registry) Adopt() {
	r.store.mu.Lock()
	entries := append([]ModelEntry{}, r.store.Models...)
	r.store.mu.Unlock()
	for _, m := range entries {
		if m.Status != "running" || m.Port == 0 {
			continue
		}
		eng := NewHTTPEngine(fmt.Sprintf("http://127.0.0.1:%d", m.Port))
		eng.model = m.ID
		if eng.Healthy() {
			r.server.addEngine(eng)
			r.mu.Lock()
			if m.Port >= r.nextPrt {
				r.nextPrt = m.Port + 1
			}
			r.mu.Unlock()
			continue
		}
		r.update(m.ID, func(e *ModelEntry) {
			e.Status = "ready"
			e.Port = 0
			e.Pid = 0
			e.Error = "engine was not running after gateway restart"
		})
	}
}

func (r *Registry) Remove(id string, deleteFiles bool) error {
	_ = r.Stop(id)
	r.store.mu.Lock()
	var dir string
	kept := r.store.Models[:0]
	for _, m := range r.store.Models {
		if m.ID == id {
			dir = m.Dir
			continue
		}
		kept = append(kept, m)
	}
	r.store.Models = kept
	r.store.save()
	r.store.mu.Unlock()
	r.syncPricing()
	if deleteFiles && dir != "" && strings.HasPrefix(filepath.Clean(dir), filepath.Clean(r.dir)) {
		return os.RemoveAll(dir)
	}
	return nil
}

// ---------- HTTP ----------

func (s *Server) handleModelsAdmin(w http.ResponseWriter, r *http.Request) {
	reg := s.registry
	switch r.Method {
	case "GET":
		s.store.mu.Lock()
		list := append([]ModelEntry{}, s.store.Models...)
		s.store.mu.Unlock()
		writeJSON(w, 200, map[string]any{"models": list, "models_dir": reg.dir, "engine_bin": reg.engine})
	case "POST":
		var e ModelEntry
		if err := json.NewDecoder(r.Body).Decode(&e); err != nil {
			apiError(w, 400, err.Error())
			return
		}
		m, err := reg.Add(e)
		if err != nil {
			apiError(w, 400, err.Error())
			return
		}
		writeJSON(w, 200, m)
	case "PUT": // update prices/context/quant/draft
		var e ModelEntry
		if err := json.NewDecoder(r.Body).Decode(&e); err != nil {
			apiError(w, 400, err.Error())
			return
		}
		reg.update(e.ID, func(m *ModelEntry) {
			m.PromptPrice, m.OutputPrice, m.CachedPrice = e.PromptPrice, e.OutputPrice, e.CachedPrice
			if e.ContextLength > 0 {
				m.ContextLength = e.ContextLength
			}
			if e.Quant != "" {
				m.Quant = e.Quant
			}
			m.Draft = e.Draft
			m.Device = e.Device
		})
		writeJSON(w, 200, map[string]string{"status": "ok"})
	default:
		apiError(w, 405, "method not allowed")
	}
}

func (s *Server) handleModelAction(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	var err error
	switch r.PathValue("action") {
	case "start":
		err = s.registry.Start(id)
	case "stop":
		err = s.registry.Stop(id)
	case "retry":
		s.store.mu.Lock()
		m := s.registry.find(id)
		var e ModelEntry
		if m != nil {
			m.Status, m.Error = "downloading", ""
			e = *m
		}
		s.store.mu.Unlock()
		if m == nil {
			err = fmt.Errorf("unknown model")
		} else {
			go s.registry.download(e)
		}
	case "delete":
		err = s.registry.Remove(id, r.URL.Query().Get("files") == "1")
	default:
		err = fmt.Errorf("unknown action")
	}
	if err != nil {
		apiError(w, 400, err.Error())
		return
	}
	writeJSON(w, 200, map[string]string{"status": "ok"})
}
