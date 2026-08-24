package main

// Server health: the numbers that explain a slow or failing engine before the logs do.
// Everything is read from /proc and statfs, so there is no agent to install and no
// dependency to keep current. On non-Linux (a dev laptop) the fields we cannot read are
// simply absent, and the dashboard renders what it got.

import (
	"bufio"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"time"
)

var startedAt = time.Now()

type Health struct {
	Hostname     string        `json:"hostname"`
	OS           string        `json:"os"`
	Cores        int           `json:"cores"`         // logical
	PhysCores    int           `json:"phys_cores"`    // what the engine defaults THREADS to
	Sockets      int           `json:"sockets"`       // >1 means NUMA effects are in play
	CPUModel     string        `json:"cpu_model"`
	Load1        float64       `json:"load1"`
	Load5        float64       `json:"load5"`
	Load15       float64       `json:"load15"`
	LoadPct      float64       `json:"load_pct"` // load1 as a share of logical cores
	MemTotalMB   float64       `json:"mem_total_mb"`
	MemUsedMB    float64       `json:"mem_used_mb"`
	MemFreeMB    float64       `json:"mem_free_mb"`
	MemPct       float64       `json:"mem_pct"`
	SwapTotalMB  float64       `json:"swap_total_mb"`
	SwapUsedMB   float64       `json:"swap_used_mb"`
	DiskPath     string        `json:"disk_path"`
	DiskTotalGB  float64       `json:"disk_total_gb"`
	DiskUsedGB   float64       `json:"disk_used_gb"`
	DiskFreeGB   float64       `json:"disk_free_gb"`
	DiskPct      float64       `json:"disk_pct"`
	ModelsGB     float64       `json:"models_gb"` // what the checkpoints and weight caches occupy
	UptimeSec    float64       `json:"uptime_sec"`
	GatewayUpSec float64       `json:"gateway_up_sec"`
	GatewayMemMB float64       `json:"gateway_mem_mb"`
	Engines      []EngineProc  `json:"engine_procs"`
	Warnings     []string      `json:"warnings"`
}

type EngineProc struct {
	Model   string  `json:"model"`
	PID     int     `json:"pid"`
	RSSMB   float64 `json:"rss_mb"`
	CPUPct  float64 `json:"cpu_pct"`
	Threads int     `json:"threads"`
}

func (s *Server) health() Health {
	h := Health{OS: runtime.GOOS, Cores: runtime.NumCPU(), GatewayUpSec: time.Since(startedAt).Seconds()}
	h.Hostname, _ = os.Hostname()
	h.PhysCores, h.Sockets, h.CPUModel = cpuTopology()
	if h.PhysCores == 0 {
		h.PhysCores = h.Cores
	}
	h.Load1, h.Load5, h.Load15 = loadAvg()
	if h.Cores > 0 {
		h.LoadPct = h.Load1 / float64(h.Cores) * 100
	}
	h.MemTotalMB, h.MemFreeMB, h.SwapTotalMB, h.SwapUsedMB = memInfo()
	h.MemUsedMB = h.MemTotalMB - h.MemFreeMB
	if h.MemTotalMB > 0 {
		h.MemPct = h.MemUsedMB / h.MemTotalMB * 100
	}

	h.DiskPath = envOr("MODELS_DIR", ".")
	h.DiskTotalGB, h.DiskFreeGB = diskUsage(h.DiskPath)
	h.DiskUsedGB = h.DiskTotalGB - h.DiskFreeGB
	if h.DiskTotalGB > 0 {
		h.DiskPct = h.DiskUsedGB / h.DiskTotalGB * 100
	}
	h.ModelsGB = dirSizeGB(h.DiskPath)
	h.UptimeSec = sysUptime()

	var self EngineProc
	if p := procStat(os.Getpid()); p != nil {
		self = *p
	}
	h.GatewayMemMB = self.RSSMB

	for id, pid := range s.registry.pids() {
		e := EngineProc{Model: id, PID: pid}
		if p := procStat(pid); p != nil {
			e.RSSMB, e.CPUPct, e.Threads = p.RSSMB, p.CPUPct, p.Threads
		}
		h.Engines = append(h.Engines, e)
	}

	// The three conditions that actually break inference on this box, in the order they bite.
	if h.SwapUsedMB > 256 {
		h.Warnings = append(h.Warnings, "swap in use — a model does not fit in RAM, decode will crawl")
	}
	if h.DiskTotalGB > 0 && h.DiskFreeGB < 20 {
		h.Warnings = append(h.Warnings, "under 20 GB free — downloads and weight caches will fail")
	}
	if h.LoadPct > 130 {
		h.Warnings = append(h.Warnings, "load above core count — engines are competing for CPU")
	}
	if h.Sockets > 1 {
		h.Warnings = append(h.Warnings, "dual-socket: decode bandwidth peaks below the full core count, benchmark THREADS")
	}
	return h
}

func (r *Registry) pids() map[string]int {
	out := map[string]int{}
	r.mu.Lock()
	defer r.mu.Unlock()
	for id, c := range r.procs {
		if c.Process != nil {
			out[id] = c.Process.Pid
		}
	}
	return out
}

func loadAvg() (float64, float64, float64) {
	if b, err := os.ReadFile("/proc/loadavg"); err == nil {
		f := strings.Fields(string(b))
		if len(f) >= 3 {
			return atof(f[0]), atof(f[1]), atof(f[2])
		}
	}
	// macOS dev boxes: sysctl is the only portable source.
	if out, err := exec.Command("sysctl", "-n", "vm.loadavg").Output(); err == nil {
		f := strings.Fields(strings.Trim(string(out), "{} \n"))
		if len(f) >= 3 {
			return atof(f[0]), atof(f[1]), atof(f[2])
		}
	}
	return 0, 0, 0
}

// memInfo returns total, available, swap total, swap used — all MB. "Available" rather than
// "free": page cache is reclaimable, and counting it as used would make every healthy box
// look like it is out of memory.
func memInfo() (total, avail, swapTotal, swapUsed float64) {
	f, err := os.Open("/proc/meminfo")
	if err != nil {
		return
	}
	defer f.Close()
	var swapFree float64
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		k, v, ok := strings.Cut(sc.Text(), ":")
		if !ok {
			continue
		}
		kb := atof(strings.Fields(strings.TrimSpace(v))[0]) / 1024 // kB → MB
		switch k {
		case "MemTotal":
			total = kb
		case "MemAvailable":
			avail = kb
		case "SwapTotal":
			swapTotal = kb
		case "SwapFree":
			swapFree = kb
		}
	}
	return total, avail, swapTotal, swapTotal - swapFree
}

// cpuTopology returns physical cores, sockets and the CPU model. Sockets matter: on a
// two-socket box, memory bandwidth does not scale past one socket's worth of threads unless
// the weights are placed on both nodes.
func cpuTopology() (cores, sockets int, model string) {
	f, err := os.Open("/proc/cpuinfo")
	if err != nil {
		return 0, 1, ""
	}
	defer f.Close()
	pkgs, coreIDs := map[string]bool{}, map[string]bool{}
	var pkg string
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		k, v, ok := strings.Cut(sc.Text(), ":")
		if !ok {
			continue
		}
		k, v = strings.TrimSpace(k), strings.TrimSpace(v)
		switch k {
		case "model name":
			model = v
		case "physical id":
			pkg = v
			pkgs[v] = true
		case "core id":
			coreIDs[pkg+"/"+v] = true
		}
	}
	if len(pkgs) == 0 {
		return 0, 1, model
	}
	return len(coreIDs), len(pkgs), model
}

func diskUsage(path string) (totalGB, freeGB float64) {
	var st syscall.Statfs_t
	// MODELS_DIR may not exist yet on a fresh install; the filesystem it will live on is
	// the one we actually care about, so walk up until statfs succeeds.
	for syscall.Statfs(path, &st) != nil {
		i := strings.LastIndex(path, "/")
		if i <= 0 {
			return 0, 0
		}
		path = path[:i]
	}
	bs := float64(st.Bsize)
	const gb = 1 << 30
	return float64(st.Blocks) * bs / gb, float64(st.Bavail) * bs / gb
}

func dirSizeGB(path string) float64 {
	var total int64
	// One level of recursion is enough: models/<id>/<files>. Walking deeper on a directory
	// holding tens of GB of checkpoints costs more than the number is worth.
	entries, err := os.ReadDir(path)
	if err != nil {
		return 0
	}
	for _, e := range entries {
		p := path + "/" + e.Name()
		if !e.IsDir() {
			if fi, err := e.Info(); err == nil {
				total += fi.Size()
			}
			continue
		}
		sub, err := os.ReadDir(p)
		if err != nil {
			continue
		}
		for _, f := range sub {
			if fi, err := f.Info(); err == nil && !f.IsDir() {
				total += fi.Size()
			}
		}
	}
	return float64(total) / (1 << 30)
}

func sysUptime() float64 {
	if b, err := os.ReadFile("/proc/uptime"); err == nil {
		if f := strings.Fields(string(b)); len(f) > 0 {
			return atof(f[0])
		}
	}
	return 0
}

// procStat reads RSS, thread count and lifetime CPU share for one pid.
func procStat(pid int) *EngineProc {
	b, err := os.ReadFile("/proc/" + strconv.Itoa(pid) + "/stat")
	if err != nil {
		return nil
	}
	// comm can contain spaces and parentheses; everything after the last ')' is fixed-width.
	i := strings.LastIndex(string(b), ")")
	if i < 0 {
		return nil
	}
	f := strings.Fields(string(b)[i+1:])
	if len(f) < 22 {
		return nil
	}
	hz := 100.0 // USER_HZ is 100 on every Linux we ship to
	utime, stime := atof(f[11]), atof(f[12])
	threads := int(atof(f[17]))
	startTicks := atof(f[19])
	rssPages := atof(f[21])
	p := &EngineProc{PID: pid, RSSMB: rssPages * float64(os.Getpagesize()) / (1 << 20), Threads: threads}
	if up := sysUptime(); up > 0 {
		if alive := up - startTicks/hz; alive > 0 {
			p.CPUPct = (utime + stime) / hz / alive * 100
		}
	}
	return p
}

func atof(s string) float64 {
	v, _ := strconv.ParseFloat(s, 64)
	return v
}
