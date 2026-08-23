//! Qwen3 dense decoder: config, weights, and the per-token forward pass.
//!
//! Per layer:  x ──RMSNorm──► Q,K,V proj ──per-head RMSNorm (Qwen3-specific)──► RoPE
//!              ──GQA attention over KV cache──► O proj ──► + x
//!             x ──RMSNorm──► gate,up proj ──SiLU(gate)*up──► down proj ──► + x
//! Then final RMSNorm and logits = embedding matrix · x (weights are tied in the 0.6B model).

use crate::kernels::*;
use crate::safetensors::SafeTensors;

/// A linear layer's weights in whichever format we're running.
pub enum Weight {
    Bf16 { w: Vec<u16>, n: usize, k: usize },
    Q8(QMat),
    Q4(Q4Mat),
}

impl Weight {
    fn load(st: &SafeTensors, name: &str, quant: &str) -> Weight {
        let info = st.info(name);
        let (n, k) = (info.shape[0], info.shape[1]);
        let w = st.bf16(name);
        match quant {
            "q8" => Weight::Q8(QMat::from_bf16(&w, n, k)),
            "q4" => Weight::Q4(Q4Mat::from_bf16(&w, n, k)),
            _ => Weight::Bf16 { w, n, k },
        }
    }

    #[inline]
    pub fn matvec(&self, x: &[f32], y: &mut [f32]) {
        match self {
            Weight::Bf16 { w, n, k } => matvec_bf16(w, x, *n, *k, y),
            Weight::Q8(q) => matvec_q8(q, x, y),
            Weight::Q4(q) => matvec_q4(q, x, y),
        }
    }

    #[inline]
    pub fn matmul(&self, xs: &[f32], m: usize, ys: &mut [f32]) {
        match self {
            Weight::Bf16 { w, n, k } => matmul_bf16(w, xs, m, *n, *k, ys),
            Weight::Q8(q) => matmul_q8(q, xs, m, ys),
            Weight::Q4(q) => matmul_q4(q, xs, m, ys),
        }
    }

    pub fn bytes(&self) -> usize {
        match self {
            Weight::Bf16 { w, .. } => w.len() * 2,
            Weight::Q8(q) => q.bytes(),
            Weight::Q4(q) => q.bytes(),
        }
    }

    pub fn params(&self) -> usize {
        match self {
            Weight::Bf16 { n, k, .. } => n * k,
            Weight::Q8(q) => q.n * q.k,
            Weight::Q4(q) => q.n * q.k,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    pub hidden: usize,
    pub intermediate: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub vocab: usize,
    pub rope_theta: f32,
    pub eps: f32,
    pub max_context: usize,
    // Mixture of experts (0 experts = dense MLP)
    pub num_experts: usize,
    pub experts_per_tok: usize,
    pub moe_intermediate: usize,
    pub norm_topk_prob: bool,
}

impl Config {
    pub fn load(dir: &str) -> Config {
        let j: serde_json::Value = serde_json::from_slice(&std::fs::read(format!("{dir}/config.json")).expect("config.json")).unwrap();
        let u = |k: &str| j[k].as_u64().unwrap_or_else(|| panic!("config missing {k}")) as usize;
        let heads = u("num_attention_heads");
        Config {
            hidden: u("hidden_size"),
            intermediate: u("intermediate_size"),
            layers: u("num_hidden_layers"),
            heads,
            kv_heads: u("num_key_value_heads"),
            head_dim: j["head_dim"].as_u64().map(|v| v as usize).unwrap_or(u("hidden_size") / heads),
            vocab: u("vocab_size"),
            rope_theta: j["rope_theta"].as_f64().unwrap_or(10000.0) as f32,
            eps: j["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32,
            max_context: std::env::var("MAX_CONTEXT").ok().and_then(|v| v.parse().ok()).unwrap_or(4096),
            num_experts: j["num_experts"].as_u64().unwrap_or(0) as usize,
            experts_per_tok: j["num_experts_per_tok"].as_u64().unwrap_or(0) as usize,
            moe_intermediate: j["moe_intermediate_size"].as_u64().unwrap_or(0) as usize,
            norm_topk_prob: j["norm_topk_prob"].as_bool().unwrap_or(true),
        }
    }
}

pub(crate) struct Layer {
    pub(crate) ln1: Vec<f32>,
    pub(crate) wq: Weight,
    pub(crate) wk: Weight,
    pub(crate) wv: Weight,
    pub(crate) wo: Weight,
    pub(crate) q_norm: Vec<f32>,
    pub(crate) k_norm: Vec<f32>,
    pub(crate) ln2: Vec<f32>,
    pub(crate) mlp: Mlp,
}

pub struct Expert {
    pub(crate) w_gate: Weight,
    pub(crate) w_up: Weight,
    pub(crate) w_down: Weight,
}

pub enum Mlp {
    Dense(Expert),
    Moe { router: Vec<f32>, experts: Vec<Expert> }, // router: [num_experts, hidden] f32
}

pub struct Model {
    pub config: Config,
    pub(crate) inv_freq: Vec<f32>, // head_dim/2 rotary frequencies
    pub(crate) embed: Vec<u16>, // [vocab, hidden]
    pub(crate) layers: Vec<Layer>,
    pub(crate) norm: Vec<f32>,
    pub(crate) lm_head: Weight, // tied to embed in small models; stored separately so it can be quantized
}

/// Per-sequence KV cache, one growable buffer per layer: memory ∝ tokens actually stored.
#[derive(Clone)]
pub struct KvCache {
    k: Vec<Vec<f32>>, // [layer][pos * stride]
    v: Vec<Vec<f32>>,
    stride: usize,
    pub len: usize,
}

impl KvCache {
    pub fn new(cfg: &Config) -> KvCache {
        let stride = cfg.kv_heads * cfg.head_dim;
        KvCache { k: (0..cfg.layers).map(|_| Vec::new()).collect(), v: (0..cfg.layers).map(|_| Vec::new()).collect(), stride, len: 0 }
    }

    pub fn bytes(&self) -> usize {
        self.k.iter().map(|l| l.capacity() * 8).sum()
    }

    /// Drop everything after the first `n` positions.
    pub fn truncate(&mut self, n: usize) {
        for l in 0..self.k.len() {
            self.k[l].truncate(n * self.stride);
            self.v[l].truncate(n * self.stride);
        }
        self.len = n;
    }
}

impl Model {
    /// Bytes of weights streamed per token (what has to live on the device).
    pub fn weight_bytes(&self) -> usize {
        let mut b = self.lm_head.bytes();
        for l in &self.layers {
            b += l.wq.bytes() + l.wk.bytes() + l.wv.bytes() + l.wo.bytes();
            if let Mlp::Dense(e) = &l.mlp {
                b += e.w_gate.bytes() + e.w_up.bytes() + e.w_down.bytes();
            }
        }
        b
    }

    /// True when every MLP is dense (the GPU backend doesn't do MoE yet).
    pub fn layers_mlp_dense(&self) -> bool {
        self.layers.iter().all(|l| matches!(l.mlp, Mlp::Dense(_)))
    }

    pub fn load(dir: &str) -> Model {
        Model::load_with(dir, &std::env::var("QUANT").unwrap_or_else(|_| "q8".into()))
    }

    pub fn load_with(dir: &str, quant: &str) -> Model {
        let t0 = std::time::Instant::now();
        let config = Config::load(dir);
        let st = SafeTensors::open_dir(dir);
        let quant = quant.to_string();
        let mut layers = Vec::with_capacity(config.layers);
        for l in 0..config.layers {
            let p = format!("model.layers.{l}.");
            layers.push(Layer {
                ln1: st.f32(&format!("{p}input_layernorm.weight")),
                wq: Weight::load(&st, &format!("{p}self_attn.q_proj.weight"), &quant),
                wk: Weight::load(&st, &format!("{p}self_attn.k_proj.weight"), &quant),
                wv: Weight::load(&st, &format!("{p}self_attn.v_proj.weight"), &quant),
                wo: Weight::load(&st, &format!("{p}self_attn.o_proj.weight"), &quant),
                q_norm: st.f32(&format!("{p}self_attn.q_norm.weight")),
                k_norm: st.f32(&format!("{p}self_attn.k_norm.weight")),
                ln2: st.f32(&format!("{p}post_attention_layernorm.weight")),
                mlp: if st.has(&format!("{p}mlp.gate.weight")) {
                    let experts = (0..config.num_experts).map(|e| Expert {
                        w_gate: Weight::load(&st, &format!("{p}mlp.experts.{e}.gate_proj.weight"), &quant),
                        w_up: Weight::load(&st, &format!("{p}mlp.experts.{e}.up_proj.weight"), &quant),
                        w_down: Weight::load(&st, &format!("{p}mlp.experts.{e}.down_proj.weight"), &quant),
                    }).collect();
                    if l == 0 {
                        eprintln!("MoE: {} experts, top-{} per token, expert width {}", config.num_experts, config.experts_per_tok, config.moe_intermediate);
                    }
                    Mlp::Moe { router: st.f32(&format!("{p}mlp.gate.weight")), experts }
                } else {
                    Mlp::Dense(Expert {
                        w_gate: Weight::load(&st, &format!("{p}mlp.gate_proj.weight"), &quant),
                        w_up: Weight::load(&st, &format!("{p}mlp.up_proj.weight"), &quant),
                        w_down: Weight::load(&st, &format!("{p}mlp.down_proj.weight"), &quant),
                    })
                },
            });
        }
        let hd = config.head_dim;
        let inv_freq: Vec<f32> = (0..hd / 2).map(|i| 1.0 / config.rope_theta.powf(2.0 * i as f32 / hd as f32)).collect();
        let m = Model {
            inv_freq,
            embed: st.bf16("model.embed_tokens.weight"),
            norm: st.f32("model.norm.weight"),
            lm_head: Weight::load(&st, if st.has("lm_head.weight") { "lm_head.weight" } else { "model.embed_tokens.weight" }, &quant),
            layers,
            config,
        };
        // (total params, total bytes, bytes touched per token — for MoE only the active experts count)
        let ex = |e: &Expert| (e.w_gate.params() + e.w_up.params() + e.w_down.params(), e.w_gate.bytes() + e.w_up.bytes() + e.w_down.bytes());
        let mut params = m.lm_head.params();
        let mut bytes = m.lm_head.bytes();
        let mut active = m.lm_head.bytes();
        for l in &m.layers {
            for w in [&l.wq, &l.wk, &l.wv, &l.wo] {
                params += w.params();
                bytes += w.bytes();
                active += w.bytes();
            }
            match &l.mlp {
                Mlp::Dense(e) => { let (p, b) = ex(e); params += p; bytes += b; active += b; }
                Mlp::Moe { experts, .. } => {
                    for e in experts { let (p, b) = ex(e); params += p; bytes += b; }
                    let (_, b) = ex(&experts[0]);
                    active += b * m.config.experts_per_tok;
                }
            }
        }
        eprintln!("loaded {} layers, {:.2}B params, {:.2} GB in RAM, {:.2} GB streamed per token ({quant}) in {:.1}s",
            m.config.layers, params as f64 / 1e9, bytes as f64 / 1e9, active as f64 / 1e9, t0.elapsed().as_secs_f32());
        m
    }

    /// One token in, logits over the vocabulary out. `cache.len` is the position.
    pub fn forward(&self, token: u32, cache: &mut KvCache) -> Vec<f32> {
        let mut caches = [cache];
        self.forward_multi(&[(token, 0)], &mut caches).pop().unwrap()
    }

    /// Prefill: run `tokens` (positions cache.len..) through the network in one batched pass
    /// per layer, filling the KV cache. Returns logits for the last token only.
    pub fn forward_batch(&self, tokens: &[u32], cache: &mut KvCache) -> Vec<f32> {
        let items: Vec<(u32, usize)> = tokens.iter().map(|&t| (t, 0)).collect();
        let mut caches = [cache];
        self.forward_multi(&items, &mut caches).pop().unwrap()
    }

    /// The general step: `items` are (token, sequence index) pairs, in order; consecutive items
    /// of the same sequence take consecutive positions. Every weight matrix is streamed from
    /// RAM exactly once for the whole batch — this is what makes prefill fast (many tokens of
    /// one sequence) and batched decode efficient (one token each of many sequences).
    /// Returns logits for the last item of each sequence, in order of first appearance.
    pub fn forward_multi(&self, items: &[(u32, usize)], caches: &mut [&mut KvCache]) -> Vec<Vec<f32>> {
        self.forward_impl(items, caches, false)
    }

    /// Like forward_multi but returns logits for EVERY item (needed to verify draft tokens).
    pub fn forward_multi_all(&self, items: &[(u32, usize)], caches: &mut [&mut KvCache]) -> Vec<Vec<f32>> {
        self.forward_impl(items, caches, true)
    }

    fn forward_impl(&self, items: &[(u32, usize)], caches: &mut [&mut KvCache], all_logits: bool) -> Vec<Vec<f32>> {
        let m = items.len();
        let c = &self.config;
        let (h, hd, kvh) = (c.heads, c.head_dim, c.kv_heads);
        let qd = h * hd;
        let kd = kvh * hd;
        let stride = kd;

        // Position of each item and which items are the last of their sequence.
        let mut pos = vec![0usize; m];
        let mut seen = vec![0usize; caches.len()];
        for (j, &(_, sq)) in items.iter().enumerate() {
            pos[j] = caches[sq].len + seen[sq];
            seen[sq] += 1;
            assert!(pos[j] < c.max_context, "context full");
        }
        // Rotary cos/sin per item (same for every layer and head).
        let half = hd / 2;
        let mut rcos = vec![0f32; m * half];
        let mut rsin = vec![0f32; m * half];
        for j in 0..m {
            for i in 0..half {
                let (sn, cs) = (pos[j] as f32 * self.inv_freq[i]).sin_cos();
                rcos[j * half + i] = cs;
                rsin[j * half + i] = sn;
            }
        }
        let mut is_last = vec![false; m];
        let mut order = Vec::new();
        for j in (0..m).rev() {
            let sq = items[j].1;
            if !order.contains(&sq) {
                order.push(sq);
                is_last[j] = true;
            }
        }
        order.reverse();
        for (sq, n) in seen.iter().enumerate() {
            let need = (caches[sq].len + n) * stride;
            for l in 0..c.layers {
                if caches[sq].k[l].len() < need {
                    caches[sq].k[l].resize(need, 0.0);
                    caches[sq].v[l].resize(need, 0.0);
                }
            }
        }

        let mut x = vec![0.0; m * c.hidden];
        for (j, &(t, _)) in items.iter().enumerate() {
            for (d, &b) in self.embed[t as usize * c.hidden..(t as usize + 1) * c.hidden].iter().enumerate() {
                x[j * c.hidden + d] = bf16_to_f32(b);
            }
        }
        let profile = std::env::var("PROFILE").is_ok();
        let mut tm = [0f64; 6]; // qkv, norm+rope+store, attention, o, mlp, head
        let mut tick = std::time::Instant::now();
        let mut lap = |slot: usize, tick: &mut std::time::Instant| {
            if profile {
                tm[slot] += tick.elapsed().as_secs_f64();
                *tick = std::time::Instant::now();
            }
        };
        let mut hn = vec![0.0; m * c.hidden];
        let mut q = vec![0.0; m * qd];
        let mut k = vec![0.0; m * kd];
        let mut v = vec![0.0; m * kd];
        let mut attn = vec![0.0; m * qd];
        let mut o = vec![0.0; m * c.hidden];
        let mut gate = vec![0.0; m * c.intermediate];
        let mut up = vec![0.0; m * c.intermediate];

        for (li, l) in self.layers.iter().enumerate() {
            hn.copy_from_slice(&x);
            for j in 0..m {
                rmsnorm(&mut hn[j * c.hidden..(j + 1) * c.hidden], &l.ln1, c.eps);
            }
            l.wq.matmul(&hn, m, &mut q);
            l.wk.matmul(&hn, m, &mut k);
            l.wv.matmul(&hn, m, &mut v);
            lap(0, &mut tick);
            for j in 0..m {
                let p = pos[j];
                let (cs, sn) = (&rcos[j * half..(j + 1) * half], &rsin[j * half..(j + 1) * half]);
                for hi in 0..h {
                    let qh = &mut q[j * qd + hi * hd..j * qd + (hi + 1) * hd];
                    rmsnorm(qh, &l.q_norm, c.eps);
                    rope_tab(qh, cs, sn);
                }
                for hi in 0..kvh {
                    let kh = &mut k[j * kd + hi * hd..j * kd + (hi + 1) * hd];
                    rmsnorm(kh, &l.k_norm, c.eps);
                    rope_tab(kh, cs, sn);
                }
                let cache = &mut *caches[items[j].1];
                cache.k[li][p * stride..(p + 1) * stride].copy_from_slice(&k[j * kd..(j + 1) * kd]);
                cache.v[li][p * stride..(p + 1) * stride].copy_from_slice(&v[j * kd..(j + 1) * kd]);
            }
            lap(1, &mut tick);
            {
                let kv: Vec<(&[f32], &[f32], usize)> = (0..m).map(|j| {
                    let cache = &*caches[items[j].1];
                    let end = (pos[j] + 1) * stride;
                    (&cache.k[li][..end], &cache.v[li][..end], pos[j])
                }).collect();
                attention_multi(&q, &kv, stride, h, kvh, hd, &mut attn);
            }
            lap(2, &mut tick);
            l.wo.matmul(&attn, m, &mut o);
            for i in 0..m * c.hidden {
                x[i] += o[i];
            }
            lap(3, &mut tick);

            hn.copy_from_slice(&x);
            for j in 0..m {
                rmsnorm(&mut hn[j * c.hidden..(j + 1) * c.hidden], &l.ln2, c.eps);
            }
            match &l.mlp {
                Mlp::Dense(e) => {
                    e.w_gate.matmul(&hn, m, &mut gate);
                    e.w_up.matmul(&hn, m, &mut up);
                    for i in 0..m * c.intermediate {
                        gate[i] = silu(gate[i]) * up[i];
                    }
                    e.w_down.matmul(&gate, m, &mut o);
                    for i in 0..m * c.hidden {
                        x[i] += o[i];
                    }
                }
                Mlp::Moe { router, experts } => self.moe_forward(router, experts, &hn, m, &mut x),
            }
            lap(4, &mut tick);
        }

        for (sq, n) in seen.iter().enumerate() {
            caches[sq].len += n;
        }

        // Logits only for the last item of each sequence, batched through lm_head.
        let lasts: Vec<usize> = (0..m).filter(|&j| all_logits || is_last[j]).collect();
        let mut xl = vec![0.0; lasts.len() * c.hidden];
        for (r, &j) in lasts.iter().enumerate() {
            let row = &mut xl[r * c.hidden..(r + 1) * c.hidden];
            row.copy_from_slice(&x[j * c.hidden..(j + 1) * c.hidden]);
            rmsnorm(row, &self.norm, c.eps);
        }
        let mut logits = vec![0.0; lasts.len() * c.vocab];
        self.lm_head.matmul(&xl, lasts.len(), &mut logits);
        lap(5, &mut tick);
        if profile {
            eprintln!("step m={m}: qkv {:.1}ms | norm/rope/kv {:.1}ms | attn {:.1}ms | o {:.1}ms | mlp {:.1}ms | head {:.1}ms",
                tm[0] * 1e3, tm[1] * 1e3, tm[2] * 1e3, tm[3] * 1e3, tm[4] * 1e3, tm[5] * 1e3);
        }
        logits.chunks(c.vocab).map(|l| l.to_vec()).collect()
    }
}

impl Model {
    /// Mixture of experts for m tokens: route each token to its top-k experts, then run each
    /// expert ONCE on all tokens routed to it (a batched matmul), and scatter the weighted
    /// outputs back. Only the chosen experts' weights are touched — that's the MoE speed win.
    fn moe_forward(&self, router: &[f32], experts: &[Expert], hn: &[f32], m: usize, x: &mut [f32]) {
        let c = &self.config;
        let (ne, k) = (c.num_experts, c.experts_per_tok);
        // routing
        let mut logits = vec![0f32; ne];
        let mut assign: Vec<Vec<(usize, f32)>> = vec![Vec::new(); ne]; // expert -> [(token, weight)]
        for j in 0..m {
            let h = &hn[j * c.hidden..(j + 1) * c.hidden];
            for e in 0..ne {
                logits[e] = router[e * c.hidden..(e + 1) * c.hidden].iter().zip(h).map(|(a, b)| a * b).sum();
            }
            softmax(&mut logits);
            let mut idx: Vec<usize> = (0..ne).collect();
            idx.select_nth_unstable_by(k - 1, |&a, &b| logits[b].total_cmp(&logits[a]));
            let top = &idx[..k];
            let norm: f32 = if c.norm_topk_prob { top.iter().map(|&e| logits[e]).sum() } else { 1.0 };
            for &e in top {
                assign[e].push((j, logits[e] / norm));
            }
        }
        // expert-grouped compute
        let mi = c.moe_intermediate;
        for (e, toks) in assign.iter().enumerate() {
            if toks.is_empty() {
                continue;
            }
            let n = toks.len();
            let mut xin = vec![0f32; n * c.hidden];
            for (r, &(j, _)) in toks.iter().enumerate() {
                xin[r * c.hidden..(r + 1) * c.hidden].copy_from_slice(&hn[j * c.hidden..(j + 1) * c.hidden]);
            }
            let mut g = vec![0f32; n * mi];
            let mut u = vec![0f32; n * mi];
            let ex = &experts[e];
            ex.w_gate.matmul(&xin, n, &mut g);
            ex.w_up.matmul(&xin, n, &mut u);
            for i in 0..n * mi {
                g[i] = silu(g[i]) * u[i];
            }
            let mut out = vec![0f32; n * c.hidden];
            ex.w_down.matmul(&g, n, &mut out);
            for (r, &(j, w)) in toks.iter().enumerate() {
                let xr = &mut x[j * c.hidden..(j + 1) * c.hidden];
                for d in 0..c.hidden {
                    xr[d] += w * out[r * c.hidden + d];
                }
            }
        }
    }
}

/// Temperature + top-p sampling. temperature 0 → greedy.
pub struct Sampler {
    pub temperature: f32,
    pub top_p: f32,
    rng: u64,
}

impl Sampler {
    pub fn new(temperature: f32, top_p: f32, seed: u64) -> Sampler {
        Sampler { temperature, top_p, rng: seed | 1 }
    }

    fn next_f32(&mut self) -> f32 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        (self.rng >> 40) as f32 / (1u64 << 24) as f32
    }

    pub fn sample(&mut self, logits: &mut [f32]) -> u32 {
        if self.temperature <= 0.0 {
            return (0..logits.len()).max_by(|&a, &b| logits[a].total_cmp(&logits[b])).unwrap() as u32;
        }
        // Keep only the top-K logits (O(n) selection), then do temperature/top-p on those.
        // With top_p <= 0.95 the tail beyond K=256 carries negligible mass.
        const K: usize = 256;
        let mut idx: Vec<usize> = (0..logits.len()).collect();
        if idx.len() > K {
            idx.select_nth_unstable_by(K, |&a, &b| logits[b].total_cmp(&logits[a]));
            idx.truncate(K);
        }
        idx.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]));
        let max = logits[idx[0]];
        let inv_t = 1.0 / self.temperature;
        let mut probs: Vec<f32> = idx.iter().map(|&t| ((logits[t] - max) * inv_t).exp()).collect();
        let sum: f32 = probs.iter().sum();
        probs.iter_mut().for_each(|p| *p /= sum);
        let mut cum = 0.0;
        let mut cutoff = probs.len();
        for (i, p) in probs.iter().enumerate() {
            cum += p;
            if cum >= self.top_p {
                cutoff = i + 1;
                break;
            }
        }
        let r = self.next_f32() * cum;
        let mut acc = 0.0;
        for i in 0..cutoff {
            acc += probs[i];
            if acc >= r {
                return idx[i] as u32;
            }
        }
        idx[0] as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f2b(f: f32) -> u16 {
        (f.to_bits() >> 16) as u16
    }

    fn rand_w(n: usize, k: usize, seed: u32) -> Weight {
        let mut st = seed.wrapping_mul(2654435761) | 1;
        let w: Vec<u16> = (0..n * k).map(|_| {
            st ^= st << 13; st ^= st >> 17; st ^= st << 5;
            f2b(((st % 2000) as f32 / 1000.0 - 1.0) * 0.05)
        }).collect();
        Weight::Bf16 { w, n, k }
    }

    /// A tiny random MoE model: checks that routing/batching is self-consistent — running
    /// tokens one at a time must give the same result as running them as one batch.
    fn tiny_moe() -> Model {
        let config = Config { hidden: 64, intermediate: 128, layers: 2, heads: 4, kv_heads: 2, head_dim: 16, vocab: 100,
            rope_theta: 10000.0, eps: 1e-6, max_context: 64, num_experts: 8, experts_per_tok: 2, moe_intermediate: 32, norm_topk_prob: true };
        let hd = config.head_dim;
        let layers = (0..config.layers).map(|l| Layer {
            ln1: vec![1.0; config.hidden],
            wq: rand_w(config.heads * hd, config.hidden, 1 + l as u32 * 10), wk: rand_w(config.kv_heads * hd, config.hidden, 2 + l as u32 * 10),
            wv: rand_w(config.kv_heads * hd, config.hidden, 3 + l as u32 * 10), wo: rand_w(config.hidden, config.heads * hd, 4 + l as u32 * 10),
            q_norm: vec![1.0; hd], k_norm: vec![1.0; hd], ln2: vec![1.0; config.hidden],
            mlp: Mlp::Moe {
                router: (0..config.num_experts * config.hidden).map(|i| ((i * 37 % 101) as f32 / 101.0 - 0.5) * 0.2).collect(),
                experts: (0..config.num_experts).map(|e| Expert {
                    w_gate: rand_w(config.moe_intermediate, config.hidden, 100 + e as u32 + l as u32 * 50),
                    w_up: rand_w(config.moe_intermediate, config.hidden, 200 + e as u32 + l as u32 * 50),
                    w_down: rand_w(config.hidden, config.moe_intermediate, 300 + e as u32 + l as u32 * 50),
                }).collect(),
            },
        }).collect();
        let inv_freq = (0..hd / 2).map(|i| 1.0 / config.rope_theta.powf(2.0 * i as f32 / hd as f32)).collect();
        let embed = match rand_w(config.vocab, config.hidden, 999) { Weight::Bf16 { w, .. } => w, _ => unreachable!() };
        Model { lm_head: Weight::Bf16 { w: embed.clone(), n: config.vocab, k: config.hidden }, embed, norm: vec![1.0; config.hidden], layers, inv_freq, config }
    }

    #[test]
    fn moe_batched_matches_sequential() {
        let m = tiny_moe();
        let toks = [3u32, 17, 42, 7, 99, 5];
        let mut c1 = KvCache::new(&m.config);
        let mut last = Vec::new();
        for &t in &toks {
            last = m.forward(t, &mut c1);
        }
        let mut c2 = KvCache::new(&m.config);
        let batched = m.forward_batch(&toks, &mut c2);
        assert_eq!(c1.len, c2.len);
        for (a, b) in last.iter().zip(&batched) {
            assert!((a - b).abs() < 1e-3, "{a} vs {b}");
        }
        // and two independent sequences decoded together match decoding them alone
        let mut ca = KvCache::new(&m.config);
        let mut cb = KvCache::new(&m.config);
        let la = m.forward(11, &mut ca);
        let lb = m.forward(22, &mut cb);
        let mut ca2 = KvCache::new(&m.config);
        let mut cb2 = KvCache::new(&m.config);
        let both = m.forward_multi(&[(11, 0), (22, 1)], &mut [&mut ca2, &mut cb2]);
        for (a, b) in la.iter().zip(&both[0]) { assert!((a - b).abs() < 1e-3); }
        for (a, b) in lb.iter().zip(&both[1]) { assert!((a - b).abs() < 1e-3); }
    }
}
